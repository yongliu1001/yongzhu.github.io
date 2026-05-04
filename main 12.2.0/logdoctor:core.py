python
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, D
from .git_ops import GitOps, PRBot
from .llm import create_llm
from .models import (
    ExecutionPlan,
    LogAnalysis,
    PatchBundle,
    PatchDetail,
    ProjectType,
    State,
)
from .safety import ASTGuard, PatchGuard
from .sandbox import (
    DependencyResolver,
    Sandbox,
    TransactionManager,
    detect_project,
    LANG_SPEC,
)
from .utils import FileLock, LogSanitizer, setup_logger
from .analysis import Analyzer, Planner, StrategyMemory
from .fixer import Fixer


class StateMachine:
    def __init__(self) -> None:
        self.state = State.INIT

    TRANSITIONS: dict[tuple[State, bool], State] = {
        (State.INIT, True):     State.ANALYZE,
        (State.ANALYZE, True):  State.PLAN,
        (State.PLAN, True):     State.FIX,
        (State.FIX, True):      State.VERIFY,
        (State.VERIFY, True):   State.PR,
        (State.VERIFY, False):  State.FAILED,
        (State.PR, True):       State.DONE,
        (State.PR, False):      State.FAILED,
    }

    def next(self, success: bool) -> State:
        self.state = self.TRANSITIONS.get((self.state, success), State.FAILED)
        return self.state


class Metrics:
    KEYS = (
        "runs", "success", "partial_success", "failed",
        "pr_failed", "bootstrap_failed", "manual_review",
    )

    def __init__(self) -> None:
        self.data: dict[str, int] = {k: 0 for k in self.KEYS}

    def inc(self, key: str, n: int = 1) -> None:
        if key in self.data:
            self.data[key] += n
        else:
            logging.getLogger(__name__).warning("Metrics unknown key '%s'", key)

    def report(self) -> str:
        return json.dumps(self.data)


class TracerStatus:
    SUCCESS = "success"
    PARTIAL = "partial_success"
    PR_FAILED = "pr_failed"
    BOOTSTRAP_FAILED = "bootstrap_failed"
    MANUAL_REVIEW = "manual_review"
    FAILED = "failed"


class ExecutionTracer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.rounds: list[dict] = []

    def record_round(
        self,
        round_num: int,
        log: str,
        analysis: LogAnalysis,
        plan: ExecutionPlan,
        patches: list[PatchDetail],
        result: str,
        extra: Optional[dict] = None,
    ) -> None:
        entry: dict[str, Any] = {
            "round": round_num,
            "timestamp": time.ctime(),
            "log_snippet": LogSanitizer.clean(log)[:500],
            "analysis": analysis.model_dump() if analysis else None,
            "plan": plan.model_dump(),
            "patches": [p.model_dump() for p in patches],
            "result": result,
        }
        if extra:
            entry["extra"] = extra
        self.rounds.append(entry)

    def export(self, path: Optional[str] = None) -> None:
        with open(path or self.config.trace_file, "w", encoding="utf-8") as f:
            json.dump(self.rounds, f, indent=2, ensure_ascii=False)

    def summary_line(self, rnd: int, strat: str, result: str) -> str:
        return f"[Round {rnd}] {strat} -> {result}"


class PatchImpactAnalyzer:
    def __init__(self, repo: str) -> None:
        self.repo = repo

    def validate(self, bundle: PatchBundle, analysis: LogAnalysis) -> tuple[bool, str]:
        affected = {p.file for p in bundle.patches}
        roots = {rc.file for rc in analysis.root_causes}
        if not affected & roots:
            return False, "Patch does not touch root cause files"
        return True, "ok"


class LogDoctor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = setup_logger("LogDoctor", config.log_level, config.log_file)
        self.llm = create_llm(config)
        self.analyzer = Analyzer(self.llm)
        self.planner = Planner()
        self.memory = StrategyMemory(config)
        self.fixer = Fixer(self.llm, config)
        self.guard = ASTGuard()
        self.patch_guard = PatchGuard(config)
        self.tracer = ExecutionTracer(config)
        self.metrics = Metrics()
        self.fsm = StateMachine()
        self._history: list[dict] = []

    def run(self, log_path: str, repo_path: str, dry: bool = False, force: bool = False) -> bool:
        self.metrics.inc("runs")
        repo = os.path.abspath(repo_path)
        git = GitOps(repo)

        if not git.is_repo():
            self.logger.error("Not a git repo: %s", repo)
            self.metrics.inc("failed")
            return False
        if git.has_changes() and not force:
            self.logger.error("Dirty working directory. Use --force.")
            self.metrics.inc("failed")
            return False

        ptype = detect_project(repo)
        spec = LANG_SPEC[ptype]
        self.logger.info("Project type: %s", ptype.value)

        log = self._read_log_safe(log_path)

        self.fsm.next(True)
        analysis, err = self.analyzer.run(log, ptype)
        if not analysis:
            self.logger.error("Analysis failed: %s", err)
            self.metrics.inc("failed")
            self.fsm.state = State.FAILED
            return False

        self.logger.info("Root causes: %d", len(analysis.root_causes))
        self.fsm.next(True)
        routed = self.planner.route(analysis)
        self.logger.info("Router -> %s", routed)

        impact = PatchImpactAnalyzer(repo)

        for rnd in range(1, self.config.rounds + 1):
            self.logger.info("=== Round %d/%d ===", rnd, self.config.rounds)
            if rnd > 1:
                git.rollback()

            plan = self.planner.decide(analysis, self.memory, log, forced=routed)
            self.logger.info("Strategy: %s, Steps: %s, Risk: %s", plan.strategy, plan.steps, plan.risk_level)
            rag = self.memory.search(log)
            self.fsm.next(True)

            code_context = self.fixer.extract_code_context(
                [rc.file for rc in analysis.root_causes], repo
            )
            bundle, err = self.fixer.generate_patches(analysis, plan, code_context, ptype)
            if not bundle or not bundle.patches:
                self.logger.warning("Fixer failed: %s", err)
                self.metrics.inc("failed")
                continue

            ok, msg = self.patch_guard.check(bundle)
            if not ok:
                self.logger.warning("Guard rejected: %s", msg)
                self.metrics.inc("failed")
                continue

            ok, msg = impact.validate(bundle, analysis)
            if not ok:
                self.logger.warning("Impact rejected: %s", msg)
                self.metrics.inc("failed")
                continue

            self.tracer.record_round(rnd, log, analysis, plan, bundle.patches, "pending")

            if self._process_patches(rnd, log, analysis, plan, bundle, git, ptype, spec, dry):
                return True

        self.tracer.export()
        self.fsm.state = State.FAILED
        return False

    @staticmethod
    def _read_log_safe(log_path: str) -> str:
        try:
            file_size = os.path.getsize(log_path)
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                if file_size > D.LOG_MAX_BYTES:
                    f.seek(max(0, file_size - D.LOG_MAX_BYTES))
                    f.readline()
                    content = f.read()
                else:
                    content = f.read()
            lines = content.splitlines()
            if len(lines) > D.LOG_MAX_LINES:
                lines = lines[-D.LOG_MAX_LINES:]
            return "\n".join(lines)
        except Exception as e:
            logging.getLogger(__name__).error("Failed to read log: %s", e)
            return ""

    def _process_patches(
        self,
        rnd: int,
        log: str,
        analysis: LogAnalysis,
        plan: ExecutionPlan,
        bundle: PatchBundle,
        git: GitOps,
        ptype: ProjectType,
        spec: dict,
        dry: bool,
    ) -> bool:
        for patch in bundle.patches:
            ctx = "test" if "test" in patch.file.lower() else "project"
            ok, msg = self.guard.check_patch(patch, ptype, self.config, ctx)
            if not ok:
                self.logger.warning("AST guard: %s", msg)
                continue

            review, _ = self.fixer.review_patches(
                analysis, bundle, {p.file: p.reason for p in bundle.patches}
            )
            if not review or not review.approved:
                self.logger.info("Reviewer rejected: %s", review.critique if review else "no review")
                continue

            if dry:
                self.logger.info("[DRY-RUN] %s: %s", patch.file, patch.reason)
                self.tracer.record_round(rnd, log, analysis, plan, [patch], "dry_run")
                self.metrics.inc("partial_success")
                self.tracer.export()
                return True

            if self._execute_patch(rnd, log, analysis, plan, patch, bundle, git, ptype, spec):
                return True
        return False

    def _execute_patch(
        self,
        rnd: int,
        log: str,
        analysis: LogAnalysis,
        plan: ExecutionPlan,
        patch: PatchDetail,
        bundle: PatchBundle,
        git: GitOps,
        ptype: ProjectType,
        spec: dict,
    ) -> bool:
        try:
            lock_path = os.path.join(
                tempfile.gettempdir(),
                f"logdoctor_{hashlib.md5(git.repo.encode()).hexdigest()[:12]}.lock",
            )
            with FileLock(lock_path):
                with Sandbox(git.repo, self.config) as sb:
                    sb.bootstrap_deps(ptype, spec)
                    extra: dict[str, Any] = {"bootstrap": sb.bootstrap_status or "?"}

                    if sb.bootstrap_status and sb.bootstrap_status != "ok":
                        self.logger.error("Bootstrap failed: %s", sb.bootstrap_status)
                        self.metrics.inc("bootstrap_failed")
                        self.tracer.record_round(rnd, log, analysis, plan, [], TracerStatus.BOOTSTRAP_FAILED, extra=extra)
                        return False

                    sb_git = GitOps(sb.path)
                    py_interp = sb.venv_python if ptype == ProjectType.PYTHON else None
                    tx = TransactionManager(git, sb_git, spec, self.config, plan)

                    self.fsm.next(True)
                    ok, tx_msg = tx.execute(patch.diff, analysis, sb, py_interp)
                    if ok:
                        return self._handle_success(rnd, log, analysis, plan, bundle, git, extra)

                    self.logger.warning("Transaction failed: %s", tx_msg)
                    self.metrics.inc("failed")
                    self.tracer.record_round(rnd, log, analysis, plan, [patch], TracerStatus.FAILED, extra=extra)
                    return False
        except Exception as e:
            self.logger.exception("Sandbox error")
            self.metrics.inc("failed")
            self.tracer.record_round(rnd, log, analysis, plan, [], f"exception_{e}")
            return False

    def _handle_success(
        self,
        rnd: int,
        log: str,
        analysis: LogAnalysis,
        plan: ExecutionPlan,
        bundle: PatchBundle,
        git: GitOps,
        extra: dict[str, Any],
    ) -> bool:
        judge_ok, reason = self.fixer.judge_result(analysis, bundle, "success")
        extra["judge"] = reason
        if not judge_ok:
            self.logger.info("Judge rejected: %s", reason)
            self.metrics.inc("failed")
            self.tracer.record_round(rnd, log, analysis, plan, bundle.patches, "judge_rejected", extra=extra)
            return False

        self.memory.add(log, plan, True)
        self._history.append({"strategy": plan.strategy, "result": "success", "round": rnd})
        pr = PRBot(git)
        pr_ok, pr_msg = pr.create_pr(analysis, plan, bundle.patches, "success")
        extra["pr"] = pr_msg
        if not pr_ok:
            self.metrics.inc("pr_failed")
            self.tracer.record_round(rnd, log, analysis, plan, bundle.patches, TracerStatus.PR_FAILED, extra=extra)
            self.logger.error("PR failed: %s", pr_msg)
            return False

        self.metrics.inc("success")
        self.tracer.record_round(rnd, log, analysis, plan, bundle.patches, TracerStatus.SUCCESS, extra=extra)
        self.tracer.export()
        self.fsm.next(True)
        self.fsm.next(True)
        self.logger.info(self.tracer.summary_line(rnd, plan.strategy, "success"))
        return True