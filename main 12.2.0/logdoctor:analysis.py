from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

from .config import Config
from .llm import LLMBase
from .models import ExecutionPlan, LogAnalysis, ProjectType
from .utils import LogSanitizer, setup_logger


class Analyzer:
    def __init__(self, llm: LLMBase) -> None:
        self.llm = llm
        self._logger = setup_logger(__name__ + ".Analyzer")

    @staticmethod
    def _is_valid(a: LogAnalysis) -> bool:
        return bool(a.root_causes) and all(
            rc.file and rc.error_type and rc.description for rc in a.root_causes
        )

    def run(self, log: str, ptype: ProjectType) -> tuple[LogAnalysis | None, str]:
        clean = LogSanitizer.clean(LogSanitizer.extract_error_window(log))
        sys_msg = (
            f"DevOps expert, project: {ptype.value}. "
            "Extract root causes, detect missing dependencies "
            "(set is_dependency_missing=True and missing_package field). Return JSON."
        )
        a, err = self.llm.call(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": clean}],
            LogAnalysis,
        )
        if a and self._is_valid(a):
            return a, ""

        self._logger.info("First analysis invalid, retrying with strict prompt")
        a2, err2 = self.llm.call(
            [
                {"role": "system", "content": (
                    "You are a strict bug analyst. Always include at least one root cause "
                    "with file, error_type, and description. Set is_dependency_missing=true "
                    "and missing_package if import is missing."
                )},
                {"role": "user", "content": f"Error log:\n{clean}\nOutput valid JSON with root_causes array."},
            ],
            LogAnalysis,
        )
        if a2 and self._is_valid(a2):
            return a2, ""
        return None, f"Analyzer failed: {err or err2}"


class Planner:
    PROFILES = {
        "dependency_fix": (["analyze", "install", "fix", "verify"], "low", "general_fix", "rollback", ["fix", "verify"]),
        "syntax_fix":     (["analyze", "fix", "lint", "verify"],   "low", "general_fix", "rollback", ["fix", "verify"]),
        "config_fix":     (["analyze", "fix", "verify"],            "medium", "general_fix", "manual_review", ["fix", "verify"]),
        "logic_fix":      (["analyze", "fix", "verify"],            "medium", "general_fix", "manual_review", ["fix", "verify"]),
        "general_fix":    (["analyze", "fix", "verify"],            "low", "general_fix", "rollback", ["fix", "verify"]),
    }
    ROUTES = {
        "dependency_fix": ["modulenotfound", "importerror", "no module named", "missing dependency"],
        "syntax_fix":     ["syntaxerror", "indentationerror", "invalid syntax"],
        "config_fix":     ["timeout", "connection refused", "permission denied"],
        "logic_fix":      ["assertionerror", "attributeerror", "keyerror", "valueerror"],
    }

    def decide(
        self, analysis: LogAnalysis, memory: "StrategyMemory", log: str, forced: str | None = None
    ) -> ExecutionPlan:
        base = self._detect_strategy(analysis)
        if forced and forced != "general_fix":
            base = forced
        steps, risk, fallback, fail_mode, must = self.PROFILES.get(base, self.PROFILES["general_fix"])
        confidence = 0.6
        risk_threshold = 0.7

        cases = memory.search(log, k=1)
        if cases and not forced:
            best = cases[0]
            sim = best.get("similarity", 0)
            if sim > 0.8:
                base = best.get("strategy", base)
                steps, risk, fallback, fail_mode, must = self.PROFILES.get(base, self.PROFILES["general_fix"])
                confidence = min(0.95, 0.6 + sim * 0.3)
                risk_threshold = 0.5 if sim > 0.85 else 0.7
            else:
                confidence = 0.5 + sim * 0.2

        if risk == "medium" and confidence < 0.7:
            risk_threshold = 0.4

        return ExecutionPlan(
            strategy=base,
            steps=list(steps),
            expected_files=[rc.file for rc in analysis.root_causes],
            risk_level=risk,
            confidence=confidence,
            fallback_strategy=fallback,
            success_criteria=f"All tests pass and no {analysis.summary}",
            risk_threshold=risk_threshold,
            failure_mode=fail_mode,
            must_steps=list(must),
        )

    def route(self, analysis: LogAnalysis) -> str:
        for rc in analysis.root_causes:
            desc = (rc.description + rc.error_type).lower()
            for strat, keywords in self.ROUTES.items():
                if any(kw in desc for kw in keywords):
                    return strat
        return "general_fix"

    @staticmethod
    def _detect_strategy(a: LogAnalysis) -> str:
        for rc in a.root_causes:
            err = rc.error_type.lower()
            if "modulenotfound" in err or "importerror" in err or "missing dependency" in err:
                return "dependency_fix"
            if "syntaxerror" in err or "indentationerror" in err:
                return "syntax_fix"
            if "timeout" in err or "connection" in err:
                return "config_fix"
            if "assertion" in err or "attribute" in err or "keyerror" in err:
                return "logic_fix"
        return "general_fix"


class StrategyMemory:
    def __init__(self, config: Config) -> None:
        self.enabled = config.rag_enable
        self.collection = None
        self._logger = logging.getLogger(__name__ + ".RAG")
        if self.enabled:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=config.rag_path)
                self.collection = client.get_or_create_collection("repair_cases")
            except Exception as e:
                self._logger.warning("RAG init failed: %s", e)
                self.enabled = False

    def add(self, log: str, plan: ExecutionPlan, success: bool) -> None:
        if not self.enabled or not self.collection or not success:
            return
        try:
            self.collection.add(
                documents=[LogSanitizer.clean(log)[:1000]],
                metadatas=[{
                    "strategy": plan.strategy,
                    "risk": plan.risk_level,
                    "confidence": plan.confidence,
                }],
                ids=[hashlib.md5(f"{time.time()}{os.urandom(8)}".encode()).hexdigest()],
            )
        except Exception as e:
            self._logger.debug("RAG add failed: %s", e)

    def search(self, log: str, k: int = 3) -> list[dict]:
        if not self.enabled or not self.collection:
            return []
        try:
            result = self.collection.query(
                query_texts=[LogSanitizer.clean(log)[:1000]],
                n_results=k,
                include=["metadatas", "distances"],
            )
            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]
            return [{**m, "similarity": max(0.0, 1.0 - d)} for m, d in zip(metas, dists)]
        except Exception as e:
            self._logger.debug("RAG search failed: %s", e)
            return []