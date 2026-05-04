from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from .models import ExecutionPlan, LogAnalysis, PatchDetail
from .safety import PatchParser
from .utils import setup_logger


class GitOps:
    def __init__(self, repo: str) -> None:
        self.repo = os.path.abspath(repo)
        self._logger = setup_logger(__name__)

    def _run(self, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", self.repo] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def is_repo(self) -> bool:
        return (Path(self.repo) / ".git").is_dir()

    def has_changes(self) -> bool:
        return bool(self._run(["status", "--porcelain"]).stdout.strip())

    def rollback(self) -> None:
        self._run(["reset", "--hard", "HEAD"])
        self._run(["clean", "-fd"])

    def apply(self, diff: str) -> bool:
        clean = PatchParser.extract_diff(diff)
        if not clean:
            return False
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            f.write(clean)
            tmp = f.name
        try:
            if self._run(["apply", "--check", tmp]).returncode == 0:
                return self._run(["apply", "--3way", tmp]).returncode == 0
            self.rollback()
            return False
        finally:
            os.unlink(tmp)

    def clone_to(self, dest: str) -> bool:
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        return (
            subprocess.run(
                ["git", "clone", f"file://{self.repo}", dest],
                capture_output=True,
            ).returncode
            == 0
        )

    def diff_from(self, other: str) -> str:
        return subprocess.run(
            ["git", "-C", other, "diff", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout

    def create_unique_branch(self, base: str = "auto-fix") -> str:
        h = hashlib.md5(f"{os.urandom(4)}{os.getpid()}".encode()).hexdigest()[:8]
        return f"{base}/ci-{h}"

    def push_branch(self, branch: str) -> bool:
        return self._run(["push", "-u", "origin", branch]).returncode == 0

    def delete_branch(self, branch: str) -> bool:
        return self._run(["branch", "-D", branch]).returncode == 0

    def current_commit(self) -> str:
        return self._run(["rev-parse", "--short", "HEAD"]).stdout.strip()

    def default_branch(self) -> str:
        try:
            result = self._run(["rev-parse", "--abbrev-ref", "origin/HEAD"])
            branch = result.stdout.strip()
            if branch and "origin/" in branch:
                return branch.split("origin/", 1)[1]
        except Exception:
            pass
        return "main"


class PRBot:
    def __init__(self, git: GitOps) -> None:
        self.git = git

    def create_pr(
        self,
        analysis: LogAnalysis,
        plan: ExecutionPlan,
        patches: list[PatchDetail],
        verify: str,
        base: str | None = None,
    ) -> tuple[bool, str]:
        from . import __version__, __codename__

        branch = self.git.create_unique_branch("auto-fix")
        title = f"🤖 Auto Fix: {analysis.summary[:80]}"
        body = self._body(analysis, plan, patches, verify)

        if self.git._run(["checkout", "-b", branch]).returncode != 0:
            return False, f"Branch creation failed: {branch}"
        if self.git._run(["add", "."]).returncode != 0:
            self.git.delete_branch(branch)
            return False, "git add failed"
        if self.git._run(["commit", "-m", title]).returncode != 0:
            self.git.delete_branch(branch)
            return False, "commit failed"
        if not self.git.push_branch(branch):
            self.git.delete_branch(branch)
            return False, f"push failed: {branch}"

        base_branch = base or self.git.default_branch()
        if shutil.which("gh"):
            try:
                subprocess.run(
                    ["gh", "pr", "create", "--title", title, "--body", body,
                     "--base", base_branch, "--head", branch],
                    check=True,
                    capture_output=True,
                )
                return True, f"PR created: {branch}"
            except subprocess.CalledProcessError as e:
                return False, f"gh failed: {e.stderr.decode()}"
        return True, f"Branch {branch} pushed. Create PR manually."

    @staticmethod
    def _body(
        analysis: LogAnalysis,
        plan: ExecutionPlan,
        patches: list[PatchDetail],
        verify: str,
    ) -> str:
        from . import __version__, __codename__

        files = "\n".join(f"- `{p.file}`" for p in patches)
        errors = "\n".join(
            f"- **{rc.error_type}** in `{rc.file}`: {rc.description}"
            for rc in analysis.root_causes
        )
        return (
            f"## Root Cause Analysis\n{analysis.summary}\n"
            f"### Errors\n{errors}\n"
            f"## Strategy: {plan.strategy} | Risk: {plan.risk_level} | Confidence: {plan.confidence:.2f}\n"
            f"## Files Changed\n{files}\n"
            f"## Verification\n{verify}\n"
            f"---\n*Generated by LogDoctor Pro v{__version__} {__codename__}*"
        )