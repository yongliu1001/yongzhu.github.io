from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import D, Config
from .llm import LLMBase
from .models import (
    ExecutionPlan,
    JudgeResult,
    LogAnalysis,
    PatchBundle,
    PatchDetail,
    ProjectType,
    ReviewResult,
)
from .safety import ASTGuard, PatchGuard, PatchParser
from .utils import setup_logger


class Fixer:
    def __init__(self, llm: LLMBase, config: Config) -> None:
        self.llm = llm
        self.config = config
        self.ast_guard = ASTGuard()
        self.patch_guard = PatchGuard(config)
        self._logger = setup_logger(__name__ + ".Fixer")

    def generate_patches(
        self,
        analysis: LogAnalysis,
        plan: ExecutionPlan,
        code_context: dict[str, str],
        ptype: ProjectType,
    ) -> tuple[PatchBundle | None, str]:
        system_msg = self._build_system_prompt(ptype)
        user_msg = self._build_user_prompt(analysis, plan, code_context)
        bundle, err = self.llm.call(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            PatchBundle,
        )
        if not bundle or not bundle.patches:
            return None, f"Generation failed: {err or 'empty result'}"

        clean_patches = []
        for p in bundle.patches:
            p.diff = PatchParser.extract_diff(p.diff)
            if p.diff:
                clean_patches.append(p)
        if not clean_patches:
            self._logger.info("All patches empty, retrying with strict prompt")
            bundle, err = self._retry_strict(analysis, plan, code_context, ptype)
            if not bundle:
                return None, f"Strict retry failed: {err}"
            clean_patches = bundle.patches

        bundle.patches = clean_patches
        ok, msg = self.patch_guard.check(bundle)
        if not ok:
            return None, f"PatchGuard: {msg}"

        for patch in bundle.patches:
            ok, msg = self.ast_guard.check_patch(patch, ptype, self.config, "project")
            if not ok:
                return None, f"ASTGuard: {msg}"
        return bundle, ""

    def _retry_strict(self, analysis, plan, code_context, ptype):
        strict_sys = "Output ONLY unified diff format patches inside ```diff blocks. Each must have valid --- and +++ headers."
        user_msg = self._build_user_prompt(analysis, plan, code_context)
        bundle, err = self.llm.call(
            [{"role": "system", "content": strict_sys}, {"role": "user", "content": user_msg}],
            PatchBundle,
        )
        if bundle:
            for p in bundle.patches:
                p.diff = PatchParser.extract_diff(p.diff)
            bundle.patches = [p for p in bundle.patches if p.diff]
            if bundle.patches:
                return bundle, ""
        return None, "strict retry failed"

    def review_patches(
        self,
        analysis: LogAnalysis,
        bundle: PatchBundle,
        code_context: dict[str, str],
    ) -> tuple[ReviewResult | None, str]:
        system_msg = (
            "You are a senior code reviewer. Review patches for correctness, safety, "
            "and best practices. Output risk_score (0-1) and approval decision."
        )
        user_msg = (
            f"## Root Causes\n{json.dumps([rc.model_dump() for rc in analysis.root_causes], indent=2)}\n"
            f"## Patches\n{json.dumps([p.model_dump() for p in bundle.patches], indent=2)}\n"
            f"## Affected Files: {list(code_context.keys())}\n"
            "Review and decide."
        )
        return self.llm.call(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            ReviewResult,
        )

    def judge_result(
        self,
        analysis: LogAnalysis,
        bundle: PatchBundle,
        test_output: str,
    ) -> tuple[JudgeResult | None, str]:
        system_msg = (
            "You are a QA engineer. Judge if the fix resolves the root cause without introducing new errors."
        )
        user_msg = (
            f"## Original Error\n{analysis.summary}\n"
            f"## Test Output\n{test_output}\n"
            f"## Applied Patches\n{json.dumps([p.model_dump() for p in bundle.patches], indent=2)}\n"
            "Is the fix valid?"
        )
        return self.llm.call(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            JudgeResult,
        )

    def extract_code_context(self, error_files: list[str], repo_path: str) -> dict[str, str]:
        context = {}
        for file_path in error_files:
            full_path = Path(repo_path) / file_path
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                context[file_path] = content
            except Exception as e:
                self._logger.warning("Failed to read %s: %s", full_path, e)
        return context

    @staticmethod
    def _build_system_prompt(ptype: ProjectType) -> str:
        return (
            "You are a senior developer. Generate unified diff patches to fix bugs. "
            "Only change what's necessary, follow best practices, output valid diff."
        )

    def _build_user_prompt(self, analysis, plan, code_context):
        parts = [f"## Bug Analysis\n{analysis.summary}\n## Root Causes"]
        for rc in analysis.root_causes:
            parts.append(f"- **{rc.error_type}** in `{rc.file}`: {rc.description}")
        parts.append(f"\n## Strategy: {plan.strategy} (Risk: {plan.risk_level})")
        parts.append("## Code Context")
        for file, content in list(code_context.items())[:5]:
            lines = content.splitlines()
            if len(lines) > D.CODE_MAX_LINES:
                snippet = "\n".join(lines[:D.CODE_MAX_LINES]) + f"\n... ({len(lines)} lines total)"
            else:
                snippet = content
            parts.append(f"\n### {file}\n{snippet}")
        parts.append("\nGenerate patches in ```diff code blocks.")
        return "\n".join(parts)