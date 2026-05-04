from __future__ import annotations

from enum import Enum
from typing import ClassVar, Any

from pydantic import BaseModel, Field


class ProjectType(str, Enum):
    PYTHON = "python"
    NODE = "node"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    GENERIC = "generic"


class State(str, Enum):
    INIT = "init"
    ANALYZE = "analyze"
    PLAN = "plan"
    FIX = "fix"
    VERIFY = "verify"
    PR = "pr"
    DONE = "done"
    FAILED = "failed"
    EVOLVE_READ = "evolve_read"
    EVOLVE_OPTIMIZE = "evolve_optimize"
    EVOLVE_VALIDATE = "evolve_validate"
    EVOLVE_APPLY = "evolve_apply"
    EVOLVE_RESTART = "evolve_restart"


class RootCause(BaseModel):
    file: str
    line: int | None = None
    error_type: str
    description: str
    is_dependency_missing: bool = False
    missing_package: str | None = None


class LogAnalysis(BaseModel):
    summary: str
    root_causes: list[RootCause] = Field(default_factory=list)
    project_type: str = "generic"


class PatchDetail(BaseModel):
    file: str
    diff: str
    reason: str
    risk_level: str = "low"


class PatchBundle(BaseModel):
    patches: list[PatchDetail] = Field(default_factory=list)
    overall_risk: str = "low"


class ReviewResult(BaseModel):
    approved: bool
    critique: str
    risk_score: float = 0.0


class JudgeResult(BaseModel):
    valid: bool
    reason: str


class ExecutionPlan(BaseModel):
    strategy: str
    steps: list[str] = Field(default_factory=list)
    expected_files: list[str] = Field(default_factory=list)
    risk_level: str
    confidence: float
    fallback_strategy: str = "general_fix"
    success_criteria: str | None = None
    risk_threshold: float = 0.7
    failure_mode: str | None = None
    must_steps: list[str] | None = None


class EvolutionProposal(BaseModel):
    version: str
    changelog: str
    optimizations: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    full_source: str = ""
    modules: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0

    _MAX_SOURCE_LEN: ClassVar[int] = 500_000

    def model_post_init(self, __context: Any) -> None:
        total = len(self.full_source) + sum(len(v) for v in self.modules.values())
        if total > self._MAX_SOURCE_LEN:
            raise ValueError(f"full_source too long: {total}")


class EvolutionResult(BaseModel):
    success: bool
    from_version: str
    to_version: str
    message: str
    backup_path: str | None = None
    elapsed_sec: float = 0.0