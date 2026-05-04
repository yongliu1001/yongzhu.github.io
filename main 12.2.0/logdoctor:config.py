from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Defaults:
    CODE_CONTEXT_LINES: int = 50
    CODE_MAX_LINES: int = 200
    ERROR_WINDOW_SIZE: int = 2000
    ERROR_CONTEXT_LINES: int = 30
    DIFF_MIN_LENGTH: int = 10
    LOCK_POLL_INTERVAL: float = 0.5
    SANDBOX_CPU_QUOTA: float = 0.5
    SANDBOX_NPROC: int = 100
    EVOLVE_BACKUP_KEEP: int = 5
    EVOLVE_SYNTAX_TIMEOUT: int = 10
    EVOLVE_SANDBOX_TIMEOUT: int = 60
    LOG_MAX_LINES: int = 5000
    LOG_MAX_BYTES: int = 5 * 1024 * 1024


D = Defaults()


class Config:
    _ENVPREFIX = "LOGDOCTOR_"

    def __init__(self, config_path: str | None = None) -> None:
        self.data: dict[str, Any] = self._defaults()
        if config_path and Path(config_path).exists():
            self._load_yaml(config_path)
        self._load_env()
        self._sandbox_dir: str | None = None

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "model": "gpt-4o",
            "fallback_model": "gpt-3.5-turbo",
            "llm_timeout": 300,
            "llm_retries": 3,
            "max_patch_files": 3,
            "max_patch_lines": 120,
            "danger_patterns": ["rm -rf", "os.system", "eval(", "exec(", "DROP TABLE"],
            "enable_ast": True,
            "cmd_timeout": 30,
            "test_timeout": 300,
            "git_timeout": 60,
            "rounds": 3,
            "log_level": "INFO",
            "log_file": "logdoctor.log",
            "trace_file": "trace.json",
            "rag_enable": True,
            "rag_path": "./rag_data",
            "rag_top_k": 3,
            "sandbox_enable": True,
            "sandbox_dir": None,
            "sandbox_cpu_quota": D.SANDBOX_CPU_QUOTA,
            "sandbox_nproc": D.SANDBOX_NPROC,
            "sandbox_use_virtualenv": True,
            "web_port": 9999,
            "ci_environment": "local",
            "patch_allowed_extensions": [],
            "patch_allowed_paths": [],
            "patch_denied_paths": [
                ".github/", ".git/", ".env", "credentials", "secrets",
                "__pycache__/", "node_modules/", "vendor/",
            ],
            "critical_files": ["main.py", "app.py", "settings.py"],
            "evolve_max_retries": 3,
            "evolve_confirm": False,
            "evolve_sandbox_validate": True,
        }

    def _load_yaml(self, path: str) -> None:
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    self.data.update(data)
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to load config from %s: %s", path, e)

    def _load_env(self) -> None:
        for key, default in list(self.data.items()):
            raw = os.getenv(f"{self._ENVPREFIX}{key.upper()}")
            if raw is None:
                continue
            try:
                if isinstance(default, list):
                    self.data[key] = json.loads(raw)
                elif isinstance(default, bool):
                    self.data[key] = raw.lower() in ("true", "1", "yes")
                elif isinstance(default, int):
                    self.data[key] = int(raw)
                elif isinstance(default, float):
                    self.data[key] = float(raw)
                else:
                    self.data[key] = raw
            except Exception:
                self.data[key] = raw

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "sandbox_dir" and self._sandbox_dir is None:
            self._sandbox_dir = os.path.join(tempfile.gettempdir(), "logdoctor_sandbox")
        if name == "sandbox_dir":
            return self._sandbox_dir
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)