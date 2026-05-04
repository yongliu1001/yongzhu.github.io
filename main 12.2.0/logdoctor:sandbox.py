from __future__ import annotations

import hashlib
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, D
from .git_ops import GitOps
from .models import ExecutionPlan, LogAnalysis, ProjectType
from .utils import setup_logger

# ---------------------------------------------------------------------------
# 项目检测 & 语言规格
# ---------------------------------------------------------------------------

def detect_project(repo: str) -> ProjectType:
    try:
        files = set(os.listdir(repo))
    except OSError:
        return ProjectType.GENERIC
    if files & {"poetry.lock", "requirements.txt", "pyproject.toml"}:
        return ProjectType.PYTHON
    if "package.json" in files:
        return ProjectType.NODE
    if "go.mod" in files:
        return ProjectType.GO
    if "Cargo.toml" in files:
        return ProjectType.RUST
    if "pom.xml" in files:
        return ProjectType.JAVA
    return ProjectType.GENERIC


LANG_SPEC: Dict[ProjectType, Dict[str, Any]] = {
    ProjectType.PYTHON: {
        "test": ["pytest", "-q"],
        "linter": ["ruff", "check", "."],
        "install": ["pip", "install"],
        "install_cmd_template": "{pip} install {pkg}",
        "allowed_test_commands": ["pytest", "python -m pytest", "python -m unittest"],
    },
    ProjectType.NODE: {
        "test": ["npm", "test"],
        "linter": ["npx", "eslint", "."],
        "install": ["npm", "install"],
        "install_cmd_template": "npm install {pkg}",
        "allowed_test_commands": ["npm test", "npx jest", "node -c"],
    },
    ProjectType.GO: {
        "test": ["go", "test", "./..."],
        "linter": ["golangci-lint", "run"],
        "install": ["go", "get"],
        "install_cmd_template": "go get {pkg}",
        "allowed_test_commands": ["go test"],
    },
    ProjectType.RUST: {
        "test": ["cargo", "test"],
        "linter": ["cargo", "clippy"],
        "install": ["cargo", "add"],
        "install_cmd_template": "cargo add {pkg}",
        "allowed_test_commands": ["cargo test"],
    },
    ProjectType.JAVA: {
        "test": ["mvn", "test"],
        "linter": ["mvn", "checkstyle:check"],
        "install": ["mvn", "dependency:resolve"],
        "install_cmd_template": "mvn dependency:get -Dartifact={pkg}",
        "allowed_test_commands": ["mvn test"],
    },
    ProjectType.GENERIC: {
        "test": ["make", "test"],
        "linter": None,
        "install": [],
        "install_cmd_template": None,
        "allowed_test_commands": [],
    },
}

# ---------------------------------------------------------------------------
# 审计钩子
# ---------------------------------------------------------------------------

_AUDIT_SCRIPT = """\
import sys, os, builtins
BLOCKED = {
    'os.system', 'os.popen', 'subprocess.Popen', 'subprocess.run',
    'subprocess.call', 'eval', 'exec', 'compile', 'importlib.import_module',
}
def _audit_hook(event, args):
    if event in BLOCKED:
        raise RuntimeError(f'Blocked event: {event}')
sys.addaudithook(_audit_hook)
"""


def inject_audit_hook(sandbox_dir: str) -> None:
    hook_file = os.path.join(sandbox_dir, "_logdoctor_audit.py")
    with open(hook_file, "w", encoding="utf-8") as f:
        f.write(_AUDIT_SCRIPT)

    for root, _, files in os.walk(sandbox_dir):
        for file in files:
            if file.endswith(".py") and file != "_logdoctor_audit.py":
                full = os.path.join(root, file)
                with open(full, "r+", encoding="utf-8") as f:
                    content = f.read()
                    if "import _logdoctor_audit" not in content:
                        f.seek(0)
                        f.write("import _logdoctor_audit\n" + content)


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

class Sandbox:
    def __init__(self, source_repo: str, config: Config) -> None:
        self.source_repo = source_repo
        self.config = config
        repo_hash = hashlib.md5(source_repo.encode()).hexdigest()[:12]
        self.path = os.path.join(config.sandbox_dir, f"logdoctor_{repo_hash}_{int(time.time())}")
        self.git = GitOps(source_repo)
        self.venv_python: str | None = None
        self.bootstrap_done = False
        self.bootstrap_status: str | None = None

    def create(self) -> bool:
        os.makedirs(self.config.sandbox_dir, exist_ok=True)
        if not self.git.clone_to(self.path):
            return False
        if self.config.sandbox_use_virtualenv and detect_project(self.path) == ProjectType.PYTHON:
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", os.path.join(self.path, ".venv")],
                    check=True, capture_output=True, timeout=60,
                )
                bindir = "Scripts" if platform.system() == "Windows" else "bin"
                self.venv_python = os.path.join(self.path, ".venv", bindir, "python")
            except Exception:
                self.venv_python = None
        if detect_project(self.path) == ProjectType.PYTHON:
            inject_audit_hook(self.path)
        return True

    def destroy(self) -> None:
        if os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)

    def set_process_limits(self) -> None:
        if platform.system() == "Windows":
            return
        try:
            import resource
            cpu_sec = self.config.test_timeout + 30
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
            resource.setrlimit(resource.RLIMIT_NPROC, (self.config.sandbox_nproc, self.config.sandbox_nproc))
        except Exception:
            pass

    def run_cmd(self, cmd: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
        # 网络隔离
        env = os.environ.copy()
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[key] = "http://127.0.0.1:9"

        if platform.system() == "Windows":
            try:
                return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
            except subprocess.TimeoutExpired:
                return subprocess.CompletedProcess(cmd, -1, "", "Timeout")
        def _preexec():
            os.setsid()
            self.set_process_limits()
        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, preexec_fn=_preexec, env=env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                return subprocess.CompletedProcess(cmd, -1, stdout or "", f"Timeout {timeout}s\n{stderr or ''}")
        except Exception as e:
            return subprocess.CompletedProcess(cmd, -1, "", str(e))

    def bootstrap_deps(self, ptype: ProjectType, spec: dict) -> None:
        if self.bootstrap_done:
            return
        ok = True
        try:
            if ptype == ProjectType.PYTHON:
                interp = self.venv_python or sys.executable
                timeout = self.config.cmd_timeout * 2
                req = os.path.join(self.path, "requirements.txt")
                pyproj = os.path.join(self.path, "pyproject.toml")
                if os.path.exists(req):
                    ok = self.run_cmd([interp, "-m", "pip", "install", "-r", req], self.path, timeout).returncode == 0
                elif os.path.exists(pyproj):
                    ok = self.run_cmd([interp, "-m", "pip", "install", "-e", "."], self.path, timeout).returncode == 0
            elif ptype == ProjectType.NODE:
                timeout = self.config.cmd_timeout * 2
                if os.path.exists(os.path.join(self.path, "package-lock.json")):
                    ok = self.run_cmd(["npm", "ci"], self.path, timeout).returncode == 0
                else:
                    ok = self.run_cmd(["npm", "install"], self.path, timeout).returncode == 0
            elif ptype == ProjectType.GO:
                ok = self.run_cmd(["go", "mod", "download"], self.path, self.config.cmd_timeout * 2).returncode == 0
            elif ptype == ProjectType.RUST:
                ok = self.run_cmd(["cargo", "fetch"], self.path, self.config.cmd_timeout * 2).returncode == 0
            elif ptype == ProjectType.JAVA:
                ok = self.run_cmd(["mvn", "dependency:go-offline"], self.path, self.config.cmd_timeout * 2).returncode == 0
        except Exception as e:
            self.bootstrap_status = f"bootstrap error: {e}"
        else:
            self.bootstrap_status = "ok" if ok else "bootstrap failed"
        finally:
            self.bootstrap_done = True

    def __enter__(self) -> "Sandbox":
        if not self.create():
            raise RuntimeError("Sandbox creation failed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.destroy()


# ---------------------------------------------------------------------------
# DependencyResolver
# ---------------------------------------------------------------------------

class DependencyResolver:
    PACKAGE_IMPORT_MAP = {
        "opencv-python": "cv2",
        "pyyaml": "yaml",
        "pillow": "PIL",
        "scikit-learn": "sklearn",
        "beautifulsoup4": "bs4",
    }

    def __init__(self, spec: dict, config: Config) -> None:
        self.spec = spec
        self.config = config

    def resolve_and_install(
        self, analysis: LogAnalysis, sandbox_path: str, python_interpreter: str | None = None
    ) -> Tuple[bool, str, List[str]]:
        packages = self._extract_packages(analysis)
        if not packages:
            return False, "No package identified", []
        tmpl = self.spec.get("install_cmd_template")
        if not tmpl:
            return False, "No install command template", []
        installed: list[str] = []
        for pkg in packages:
            if self._is_installed(pkg, sandbox_path, python_interpreter):
                continue
            pip = self._resolve_pip(python_interpreter)
            cmd = shlex.split(tmpl.format(pip=pip, pkg=pkg))
            r = subprocess.run(cmd, cwd=sandbox_path, capture_output=True, text=True, timeout=self.config.cmd_timeout)
            if r.returncode != 0:
                return False, f"Install failed for {pkg}: {r.stderr[:200]}", installed
            installed.append(pkg)
        return True, f"Installed: {', '.join(installed)}", installed

    def _is_installed(self, pkg: str, sandbox_path: str, interp: str | None) -> bool:
        if not interp:
            return False
        imp = self.PACKAGE_IMPORT_MAP.get(pkg, pkg.replace("-", "_"))
        return subprocess.run(
            [interp, "-c", f"import {imp}"], cwd=sandbox_path,
            timeout=5, capture_output=True, text=True,
        ).returncode == 0

    @staticmethod
    def _extract_packages(analysis: LogAnalysis) -> list[str]:
        pkgs: set[str] = set()
        for rc in analysis.root_causes:
            if rc.is_dependency_missing and rc.missing_package:
                pkgs.add(rc.missing_package)
            for text in (rc.description, rc.error_type):
                for pat in (
                    r"ModuleNotFoundError: No module named '([\w-]+)'",
                    r"ImportError: No module named '([\w-]+)'",
                ):
                    m = re.search(pat, text)
                    if m:
                        pkgs.add(m.group(1))
        return list(pkgs)

    @staticmethod
    def _resolve_pip(interp: str | None) -> str:
        if not interp:
            return "pip"
        base = os.path.dirname(interp)
        suffix = ".exe" if platform.system() == "Windows" else ""
        p = os.path.join(base, f"pip{suffix}")
        return p if os.path.exists(p) else "pip"


# ---------------------------------------------------------------------------
# TransactionManager
# ---------------------------------------------------------------------------

class TransactionManager:
    def __init__(self, git: GitOps, sandbox_git: GitOps, spec: dict, config: Config, plan: ExecutionPlan) -> None:
        self.git = git
        self.sandbox_git = sandbox_git
        self.spec = spec
        self.config = config
        self.plan = plan

    def execute(
        self, diff: str, analysis: LogAnalysis, sandbox: Sandbox,
        python_interp: str | None = None,
    ) -> Tuple[bool, str]:
        if self.plan.must_steps:
            for step in self.plan.must_steps:
                if step not in self.plan.steps:
                    return False, f"must_step '{step}' not in plan steps"
        if not self._check(diff):
            return False, "patch check failed"
        if not self.sandbox_git.apply(diff):
            return False, "patch apply failed"
        if "install" in self.plan.steps:
            resolver = DependencyResolver(self.spec, self.config)
            ok, msg, _ = resolver.resolve_and_install(analysis, sandbox.path, python_interp)
            if not ok:
                return False, f"dependency install failed: {msg}"
        if "lint" in self.plan.steps and self.spec.get("linter"):
            cmd = self._resolve_lint_cmd(python_interp)
            r = sandbox.run_cmd(cmd, sandbox.path, self.config.cmd_timeout)
            if r.returncode != 0:
                return False, f"linter failed: {r.stderr[:200]}"
        test_cmd = self.spec.get("test", [])
        if test_cmd:
            allowed = self.spec.get("allowed_test_commands", [])
            if allowed:
                cmd_str = " ".join(test_cmd)
                if not any(cmd_str.startswith(a) for a in allowed):
                    return False, f"test command not allowed: {cmd_str}"
            cmd = self._resolve_test_cmd(test_cmd, python_interp)
            r = sandbox.run_cmd(cmd, sandbox.path, self.config.test_timeout)
            if r.returncode != 0:
                mode = self.plan.failure_mode
                tag = "manual review" if mode == "manual_review" else "test"
                return False, f"{tag} failed: {r.stderr[:200]}"
        main_diff = self._get_sandbox_diff()
        if main_diff and self.git.apply(main_diff):
            return True, "success"
        return False, "merge failed"

    def _get_sandbox_diff(self) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", self.sandbox_git.repo, "diff", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            return result.stdout
        except Exception:
            return self.git.diff_from(self.sandbox_git.repo)

    def _check(self, diff: str) -> bool:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
            f.write(diff)
            tmp = f.name
        try:
            return subprocess.run(
                ["git", "-C", self.sandbox_git.repo, "apply", "--check", tmp],
                capture_output=True,
            ).returncode == 0
        except Exception:
            return False
        finally:
            os.unlink(tmp)

    def _resolve_lint_cmd(self, interp: str | None) -> list[str]:
        cmd = list(self.spec["linter"])
        if interp and cmd and cmd[0] == "ruff":
            base = os.path.dirname(interp)
            suffix = ".exe" if platform.system() == "Windows" else ""
            p = os.path.join(base, f"ruff{suffix}")
            if os.path.exists(p):
                cmd = [p] + cmd[1:]
        return cmd

    def _resolve_test_cmd(self, cmd: list[str], interp: str | None) -> list[str]:
        c = list(cmd)
        if interp and c:
            if c[0] == "pytest":
                c = [interp, "-m", "pytest"] + c[1:]
            elif c[0] == "python":
                c[0] = interp
        return c