from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Tuple

from .config import Config
from .models import PatchBundle, PatchDetail, ProjectType


class DangerCategory:
    CODE_EXEC = "code_execution"
    SYSTEM_CMD = "system_command"
    NETWORK = "network_request"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    IMPORT_DYNAMIC = "dynamic_import"
    FFI = "foreign_function_interface"
    ATTRIBUTE = "attribute_manipulation"


_DANGEROUS_CALLS: dict[str, str] = {
    "eval": DangerCategory.CODE_EXEC,
    "exec": DangerCategory.CODE_EXEC,
    "compile": DangerCategory.CODE_EXEC,
    "os.system": DangerCategory.SYSTEM_CMD,
    "os.popen": DangerCategory.SYSTEM_CMD,
    "subprocess.run": DangerCategory.SYSTEM_CMD,
    "subprocess.Popen": DangerCategory.SYSTEM_CMD,
    "subprocess.call": DangerCategory.SYSTEM_CMD,
    "subprocess.check_output": DangerCategory.SYSTEM_CMD,
    "shutil.rmtree": DangerCategory.FILE_DELETE,
    "shutil.move": DangerCategory.FILE_WRITE,
    "importlib.import_module": DangerCategory.IMPORT_DYNAMIC,
    "socket.socket": DangerCategory.NETWORK,
    "requests.get": DangerCategory.NETWORK,
    "requests.post": DangerCategory.NETWORK,
    "ctypes.CDLL": DangerCategory.FFI,
}

_DANGEROUS_ATTRS = frozenset({
    "__subclasses__", "__bases__", "__globals__", "__code__",
    "__class__", "__builtins__",
})

_TEST_PATTERN = re.compile(r'(?:^|[/\\])test_[^/\\]+\.py$|_test\.py$|[/\\]tests?[/\\]')


def _is_test_file(filepath: str) -> bool:
    return bool(_TEST_PATTERN.search(filepath.replace('\\', '/')))


def _get_func_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_get_func_name(node.value)}.{node.attr}"
    return ""


class ASTGuard:
    _TEST_ALLOWED = frozenset({"subprocess.run", "subprocess.call", "shutil.rmtree"})

    def check(self, code: str, context: str = "project") -> tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"syntax error: {e}"
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _get_func_name(node.func)
                if name in self._TEST_ALLOWED and context == "test":
                    continue
                if name in _DANGEROUS_CALLS:
                    violations.append(f"dangerous call [{_DANGEROUS_CALLS[name]}]: {name}")
                if name == "open":
                    ok, msg = self._check_file_write(node, context)
                    if not ok:
                        violations.append(msg)
            elif isinstance(node, ast.Attribute):
                if node.attr in _DANGEROUS_ATTRS:
                    violations.append(f"dangerous attribute: {node.attr}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                ok, msg = self._check_import(node, context)
                if not ok:
                    violations.append(msg)
        if violations:
            return False, "; ".join(violations[:5])
        return True, "ok"

    def check_patch(
        self,
        patch: PatchDetail,
        ptype: ProjectType,
        config: Config,
        context: str = "project",
    ) -> tuple[bool, str]:
        for pattern in config.danger_patterns:
            if pattern in patch.diff:
                return False, f"danger pattern: {pattern}"
        if config.enable_ast and ptype == ProjectType.PYTHON:
            added = [
                line[1:] for line in patch.diff.split("\n")
                if line.startswith("+") and not line.startswith("+++")
            ]
            code = "\n".join(added)
            if code.strip():
                real_ctx = "test" if _is_test_file(patch.file) else "project"
                return self.check(code, real_ctx)
        return True, "ok"

    def _check_file_write(self, node: ast.Call, context: str) -> tuple[bool, str]:
        for arg in node.args[1:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if any(m in arg.value for m in ("w", "a", "x")):
                    if context != "test":
                        return False, "file write via open() with write mode"
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str) and any(m in kw.value.value for m in ("w", "a", "x")):
                    if context != "test":
                        return False, "file write via open() with write mode"
        return True, "ok"

    def _check_import(self, node: ast.AST, context: str) -> tuple[bool, str]:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in ("ctypes", "subprocess", "shutil", "importlib"):
                    if context != "test":
                        return False, f"dangerous import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in ("ctypes", "subprocess", "shutil", "importlib"):
                if context != "test":
                    return False, f"dangerous import from: {node.module}"
        return True, "ok"


class PatchParser:
    _DIFF_BLOCK = re.compile(r"```diff\s*\n(.*?)\n```", re.DOTALL)

    @staticmethod
    def extract_diff(text: str) -> str:
        matches = PatchParser._DIFF_BLOCK.findall(text)
        if matches:
            return "\n".join(matches)
        if re.search(r"^---\s+\S+", text, re.MULTILINE) and re.search(
            r"^\+\+\+\s+\S+", text, re.MULTILINE
        ):
            return text
        return ""


class PatchGuard:
    def __init__(self, config: Config) -> None:
        self.config = config

    def check(self, bundle: PatchBundle) -> tuple[bool, str]:
        if len(bundle.patches) > self.config.max_patch_files:
            return False, f"Too many files: {len(bundle.patches)} > {self.config.max_patch_files}"
        total = 0
        for p in bundle.patches:
            ok, msg = self._check_single(p)
            if not ok:
                return False, msg
            total += sum(1 for l in p.diff.split("\n") if l.startswith(("+", "-")))
        if total > self.config.max_patch_lines * 2:
            return False, f"Total changed lines {total} exceeds limit"
        return True, "ok"

    def _check_single(self, p: PatchDetail) -> tuple[bool, str]:
        path = Path(p.file)
        if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", p.file):
            return False, f"Dangerous path: {p.file}"
        norm = p.file.replace("\\", "/")
        for deny in self.config.patch_denied_paths:
            if deny in norm:
                return False, f"Denied pattern '{deny}': {p.file}"
        if self.config.patch_allowed_paths and not any(
            norm.startswith(pl) for pl in self.config.patch_allowed_paths
        ):
            return False, f"Not in allowed paths: {p.file}"
        if self.config.patch_allowed_extensions and path.suffix not in self.config.patch_allowed_extensions:
            return False, f"File type not allowed: {path.suffix}"
        churn = sum(1 for l in p.diff.split("\n") if l.startswith(("+", "-")))
        if churn > self.config.max_patch_lines:
            return False, f"Patch {p.file} has {churn} lines > {self.config.max_patch_lines}"
        return True, "ok"