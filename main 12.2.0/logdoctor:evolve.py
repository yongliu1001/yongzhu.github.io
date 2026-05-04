from __future__ import annotations
import os
import sys
import json
import time
import shutil
import re
import subprocess
import tempfile
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

from .config import Config, D
from .models import EvolutionProposal, EvolutionResult
from .llm import LLMBase, create_llm
from .utils import setup_logger, FileLock
from .safety import ASTGuard

PROTECTED_MODULES = frozenset({"safety.py", "sandbox.py", "config.py", "models.py", "utils.py", "__init__.py"})

class SelfEvolver:
    EVOLUTION_LOG = "evolution_history.json"
    _VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?([a-z]\d*)?$")

    _EVOLVE_BOOL_FLAGS = frozenset({"--self-evolve", "--evolve-history"})
    _EVOLVE_VALUE_FLAGS = frozenset({"--evolve-target", "--evolve-instructions", "--evolve-rollback"})
    _EVOLVE_ALL_FLAGS = _EVOLVE_BOOL_FLAGS | _EVOLVE_VALUE_FLAGS

    def __init__(self, config: Config, llm: Optional[LLMBase] = None):
        self.config = config
        self.llm = llm or create_llm(config)
        self.logger = setup_logger(__name__ + ".Evolver", config.log_level)
        self.guard = ASTGuard()
        self._history: List[Dict] = self._load_history()
        self.module_dir = Path(__file__).parent
        self.entry_script = sys.argv[0]

    # ---------- 入口 ----------
    def evolve(self, target_file: Optional[str] = None, instructions: Optional[str] = None, dry_run: bool = False) -> EvolutionResult:
        t0 = time.monotonic()
        lock_path = os.path.join(tempfile.gettempdir(), ".logdoctor_evolve.lock")
        try:
            with FileLock(lock_path, timeout=5):
                if target_file:
                    target = os.path.abspath(target_file)
                    return self._evolve_single_file(target, instructions, dry_run, t0)
                else:
                    return self._evolve_package(instructions, dry_run, t0)
        except RuntimeError:
            return self._fail("另一个进化进程正在运行，请稍后重试", "", "")

    # ---------- 单文件进化 ----------
    def _evolve_single_file(self, target: str, instructions: Optional[str], dry_run: bool, t0: float) -> EvolutionResult:
        source = self._read_source(target)
        if not source:
            return self._fail("无法读取源码", "", "")
        old_version = self._extract_version(source)
        self.logger.info("📌 单文件版本: %s", old_version)

        proposal = self._single_file_llm(source, instructions)
        if not proposal:
            return self._fail("LLM 优化失败", old_version, "")

        new_version = proposal.version.strip()
        if not self._VERSION_RE.match(new_version):
            new_version = self._bump_version(old_version)
            proposal.version = new_version

        self.logger.info("📝 优化: %s", proposal.changelog)
        new_source = self._stamp_version(proposal.full_source, new_version)

        ok, msg = self._validate_syntax(new_source)
        if not ok:
            return self._fail(f"语法错误: {msg}", old_version, new_version)
        ok, msg = self._validate_ast_safety(new_source)
        if not ok:
            return self._fail(f"安全检查失败: {msg}", old_version, new_version)

        if self.config.evolve_sandbox_validate:
            ok, msg = self._sandbox_validate(new_source, target)
            if not ok:
                return self._fail(f"沙箱验证失败: {msg}", old_version, new_version)

        if self.config.evolve_confirm and not dry_run:
            print(f"进化: {old_version}→{new_version}\n{proposal.changelog}")
            ans = input("确认? [y/N] ").strip().lower()
            if ans != "y":
                return self._fail("用户取消", old_version, new_version)

        if dry_run:
            return EvolutionResult(success=True, from_version=old_version, to_version=new_version,
                                   message="[DRY-RUN] 预览成功", elapsed_sec=time.monotonic()-t0)

        backup = self._backup(target, old_version)
        self.logger.info("💾 备份: %s", backup)
        if not self._write_source(target, new_source):
            self._rollback(target, backup)
            return self._fail("写入失败，已回滚", old_version, new_version)
        self.logger.info("🧪 写入后自测...")
        if not self._post_write_self_test(target):
            self._rollback(target, backup)
            return self._fail("写入后自测失败，已回滚", old_version, new_version)

        self._record(old_version, new_version, proposal, target)
        elapsed = time.monotonic() - t0
        self.logger.info("🎉 单文件进化完成: %s→%s (%.1fs)", old_version, new_version, elapsed)
        self._restart()
        return EvolutionResult(success=True, from_version=old_version, to_version=new_version,
                               message=f"进化成功: {proposal.changelog}", backup_path=backup, elapsed_sec=elapsed)

    def _single_file_llm(self, source, instructions):
        extra = f"\n\n用户额外指令:\n{instructions}" if instructions else ""
        current_version = self._extract_version(source)
        prompt = (
            f"优化以下 Python 文件，保持功能，提升质量，版本号递增（当前{current_version}）。\n"
            f"{extra}\n"
            f"当前源码:\n```python\n{source}\n```\n\n"
            f"返回 JSON：version, changelog, optimizations, risks, confidence, full_source。"
        )
        msgs = [{"role": "system", "content": "You are an expert Python architect."}, {"role": "user", "content": prompt}]
        result, _ = self.llm.call(msgs, EvolutionProposal)
        if result and result.full_source:
            return result
        raw, _ = self.llm.call_raw(msgs)
        if raw:
            return self._parse_evolution_json(raw)
        return None

    # ---------- 包进化 ----------
    def _evolve_package(self, instructions: Optional[str], dry_run: bool, t0: float) -> EvolutionResult:
        modules = self._read_all_modules()
        if not modules:
            return self._fail("无法读取模块", "", "")
        old_version = self._extract_version(modules.get("__init__.py", ""))
        self.logger.info("📌 包版本: %s", old_version)

        proposal = self._call_llm_optimize_package(modules, instructions)
        if not proposal or not proposal.modules:
            return self._fail("LLM 优化失败", old_version, "")

        for fname, src in proposal.modules.items():
            ok, msg = self._validate_syntax(src)
            if not ok:
                return self._fail(f"模块 {fname} 语法错误: {msg}", old_version, proposal.version)
            ok, msg = self._validate_ast_safety(src)
            if not ok:
                return self._fail(f"模块 {fname} 安全检查失败: {msg}", old_version, proposal.version)

        if self.config.evolve_sandbox_validate:
            if not self._smoke_test_package(proposal.modules):
                return self._fail("包冒烟测试失败", old_version, proposal.version)

        if dry_run:
            return EvolutionResult(success=True, from_version=old_version, to_version=proposal.version,
                                   message="[DRY-RUN] 预览成功", elapsed_sec=time.monotonic()-t0)

        backup_path = self._backup_package(old_version)
        self.logger.info("💾 包备份: %s", backup_path)

        try:
            tmp_dir = self.module_dir.with_name("logdoctor_new")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir()
            for fname, src in proposal.modules.items():
                (tmp_dir / fname).write_text(src, encoding="utf-8")
            old_dir = self.module_dir
            shutil.rmtree(str(old_dir))
            shutil.move(str(tmp_dir), str(old_dir))
        except Exception as e:
            self._rollback_package(backup_path)
            return self._fail(f"包替换失败: {e}", old_version, proposal.version)

        self._record(old_version, proposal.version, proposal, "package")
        self.logger.info("🎉 包进化完成: %s→%s", old_version, proposal.version)
        self._restart()
        return EvolutionResult(success=True, from_version=old_version, to_version=proposal.version,
                               message=f"包进化成功: {proposal.changelog}", backup_path=backup_path,
                               elapsed_sec=time.monotonic()-t0)

    def _read_all_modules(self) -> dict[str, str]:
        modules = {}
        for py_file in self.module_dir.glob("*.py"):
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue
            if py_file.name in PROTECTED_MODULES:
                continue
            try:
                modules[py_file.name] = py_file.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.warning("无法读取 %s: %s", py_file.name, e)
        return modules

    def _call_llm_optimize_package(self, modules: dict[str, str], instructions: Optional[str]) -> Optional[EvolutionProposal]:
        extra = f"\n\n用户额外指令:\n{instructions}" if instructions else ""
        current_version = self._extract_version(modules.get("__init__.py", ""))
        module_text = "\n".join(f"### {name}\n```python\n{code}\n```\n" for name, code in modules.items())
        prompt = (
            f"优化整个 LogDoctor 包。当前版本 {current_version}。\n"
            f"{extra}\n"
            f"返回 JSON 包含 modules 字段，键为文件名（如 cli.py），值为完整的优化后源码。\n"
            f"当前模块：\n{module_text}"
        )
        msgs = [{"role": "system", "content": "You are an expert Python architect. Output valid JSON."},
                {"role": "user", "content": prompt}]
        result, err = self.llm.call(msgs, EvolutionProposal)
        if result and result.modules:
            return result
        raw, _ = self.llm.call_raw(msgs)
        if raw:
            return self._parse_evolution_json(raw)
        return None

    def _smoke_test_package(self, modules: dict[str, str]) -> bool:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir) / "logdoctor"
            pkg_dir.mkdir()
            if "__init__.py" not in modules:
                (pkg_dir / "__init__.py").write_text("")
            for fname, src in modules.items():
                (pkg_dir / fname).write_text(src, encoding="utf-8")
            test_script = (
                f"import sys; sys.path.insert(0, '{tmpdir}')\n"
                f"from logdoctor.config import Config\n"
                f"from logdoctor.utils import LogSanitizer\n"
                f"Config(); LogSanitizer.clean('test')\n"
                f"print('SMOKE_OK')"
            )
            r = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=30)
            return 'SMOKE_OK' in r.stdout

    def _backup_package(self, version: str) -> str:
        backup_dir = self.module_dir.parent / ".evolve_backups"
        backup_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"logdoctor_v{version}_{ts}"
        backup_path = backup_dir / backup_name
        shutil.make_archive(str(backup_path), 'zip', str(self.module_dir.parent), 'logdoctor')
        backups = sorted(backup_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[self.config.evolve_backup_keep:]:
            old.unlink()
        return str(backup_path) + ".zip"

    def _rollback_package(self, backup_zip: str) -> None:
        try:
            import zipfile
            with zipfile.ZipFile(backup_zip, 'r') as zf:
                zf.extractall(self.module_dir.parent)
            self.logger.warning("已回滚包到备份: %s", backup_zip)
        except Exception as e:
            self.logger.critical("回滚包失败: %s", e)

    # ---------- 通用辅助方法 ----------
    def list_history(self) -> List[Dict]:
        return self._history

    def rollback_to(self, backup_path: str, target: Optional[str] = None) -> bool:
        if backup_path.endswith(".zip"):
            self._rollback_package(backup_path)
            self._restart()
            return True
        target = target or __file__
        if not os.path.exists(backup_path):
            self.logger.error("备份不存在: %s", backup_path)
            return False
        shutil.copy2(backup_path, target)
        self._restart()
        return True

    @staticmethod
    def _bump_version(version: str) -> str:
        parts = version.split(".")
        if len(parts) >= 2:
            try:
                parts[-1] = str(int(parts[-1]) + 1)
            except ValueError:
                parts.append("1")
        else:
            parts.append("1")
        return ".".join(parts)

    @staticmethod
    def _stamp_version(source: str, version: str) -> str:
        pattern = r'(__version__\s*=\s*["\'])([^"\']*)(["\'])'
        if re.search(pattern, source):
            return re.sub(pattern, rf"\g<1>{version}\g<3>", source)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ", "class ", "def ", "#")):
                continue
            lines.insert(i, f'__version__ = "{version}"')
            return "\n".join(lines)
        return source

    @staticmethod
    def _post_write_self_test(target: str) -> bool:
        try:
            with open(target, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=target)
            return True
        except Exception as e:
            logging.getLogger(__name__).error("写入后自测失败: %s", e)
            return False

    def _read_source(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error("读取失败: %s", e)
            return None

    def _write_source(self, path: str, source: str) -> bool:
        tmp = path + ".evolving"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(source)
            with open(tmp, "r", encoding="utf-8") as f:
                f.read()
            shutil.move(tmp, path)
            return True
        except Exception as e:
            self.logger.error("写入失败: %s", e)
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return False

    @staticmethod
    def _extract_version(source: str) -> str:
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', source)
        return m.group(1) if m else "unknown"

    def _validate_syntax(self, source: str) -> Tuple[bool, str]:
        try:
            ast.parse(source, filename="<evolved>")
            return True, "ok"
        except SyntaxError as e:
            return False, f"line {e.lineno}: {e.msg}"

    def _validate_ast_safety(self, source: str) -> Tuple[bool, str]:
        return self.guard.check(source, context="project")

    def _sandbox_validate(self, new_source: str, target: str) -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory(prefix="evolve_sb_") as tmpdir:
            test_file = os.path.join(tmpdir, "validate_target.py")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(new_source)
            validate_script = "import ast, sys; ast.parse(open(sys.argv[1]).read(), filename=sys.argv[1]); print('OK')"
            r = subprocess.run([sys.executable, "-c", validate_script, test_file], capture_output=True, text=True, timeout=D.EVOLVE_SANDBOX_TIMEOUT)
            if r.returncode != 0:
                return False, f"沙箱验证失败: {r.stderr[:200]}"
            return True, "ok"

    def _backup(self, target: str, version: str) -> str:
        backup_dir = os.path.join(str(self.module_dir), ".evolve_backups")
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(backup_dir, f"v{version}_{ts}.py")
        shutil.copy2(target, backup)
        self._cleanup_backups(backup_dir)
        return backup

    def _cleanup_backups(self, backup_dir: str) -> None:
        files = sorted(Path(backup_dir).glob("v*.py"), key=lambda p: p.stat().st_mtime)
        while len(files) > D.EVOLVE_BACKUP_KEEP:
            old = files.pop(0)
            old.unlink(missing_ok=True)

    def _rollback(self, target: str, backup: str) -> None:
        try:
            shutil.copy2(backup, target)
            self.logger.warning("已回滚到: %s", backup)
        except Exception as e:
            self.logger.critical("回滚失败: %s", e)

    def _record(self, old_ver: str, new_ver: str, proposal: EvolutionProposal, target: str) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "from": old_ver,
            "to": new_ver,
            "target": target,
            "changelog": proposal.changelog,
            "optimizations": proposal.optimizations,
            "risks": proposal.risks,
            "confidence": proposal.confidence,
        }
        self._history.append(entry)
        log_path = os.path.join(str(self.module_dir), self.EVOLUTION_LOG)
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning("进化日志写入失败: %s", e)

    def _load_history(self) -> List[Dict]:
        log_path = os.path.join(str(self.module_dir), self.EVOLUTION_LOG)
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _compress_source(self, source: str) -> str:
        lines = source.split("\n")
        result = []
        in_string = None
        for line in lines:
            stripped = line.strip()
            if in_string:
                result.append(line)
                count = stripped.count(in_string)
                if count % 2 == 1:
                    in_string = None
                continue
            if not stripped:
                continue
            if stripped.startswith("#") and not stripped.startswith("#!"):
                continue
            for quote in ('"""', "'''"):
                count = stripped.count(quote)
                if count >= 2:
                    if count % 2 == 0:
                        continue
                    in_string = quote
                    break
                elif count == 1:
                    in_string = quote
                    break
            result.append(line)
        return "\n".join(result)

    @staticmethod
    def _parse_evolution_json(raw: str) -> Optional[EvolutionProposal]:
        try:
            data = json.loads(raw)
            return EvolutionProposal(**data)
        except Exception:
            pass
        m = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                return EvolutionProposal(**data)
            except Exception:
                pass
        return None

    def _restart(self) -> None:
        python = sys.executable
        clean_argv = self._strip_evolve_args(sys.argv[1:])
        if platform.system() == "Windows":
            bat = Path(tempfile.gettempdir()) / "_logdoctor_restart.bat"
            script = (
                "@echo off\n"
                "timeout /t 2 >nul\n"
                f"{' '.join([python, self.entry_script] + clean_argv)}\n"
                'del "%~f0"\n'
            )
            bat.write_text(script)
            subprocess.Popen(["cmd", "/c", str(bat)], shell=True)
            sys.exit(0)
        else:
            args = [python, self.entry_script] + clean_argv
            try:
                os.execvp(python, args)
            except OSError as e:
                self.logger.critical("重启失败: %s", e)
                sys.exit(1)

    @classmethod
    def _strip_evolve_args(cls, argv: List[str]) -> List[str]:
        result = []
        skip_next = False
        for arg in argv:
            if skip_next:
                skip_next = False
                continue
            if any(arg.startswith(f + "=") for f in cls._EVOLVE_ALL_FLAGS):
                continue
            if arg in cls._EVOLVE_BOOL_FLAGS:
                continue
            if arg in cls._EVOLVE_VALUE_FLAGS:
                skip_next = True
                continue
            result.append(arg)
        return result

    def _fail(self, msg: str, old_ver: str, new_ver: str) -> EvolutionResult:
        self.logger.error("进化失败: %s", msg)
        return EvolutionResult(success=False, from_version=old_ver, to_version=new_ver, message=msg)