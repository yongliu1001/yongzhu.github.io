from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

from .config import D

_LOG_FMT = logging.Formatter(
    "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logger(
    name: str, level: str = "INFO", log_file: str | None = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(_LOG_FMT)
        logger.addHandler(sh)
        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(_LOG_FMT)
            logger.addHandler(fh)
    return logger


class LogSanitizer:
    _PATH_RE = re.compile(r"(?:\w:)?(?:[\\/][\w.-]+)+")
    _ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
    _TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")

    @classmethod
    def clean(cls, text: str) -> str:
        text = cls._PATH_RE.sub("<PATH>", text)
        text = cls._ADDR_RE.sub("<ADDR>", text)
        text = cls._TIME_RE.sub("<TIME>", text)
        return text

    @staticmethod
    def extract_error_window(text: str, window_size: int = D.ERROR_WINDOW_SIZE) -> str:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if any(k in line.lower() for k in ("error", "exception", "traceback", "failed")):
                start = max(0, i - D.ERROR_CONTEXT_LINES)
                end = min(len(lines), i + D.ERROR_CONTEXT_LINES)
                return "\n".join(lines[start:end])
        return text[-window_size:]


class FileLock:
    """跨平台文件锁（自实现，无额外依赖）"""

    def __init__(self, path: str | Path, timeout: float = 30) -> None:
        self.path = str(path)
        self.timeout = timeout
        self._fd = None

    def acquire(self) -> bool:
        import platform
        try:
            if platform.system() == "Windows":
                import msvcrt
                self._fd = open(self.path, "w")
                msvcrt.locking(self._fd.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                self._fd = open(self.path, "w")
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except Exception:
            if self._fd:
                try:
                    self._fd.close()
                except Exception:
                    pass
                self._fd = None
            return False

    def release(self) -> None:
        if self._fd is None:
            return
        import platform
        try:
            if platform.system() == "Windows":
                import msvcrt
                msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            self._fd.close()
        except Exception:
            pass
        finally:
            self._fd = None
            try:
                if os.path.exists(self.path):
                    os.unlink(self.path)
            except OSError:
                pass

    def __enter__(self) -> "FileLock":
        t0 = time.monotonic()
        while not self.acquire():
            if time.monotonic() - t0 > self.timeout:
                raise RuntimeError(f"Lock timeout after {self.timeout}s: {self.path}")
            time.sleep(D.LOCK_POLL_INTERVAL)
        return self

    def __exit__(self, *args: object) -> None:
        self.release()