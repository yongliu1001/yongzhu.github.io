#!/usr/bin/env python3
"""LogDoctor Pro 命令行入口"""
from __future__ import annotations

import argparse
import os
import sys

from .config import Config
from .core import LogDoctor
from .evolve import SelfEvolver


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LogDoctor Pro v12.2.0 Phoenix — Agentic DevOps 自愈平台 + 自我进化引擎",
    )
    ap.add_argument("log", nargs="?", help="错误日志文件路径")
    ap.add_argument("--repo", default=".", help="Git 仓库路径")
    ap.add_argument("--dry-run", action="store_true", help="仅预览，不实际修改")
    ap.add_argument("--force", action="store_true", help="允许脏工作目录")
    ap.add_argument("--config", help="配置文件路径 (YAML/JSON)")

    eg = ap.add_argument_group("🧬 自我进化")
    eg.add_argument("--self-evolve", action="store_true", help="启动自我进化")
    eg.add_argument("--evolve-target", help="单文件进化目标（默认全包进化）")
    eg.add_argument("--evolve-instructions", help="额外优化指令")
    eg.add_argument("--evolve-history", action="store_true", help="查看进化历史")
    eg.add_argument("--evolve-rollback", metavar="BACKUP", help="回滚到指定备份文件")

    args = ap.parse_args()
    config = Config(args.config)

    # 进化模式
    if args.self_evolve or args.evolve_history or args.evolve_rollback:
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("OLLAMA_HOST"):
            sys.exit("Error: 请设置 OPENAI_API_KEY 或 OLLAMA_HOST 环境变量")

        evolver = SelfEvolver(config)
        if args.evolve_history:
            history = evolver.list_history()
            if not history:
                print("暂无进化历史")
            else:
                for i, h in enumerate(history, 1):
                    print(f"[{i}] {h['timestamp']}  {h['from']} → {h['to']}  {h['changelog']}")
            sys.exit(0)

        if args.evolve_rollback:
            ok = evolver.rollback_to(args.evolve_rollback, args.evolve_target)
            sys.exit(0 if ok else 1)

        result = evolver.evolve(
            target_file=args.evolve_target,
            instructions=args.evolve_instructions,
            dry_run=args.dry_run,
        )
        # 避免重复打印，进化内部已经打印了结果
        sys.exit(0 if result.success else 1)

    # 传统自愈模式
    if not args.log:
        ap.error("传统模式需要指定日志文件，或使用 --self-evolve 进入进化模式")
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OLLAMA_HOST"):
        sys.exit("Error: 请设置 OPENAI_API_KEY 或 OLLAMA_HOST 环境变量")

    doctor = LogDoctor(config)
    ok = doctor.run(args.log, args.repo, dry=args.dry_run, force=args.force)
    print(doctor.metrics.report())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()