#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    vipe_root = repo_root / "extern" / "vipe"

    if not vipe_root.exists():
        print(f"[vipe_test] Missing repo: {vipe_root}")
        return 1

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(vipe_root) not in sys.path:
        sys.path.insert(0, str(vipe_root))

    print(f"[vipe_test] repo_root={repo_root}")
    print(f"[vipe_test] vipe_root={vipe_root}")
    print(f"[vipe_test] TORCH_EXTENSIONS_DIR={os.environ.get('TORCH_EXTENSIONS_DIR', '<default>')}")

    try:
        import vipe.ext as vipe_ext  # noqa: F401
    except Exception as exc:
        print(f"[vipe_test] FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("[vipe_test] OK: vipe.ext loaded successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
