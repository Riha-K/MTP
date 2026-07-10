"""Copy only the S1 TIFF files referenced by a bench JSONL (for PARAM upload).

Example (laptop / remote CPU):

  python -m baresoil.pack_bench_s1 \\
    --bench-jsonl data/baresoil_s1/bench/v0.1/ai4lcc_val.jsonl \\
    --src-s1-dir data/baresoil_s1/ai4lcc/multisenge/s1 \\
    --out-dir data/baresoil_s1/ai4lcc/multisenge/s1_val_bench
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack bench-referenced S1 TIFFs into one folder.")
    parser.add_argument("--bench-jsonl", type=Path, required=True)
    parser.add_argument("--src-s1-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.src_s1_dir.is_dir():
        print(f"S1 dir not found: {args.src_s1_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    seen: set[str] = set()

    with args.bench_jsonl.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="pack-s1"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = Path(str(row.get("s1_path", "")).replace("\\", "/")).name
            if not name or name in seen:
                continue
            seen.add(name)
            src = args.src_s1_dir / name
            if not src.is_file():
                missing += 1
                print(f"missing: {name}", file=sys.stderr)
                continue
            dst = args.out_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1

    print(f"Copied {copied} files -> {args.out_dir} (missing={missing})")
    return 0 if missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
