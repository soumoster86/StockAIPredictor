#!/usr/bin/env python3
"""Sanity-check global model artifacts (local dir or GLOBAL_MODEL_DIR).

Usage:
    python scripts/check_models.py
    GLOBAL_MODEL_DIR=/data/models python scripts/check_models.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import (  # noqa: E402
    FEATURES,
    GLOBAL_META_FILE,
    GLOBAL_MODEL_DIR,
    global_model_available,
    load_global_model,
)


def main():
    directory = os.environ.get("GLOBAL_MODEL_DIR", GLOBAL_MODEL_DIR)
    print(f"Model store: {Path(directory).resolve()}")
    meta_path = Path(directory) / GLOBAL_META_FILE
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  meta: {meta.get('n_stocks')} stocks, "
              f"trained {meta.get('trained_at')}, split={meta.get('split')}")
    else:
        print("  meta: missing")

    ok = 0
    for h in [1, 3, 5, 10, 20]:
        b = load_global_model(h, directory)
        if b is None:
            print(f"  h{h}: MISSING or feature mismatch")
        else:
            feats = b.get("features", [])
            match = list(feats) == list(FEATURES)
            print(f"  h{h}: ok (features match={match})")
            ok += 1
    print(f"\n{ok} horizon model(s) loadable. "
          f"global_model_available={global_model_available(directory)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
