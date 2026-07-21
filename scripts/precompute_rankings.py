#!/usr/bin/env python3
"""Offline full-universe screener → rankings/rankings_latest.csv

Run on your machine (or a scheduled job), then commit/sync the rankings/
folder so Streamlit Cloud can load instant screener results:

    python scripts/precompute_rankings.py
    python scripts/precompute_rankings.py --stocks stocks_universe.csv
    python scripts/precompute_rankings.py --max 200          # smoke test
    python scripts/precompute_rankings.py --batch-size 60

Do NOT run this inside a Streamlit request — it is deliberately offline.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from screener import (  # noqa: E402
    RANKINGS_DIR,
    SCAN_BATCH,
    save_rankings,
    scan_universe,
)


def _norm_key(k: str) -> str:
    # Excel / Windows often saves CSV with a UTF-8 BOM on the first header.
    return (k or "").replace("\ufeff", "").strip().lower()


def load_watchlist(path: Path) -> list[tuple[str, str]]:
    """Load Name,Symbol rows. Tolerates BOM and odd whitespace."""
    rows = []
    seen = set()
    # utf-8-sig strips a leading BOM so DictReader sees "Name" not "\ufeffName"
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        for raw in reader:
            r = {_norm_key(k): (v or "").strip() for k, v in raw.items() if k is not None}
            name = r.get("name") or r.get("company") or r.get("security")
            sym = (r.get("symbol") or r.get("ticker") or r.get("yahoo") or "").upper()
            if name and sym and sym not in seen:
                seen.add(sym)
                rows.append((name, sym))
    return rows


def _resolve_stocks_path(arg: str) -> Path:
    """Prefer explicit path, then CWD, then repo root."""
    p = Path(arg)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([Path.cwd() / p, ROOT / p, ROOT / "stocks_universe.csv", ROOT / "stocks.csv"])
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return p  # for error message


def main():
    ap = argparse.ArgumentParser(description="Precompute screener rankings")
    ap.add_argument(
        "--stocks", default=str(ROOT / "stocks_universe.csv"),
        help="Watchlist CSV (Name,Symbol)",
    )
    ap.add_argument("--batch-size", type=int, default=SCAN_BATCH)
    ap.add_argument("--max", type=int, default=None, help="Cap symbols (debug)")
    ap.add_argument("--pause", type=float, default=0.35, help="Seconds between batches")
    ap.add_argument(
        "--out", default=str(RANKINGS_DIR),
        help="Output directory for rankings_latest.csv + meta",
    )
    args = ap.parse_args()

    path = _resolve_stocks_path(args.stocks)
    if not path.exists():
        sys.exit(
            f"Watchlist not found: {args.stocks}\n"
            f"Tried under CWD and repo root ({ROOT})."
        )

    items = load_watchlist(path)
    if not items:
        # Diagnose common BOM / column issues
        with path.open(encoding="utf-8", errors="replace") as f:
            head = f.readline().rstrip("\n")
        sys.exit(
            f"Watchlist empty after parse: {path}\n"
            f"First line (raw): {head!r}\n"
            f"Expected columns: Name, Symbol (UTF-8; Excel BOM is OK with this fix)."
        )
    print(f"Precomputing rankings for {len(items)} symbols from {path}")
    print(f"Batch size={args.batch_size}  pause={args.pause}s  max={args.max}")

    def on_batch(done, total, n_ok, n_fail):
        print(f"  [{done}/{total}] batch ok={n_ok} fail={n_fail}", flush=True)

    df, failures, meta = scan_universe(
        items,
        batch_size=args.batch_size,
        max_symbols=args.max,
        pause_s=args.pause,
        on_batch=on_batch,
    )
    meta["watchlist"] = path.name
    csv_path, meta_path = save_rankings(df, failures, meta, directory=args.out)

    print(f"\nDone in {meta['elapsed_s']}s")
    print(f"  scored={meta['n_scored']}  failed={meta['n_failed']}  engine={meta['engine']}")
    print(f"  wrote {csv_path}")
    print(f"  wrote {meta_path}")
    print("Commit or sync the rankings/ folder for instant app loads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
