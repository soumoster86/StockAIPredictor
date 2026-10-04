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
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rankings_log import log_rankings_run  # noqa: E402
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

    meta = {
        "watchlist": path.name,
        "batch_size": args.batch_size,
        "max_symbols": args.max,
        "n_requested": len(items) if args.max is None else min(len(items), args.max),
    }
    if os.environ.get("GITHUB_ACTIONS"):
        meta["runner"] = "github-actions"
        meta["run_id"] = os.environ.get("GITHUB_RUN_ID")
        meta["run_url"] = (
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/"
            f"{os.environ.get('GITHUB_RUN_ID', '')}"
        )
        meta["workflow"] = os.environ.get("GITHUB_WORKFLOW")
    else:
        meta["runner"] = "local"

    try:
        df, failures, scan_meta = scan_universe(
            items,
            batch_size=args.batch_size,
            max_symbols=args.max,
            pause_s=args.pause,
            on_batch=on_batch,
        )
        meta.update(scan_meta or {})
        meta["watchlist"] = path.name
        if args.max is not None:
            meta["max_symbols"] = args.max

        # Soft quality gate: fail CI if almost nothing scored
        min_ok = max(10, int(0.05 * (meta.get("n_requested") or len(items))))
        if meta.get("n_scored", 0) < min_ok:
            msg = (
                f"Too few scored rows ({meta.get('n_scored')} < {min_ok}). "
                "Check yfinance rate limits and global_models/ artifacts."
            )
            _log_to_supabase(
                meta, status="failed", error_message=msg,
                max_symbols=args.max, top_df=df,
            )
            sys.exit(msg)

        csv_path, meta_path = save_rankings(df, failures, meta, directory=args.out)

        print(f"\nDone in {meta['elapsed_s']}s")
        print(f"  scored={meta['n_scored']}  failed={meta['n_failed']}  engine={meta['engine']}")
        print(f"  runner={meta.get('runner')}")
        print(f"  wrote {csv_path}")
        print(f"  wrote {meta_path}")
        if meta.get("run_url"):
            print(f"  run={meta['run_url']}")

        log_res = _log_to_supabase(
            meta, status="success", max_symbols=args.max, top_df=df,
        )
        if log_res.get("skipped"):
            print(f"  supabase log: skipped ({log_res.get('reason')})")
        elif log_res.get("ok"):
            print("  supabase log: inserted")
        else:
            print(f"  supabase log: FAILED — {log_res.get('reason')}")

        print("Commit or sync the rankings/ folder for instant app loads.")
        return 0
    except SystemExit:
        raise
    except Exception as e:
        _log_to_supabase(
            meta, status="failed", error_message=str(e), max_symbols=args.max,
        )
        raise


def _top_symbols_from_df(df, n: int = 10) -> list[str]:
    if df is None or getattr(df, "empty", True):
        return []
    work = df
    if "Screen" in work.columns:
        buys = work[work["Screen"] == "BUY"]
        if not buys.empty:
            work = buys
    if "Buy Score" in work.columns:
        work = work.sort_values("Buy Score", ascending=False)
    if "Symbol" not in work.columns:
        return []
    return work["Symbol"].astype(str).head(n).tolist()


def _log_to_supabase(
    meta: dict,
    *,
    status: str,
    error_message: str | None = None,
    max_symbols=None,
    top_df=None,
) -> dict:
    """Best-effort Supabase insert; never raises into the main path."""
    try:
        return log_rankings_run(
            meta,
            status=status,
            error_message=error_message,
            top_symbols=_top_symbols_from_df(top_df),
            max_symbols=max_symbols,
        )
    except Exception as e:
        print(f"  supabase log: exception — {e}")
        return {"ok": False, "skipped": False, "reason": str(e)}


if __name__ == "__main__":
    raise SystemExit(main())
