"""
data_loader.py
Load the Yelp Open Dataset and filter to one metropolitan area.

The group chose Philadelphia (Section 3.1 / Section 3.2 deliverable by Sai
Teja Phanitina). This module reproduces the same canonical subset
(df_business_Philadelphia ~ 14,569 rows; df_review_philadelphia ~ 967,552
rows) so the Section 3.5 application runs on the same data as the rest of
the report.

Usage from another script:
    from data_loader import load_business, philly_business_ids, iter_philly_reviews

Standalone smoke tests:
    python data_loader.py stats
    python data_loader.py sample-reviews --n 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

# ---------- paths ----------

_HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _HERE.parent / "dataset"
BUSINESS_FILE = DEFAULT_DATA_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = DEFAULT_DATA_DIR / "yelp_academic_dataset_review.json"

CITY = "Philadelphia"


# ---------- loaders ----------

def load_business(path: Path = BUSINESS_FILE) -> pd.DataFrame:
    """Load the full Yelp business.json as a DataFrame.

    The business file is ~114 MB and fits comfortably in memory.
    """
    if not path.exists():
        raise FileNotFoundError(f"Yelp business file not found at {path}")
    return pd.read_json(path, lines=True)


def philly_businesses(df_business: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the Philadelphia subset of the business DataFrame."""
    if df_business is None:
        df_business = load_business()
    return df_business.loc[df_business["city"] == CITY].reset_index(drop=True)


def philly_business_ids(df_business: pd.DataFrame | None = None) -> set[str]:
    """Set of business_id values for the Philadelphia subset."""
    return set(philly_businesses(df_business)["business_id"].tolist())


def iter_philly_reviews(
    business_ids: Iterable[str] | None = None,
    path: Path = REVIEW_FILE,
    limit: int | None = None,
) -> Iterator[dict]:
    """Stream review.json line-by-line, yielding only Philadelphia reviews.

    The review file is ~5 GB; we never load it as a DataFrame. The caller
    receives a generator of dicts (one per review). Pass `limit` to cap.
    """
    if business_ids is None:
        business_ids = philly_business_ids()
    ids = set(business_ids)
    if not path.exists():
        raise FileNotFoundError(f"Yelp review file not found at {path}")
    yielded = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("business_id") in ids:
                yield obj
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


# ---------- standalone smoke tests ----------

def _cmd_stats(args: argparse.Namespace) -> None:
    df = load_business()
    print(f"Total businesses in Yelp dataset: {len(df):,}")
    philly = philly_businesses(df)
    print(f"  ... of which in {CITY}: {len(philly):,}")

    if args.count_reviews:
        ids = set(philly["business_id"].tolist())
        n = 0
        for _ in iter_philly_reviews(business_ids=ids, limit=args.review_cap):
            n += 1
            if n % 100_000 == 0:
                print(f"  ... seen {n:,} {CITY} reviews", file=sys.stderr)
        print(f"  Total {CITY} reviews scanned: {n:,}"
              f"{' (capped at --review-cap)' if args.review_cap and n >= args.review_cap else ''}")


def _cmd_sample_reviews(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    samples: list[dict] = []
    for r in iter_philly_reviews(limit=args.scan_cap):
        if len(samples) < args.n:
            samples.append(r)
        else:
            j = rng.randrange(0, len(samples) + 1)
            if j < args.n:
                samples[j] = r
    for i, r in enumerate(samples, 1):
        text = r.get("text", "").replace("\n", " ")
        if len(text) > 400:
            text = text[:397] + "..."
        print(f"[{i}] business_id={r.get('business_id')} stars={r.get('stars')}")
        print(f"    {text}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="print row counts")
    s.add_argument("--count-reviews", action="store_true",
                   help="also stream review.json and count Philadelphia reviews (slow)")
    s.add_argument("--review-cap", type=int, default=None,
                   help="stop counting after this many review rows scanned")
    s.set_defaults(func=_cmd_stats)

    s = sub.add_parser("sample-reviews", help="print N random Philadelphia reviews")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--scan-cap", type=int, default=50_000,
                   help="how many reviews to scan before stopping reservoir sample")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=_cmd_sample_reviews)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
