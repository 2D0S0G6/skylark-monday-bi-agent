"""Profile the supplied seed spreadsheets.

DEVELOPER TOOL ONLY -- this never runs as part of the application. It exists to
document what the source data actually looks like so the normalisation rules can
be written against reality rather than assumptions.

    python -m tools.inspect_seed_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_FILES = {
    "deals": ("seed_data/Deal funnel Data.xlsx", 0),
    "work_orders": ("seed_data/Work_Order_Tracker Data.xlsx", 1),
}


def profile(path: Path, header_row: int) -> None:
    try:
        frame = pd.read_excel(path, sheet_name=0, header=header_row, dtype=str)
    except FileNotFoundError:
        print(f"!! {path} not found — skipping")
        return
    except ImportError:
        print("!! openpyxl is required: pip install -r requirements-dev.txt")
        return

    print("=" * 90)
    print(f"{path}  ->  {frame.shape[0]} rows x {frame.shape[1]} columns")
    print("=" * 90)
    for column in frame.columns:
        non_null = frame[column].notna().sum()
        unique = frame[column].dropna().unique()
        print(f"\n## {column!r}")
        print(f"   populated: {non_null}/{len(frame)}   distinct: {len(unique)}")
        if 0 < len(unique) <= 25:
            print(f"   values: {sorted(map(str, unique))}")
        elif len(unique):
            print(f"   sample: {list(unique[:6])}")
    print(f"\nExact duplicate rows: {frame.duplicated().sum()}")
    echoes = frame.apply(
        lambda row: sum(str(row[c]).strip() == str(c).strip() for c in frame.columns) >= 3,
        axis=1,
    )
    print(f"Rows that echo the header: {int(echoes.sum())}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Profile a single spreadsheet instead of the defaults")
    parser.add_argument("--header-row", type=int, default=0)
    args = parser.parse_args()

    if args.file:
        profile(Path(args.file), args.header_row)
        return 0
    for label, (path, header_row) in DEFAULT_FILES.items():
        print(f"\n\n########## {label} ##########")
        profile(Path(path), header_row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
