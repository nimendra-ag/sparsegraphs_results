"""Same table as analysis/all_results.csv, restricted to one NCI screen.

The aggregation logic lives in analysis/build_all_results.py; this only pins
DATASET_ID to the requested screen, redirects the outputs into extra/, and drops
the source=missing placeholders (with one screen in scope most of the 31 methods
have no run at all, and the empty rows would bury the ones that do).

Run:  python extra/build_screen_results.py --id 41
Out:  extra/all_results_nci41.csv, extra/build_report_nci41.txt

--out overrides the CSV name; it is resolved against the repo root.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import build_all_results as base  # noqa: E402


def build(dataset_id: int, out_csv: Path) -> None:
    base.DATASET_ID = dataset_id
    base.OUT_CSV = out_csv
    base.OUT_REPORT = out_csv.parent / f"build_report_nci{dataset_id}.txt"
    base.main()

    with out_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if r["source"] != "missing"]

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nkept {len(rows)} id{dataset_id} rows -> {out_csv.relative_to(ROOT)}")
    print(f"methods: {sorted({r['method'] for r in rows})}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", type=int, required=True, help="NCI screen id, e.g. 33 or 41")
    ap.add_argument("--out", type=Path, default=None, help="output CSV path")
    args = ap.parse_args()
    out = ROOT / args.out if args.out else ROOT / "extra" / f"all_results_nci{args.id}.csv"
    build(args.id, out)


if __name__ == "__main__":
    main()
