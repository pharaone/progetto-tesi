#!/usr/bin/env python3
"""Compare a gap report's verdicts with a ground-truth CSV.

Usage:
    python scripts/compare_ground_truth.py \\
        --report report.json \\
        --ground-truth docs/validation/aerarium_ground_truth.csv

The report can be the orchestrator's /reports/{id} payload, a raw GapReport,
or a run_<n>.json from run_evaluation.py --save-all-runs. The ground truth
needs the columns requirement_id and gt_verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

VERDICTS = ["CONFORME", "PARZIALMENTE_CONFORME", "NON_CONFORME", "NON_APPLICABILE"]
SHORT = {"CONFORME": "C", "PARZIALMENTE_CONFORME": "PC", "NON_CONFORME": "NC", "NON_APPLICABILE": "NA"}
RANK = {"NON_CONFORME": 0, "PARZIALMENTE_CONFORME": 1, "CONFORME": 2}


def _load_cards(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    report = data.get("report") or data.get("gap_report") or data
    return {c["requirement_id"]: c for c in report["evaluation_cards"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()

    cards = _load_cards(args.report)
    with open(args.ground_truth, newline="") as f:
        truth = {row["requirement_id"]: row for row in csv.DictReader(f)}

    missing = sorted(set(truth) - set(cards))
    if missing:
        print(f"Missing from the report: {missing}")

    pairs = [(rid, truth[rid]["gt_verdict"], cards[rid]["verdict"]) for rid in truth if rid in cards]
    matrix = Counter((gt, sysv) for _, gt, sysv in pairs)

    print(f"\nConfusion matrix (rows = ground truth, columns = system), n={len(pairs)}\n")
    print("GT \\ SYS " + "".join(f"{SHORT[v]:>6}" for v in VERDICTS))
    for gt in VERDICTS:
        print(f"{SHORT[gt]:<9}" + "".join(f"{matrix[(gt, sysv)]:>6}" for sysv in VERDICTS))

    exact = sum(1 for _, gt, sysv in pairs if gt == sysv)
    print(f"\nExact agreement: {exact}/{len(pairs)} = {exact / len(pairs) * 100:.1f}%")

    lenient = [(rid, gt, sysv) for rid, gt, sysv in pairs
               if gt in RANK and sysv in RANK and RANK[sysv] > RANK[gt]]
    strict = [(rid, gt, sysv) for rid, gt, sysv in pairs
              if gt in RANK and sysv in RANK and RANK[sysv] < RANK[gt]]
    other = [(rid, gt, sysv) for rid, gt, sysv in pairs
             if gt != sysv and (gt not in RANK or sysv not in RANK)]

    for title, rows in (("System more lenient than ground truth", lenient),
                        ("System stricter than ground truth", strict),
                        ("Other disagreements (N/A involved)", other)):
        print(f"\n{title}: {len(rows)}")
        for rid, gt, sysv in rows:
            print(f"  {rid:<10} GT={SHORT[gt]:<3} SYS={SHORT[sysv]:<3} "
                  f"(confidence: {truth[rid].get('confidenza', '-')})")


if __name__ == "__main__":
    main()
