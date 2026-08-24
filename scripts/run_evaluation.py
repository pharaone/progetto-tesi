#!/usr/bin/env python3
"""Script to run full ISO 42001 gap analysis evaluation and print KPI metrics.

Usage:
    python scripts/run_evaluation.py \
        --org-id acme-corp \
        --docs-dir /path/to/org/docs \
        [--runs 3] \
        [--output-json /path/to/report.json]

KPI Metrics reported:
    TCN (Total Coverage Number): Number of requirements evaluated (target: 53)
    TA  (Test Accuracy): % of cards with valid schema
    IR  (Idempotency Rate): % of verdicts that are identical across N runs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

AS1_URL = os.getenv("AS1_URL", "http://localhost:8001")
AS2_URL = os.getenv("AS2_URL", "http://localhost:8002")
AS3_URL = os.getenv("AS3_URL", "http://localhost:8003")
AGA_URL = os.getenv("AGA_URL", "http://localhost:8004")

AS_TIMEOUT = float(os.getenv("AS_TIMEOUT", "300"))
AGA_TIMEOUT = float(os.getenv("AGA_TIMEOUT", "120"))

EXPECTED_TOTAL_REQUIREMENTS = 62  # 10 (AS-1) + 33 (AS-2) + 19 (AS-3)

REQUIRED_CARD_FIELDS = {
    "requirement_id",
    "requirement_text",
    "verdict",
    "evidences",
    "gaps",
    "corrective_action",
    "execution_metadata",
}

VALID_VERDICTS = {
    "CONFORME",
    "NON_CONFORME",
    "PARZIALMENTE_CONFORME",
    "NON_APPLICABILE",
}

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".rst", ".text"}


def _load_documents(docs_dir: str, org_id: str) -> List[Dict[str, Any]]:
    """Load org documents from directory as DocumentInput dicts."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")

    documents = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in sorted(docs_path.rglob(f"*{ext}")):
            try:
                if ext == ".pdf":
                    try:
                        import io
                        import pypdf

                        with open(file_path, "rb") as f:
                            reader = pypdf.PdfReader(f)
                            pages = []
                            for page in reader.pages:
                                text = page.extract_text()
                                if text:
                                    pages.append(text)
                            content = "\n\n".join(pages)
                    except ImportError:
                        logger.warning(f"pypdf not available, skipping PDF: {file_path.name}")
                        continue
                else:
                    content = file_path.read_text(encoding="utf-8", errors="replace")

                if content.strip():
                    documents.append({
                        "filename": file_path.name,
                        "content": content,
                        "metadata": {"org_id": org_id, "source_path": str(file_path)},
                    })
                    logger.info(f"Loaded document: {file_path.name} ({len(content)} chars)")
            except Exception as exc:
                logger.warning(f"Failed to load {file_path}: {exc}")

    return documents


def _run_single_analysis(
    org_id: str,
    documents: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Run one complete analysis pass. Returns GapReport dict or None on failure."""
    analyze_payload = {"org_id": org_id, "documents": documents}

    as1_cards = None
    as2_cards = None
    as3_cards = None
    failed_agents = []

    # Run all three AS agents
    for url, agent_name in [(AS1_URL, "AS-1"), (AS2_URL, "AS-2"), (AS3_URL, "AS-3")]:
        try:
            logger.info(f"Running {agent_name}...")
            with httpx.Client(timeout=AS_TIMEOUT) as client:
                resp = client.post(f"{url}/analyze", json=analyze_payload)
                resp.raise_for_status()
                cards = resp.json()
                logger.info(f"{agent_name} completed: {len(cards)} cards")

                if agent_name == "AS-1":
                    as1_cards = cards
                elif agent_name == "AS-2":
                    as2_cards = cards
                elif agent_name == "AS-3":
                    as3_cards = cards
        except Exception as exc:
            logger.error(f"{agent_name} failed: {exc}")
            failed_agents.append(agent_name)

    if as1_cards is None and as2_cards is None and as3_cards is None:
        logger.error("All AS agents failed. Cannot produce report.")
        return None

    # Consolidate with AGA
    try:
        logger.info("Running AGA consolidation...")
        consolidate_payload = {
            "org_id": org_id,
            "as1_output": as1_cards,
            "as2_output": as2_cards,
            "as3_output": as3_cards,
            "failed_agents": failed_agents,
        }
        with httpx.Client(timeout=AGA_TIMEOUT) as client:
            resp = client.post(f"{AGA_URL}/consolidate", json=consolidate_payload)
            resp.raise_for_status()
            report = resp.json()
            logger.info(
                f"AGA completed: score={report.get('overall_compliance_score', 'N/A'):.1f}"
            )
            return report
    except Exception as exc:
        logger.error(f"AGA failed: {exc}")
        return None


def _validate_card(card: Dict[str, Any]) -> bool:
    """Return True if card has all required fields and valid verdict."""
    for field in REQUIRED_CARD_FIELDS:
        if field not in card:
            return False
    if card.get("verdict") not in VALID_VERDICTS:
        return False
    ca = card.get("corrective_action", {})
    if not isinstance(ca, dict):
        return False
    if "description" not in ca or "expected_document_type" not in ca:
        return False
    meta = card.get("execution_metadata", {})
    if not isinstance(meta, dict):
        return False
    return True


def _compute_tcn(report: Dict[str, Any]) -> Dict[str, Any]:
    """Compute TCN (Total Coverage Number) metrics."""
    cards = report.get("evaluation_cards", [])
    total = len(cards)
    returned_ids = {c.get("requirement_id") for c in cards}

    return {
        "total_cards": total,
        "expected_total": EXPECTED_TOTAL_REQUIREMENTS,
        "coverage_pct": round(total / EXPECTED_TOTAL_REQUIREMENTS * 100, 1) if EXPECTED_TOTAL_REQUIREMENTS > 0 else 0,
        "requirement_ids_found": sorted(returned_ids),
        "tcn_passed": total >= EXPECTED_TOTAL_REQUIREMENTS,
    }


def _compute_ta(report: Dict[str, Any]) -> Dict[str, Any]:
    """Compute TA (Test Accuracy / schema validity) metrics."""
    cards = report.get("evaluation_cards", [])
    if not cards:
        return {"valid_cards": 0, "total_cards": 0, "accuracy_pct": 0.0, "ta_passed": False}

    valid = sum(1 for c in cards if _validate_card(c))
    return {
        "valid_cards": valid,
        "total_cards": len(cards),
        "accuracy_pct": round(valid / len(cards) * 100, 1),
        "ta_passed": valid == len(cards),
    }


def _compute_ir(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute IR (Idempotency Rate) metrics across multiple runs."""
    if len(reports) < 2:
        return {
            "runs": len(reports),
            "idempotent_requirements": 0,
            "total_requirements": 0,
            "ir_pct": 100.0,
            "ir_passed": True,
            "differing_requirements": [],
        }

    # Build verdict maps per run
    verdict_maps = []
    for report in reports:
        cards = report.get("evaluation_cards", [])
        verdict_maps.append({c["requirement_id"]: c["verdict"] for c in cards})

    # Compare first run against all others
    base = verdict_maps[0]
    all_ids = set(base.keys())
    for vm in verdict_maps[1:]:
        all_ids |= set(vm.keys())

    differing = []
    for req_id in sorted(all_ids):
        verdicts = [vm.get(req_id) for vm in verdict_maps]
        if len(set(v for v in verdicts if v is not None)) > 1:
            differing.append({
                "requirement_id": req_id,
                "verdicts_by_run": verdicts,
            })

    idempotent = len(all_ids) - len(differing)
    ir_pct = round(idempotent / len(all_ids) * 100, 1) if all_ids else 100.0

    return {
        "runs": len(reports),
        "idempotent_requirements": idempotent,
        "total_requirements": len(all_ids),
        "ir_pct": ir_pct,
        "ir_passed": ir_pct == 100.0,
        "differing_requirements": differing,
    }


def print_report(
    org_id: str,
    report: Dict[str, Any],
    tcn: Dict[str, Any],
    ta: Dict[str, Any],
    ir: Dict[str, Any],
    elapsed_seconds: float,
) -> None:
    """Print KPI metrics to stdout."""
    score = report.get("overall_compliance_score", 0.0)
    counts = report.get("counts", {})

    print("\n" + "=" * 70)
    print(f"  ISO/IEC 42001 GAP ANALYSIS REPORT")
    print(f"  Organization: {org_id}")
    print("=" * 70)

    print(f"\n  COMPLIANCE SCORE: {score:.1f}/100")
    print(f"  Total Requirements: {report.get('total_requirements', 'N/A')}")
    print(f"  Conforme:              {counts.get('compliant', 0):>4}")
    print(f"  Non Conforme:          {counts.get('non_compliant', 0):>4}")
    print(f"  Parzialmente Conforme: {counts.get('partial', 0):>4}")
    print(f"  Non Applicabile:       {counts.get('not_applicable', 0):>4}")

    if report.get("partial_coverage"):
        print(f"\n  [!] PARTIAL COVERAGE — Failed agents: {', '.join(report.get('failed_agents', []))}")

    print("\n" + "-" * 70)
    print("  KPI METRICS")
    print("-" * 70)

    # TCN
    tcn_status = "PASS" if tcn["tcn_passed"] else "FAIL"
    print(f"\n  TCN (Total Coverage Number):")
    print(f"    Status:    [{tcn_status}]")
    print(f"    Cards:     {tcn['total_cards']} / {tcn['expected_total']} expected")
    print(f"    Coverage:  {tcn['coverage_pct']:.1f}%")

    # TA
    ta_status = "PASS" if ta["ta_passed"] else "FAIL"
    print(f"\n  TA (Test Accuracy / Schema Validity):")
    print(f"    Status:    [{ta_status}]")
    print(f"    Valid:     {ta['valid_cards']} / {ta['total_cards']}")
    print(f"    Accuracy:  {ta['accuracy_pct']:.1f}%")

    # IR
    ir_status = "PASS" if ir["ir_passed"] else "FAIL"
    print(f"\n  IR (Idempotency Rate — {ir['runs']} runs):")
    print(f"    Status:    [{ir_status}]")
    print(f"    Idempotent: {ir['idempotent_requirements']} / {ir['total_requirements']}")
    print(f"    IR Rate:    {ir['ir_pct']:.1f}%")
    if ir["differing_requirements"]:
        print(f"    Differing requirements:")
        for diff in ir["differing_requirements"][:5]:
            runs_str = " | ".join(f"Run{i+1}:{v}" for i, v in enumerate(diff["verdicts_by_run"]))
            print(f"      - {diff['requirement_id']}: {runs_str}")

    print(f"\n  Analysis time: {elapsed_seconds:.1f}s")

    # Top gaps
    gaps = report.get("prioritized_gaps", [])
    if gaps:
        print(f"\n  TOP PRIORITY GAPS ({len(gaps)} total):")
        for gap in gaps[:5]:
            severity = "CRITICO" if gap.get("severity") == 1 else "MODERATO"
            gaps_text = gap.get("gaps", ["No details"])[:1]
            gap_str = gaps_text[0][:80] if gaps_text else "N/A"
            print(f"    [{severity}] {gap.get('requirement_id')}: {gap_str}")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ISO 42001 gap analysis evaluation and print KPI metrics"
    )
    parser.add_argument("--org-id", required=True, help="Organization identifier")
    parser.add_argument(
        "--docs-dir",
        required=True,
        help="Directory containing organizational documents",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation runs (>=3 for IR metric, default: 1)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the last GapReport as JSON",
    )
    parser.add_argument(
        "--chromadb-path",
        default=None,
        help="Override CHROMADB_PATH environment variable",
    )
    parser.add_argument(
        "--save-all-runs",
        default=None,
        help=(
            "Optional directory to save EVERY run's report as run_<n>.json "
            "(not just the last one). Needed for per-requirement analyses "
            "across runs, e.g. the extended reproducibility test (see "
            "scripts/extended_reproducibility.py)."
        ),
    )

    args = parser.parse_args()

    if args.chromadb_path:
        os.environ["CHROMADB_PATH"] = args.chromadb_path

    logger.info(f"Loading documents from: {args.docs_dir}")
    try:
        documents = _load_documents(args.docs_dir, args.org_id)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if not documents:
        logger.error(f"No documents found in {args.docs_dir}")
        sys.exit(1)

    logger.info(f"Loaded {len(documents)} document(s)")
    logger.info(f"Running {args.runs} evaluation run(s)...")

    save_all_dir = None
    if args.save_all_runs:
        save_all_dir = Path(args.save_all_runs)
        save_all_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    start_time = time.time()

    for run_num in range(1, args.runs + 1):
        logger.info(f"\n--- Run {run_num}/{args.runs} ---")
        run_start = time.time()
        report = _run_single_analysis(args.org_id, documents)
        run_elapsed = time.time() - run_start

        if report is None:
            logger.error(f"Run {run_num} failed. Skipping.")
            continue

        reports.append(report)
        logger.info(
            f"Run {run_num} completed in {run_elapsed:.1f}s. "
            f"Score: {report.get('overall_compliance_score', 'N/A'):.1f}"
        )

        if save_all_dir:
            run_path = save_all_dir / f"run_{run_num}.json"
            run_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            logger.info(f"Run {run_num} report saved to: {run_path}")

    total_elapsed = time.time() - start_time

    if not reports:
        logger.error("All runs failed. No report to display.")
        sys.exit(1)

    # Use last report as primary
    last_report = reports[-1]

    # Compute KPI metrics
    tcn = _compute_tcn(last_report)
    ta = _compute_ta(last_report)
    ir = _compute_ir(reports)

    # Print results
    print_report(
        org_id=args.org_id,
        report=last_report,
        tcn=tcn,
        ta=ta,
        ir=ir,
        elapsed_seconds=total_elapsed,
    )

    # Save JSON output if requested
    if args.output_json:
        output_data = {
            "org_id": args.org_id,
            "gap_report": last_report,
            "kpi_metrics": {
                "tcn": tcn,
                "ta": ta,
                "ir": ir,
            },
            "runs": len(reports),
            "total_elapsed_seconds": round(total_elapsed, 2),
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
        logger.info(f"Report saved to: {args.output_json}")

    # Exit with non-zero if critical KPIs fail
    if not tcn["tcn_passed"] or not ta["ta_passed"]:
        logger.error("One or more critical KPIs failed (TCN or TA).")
        sys.exit(1)


if __name__ == "__main__":
    main()
