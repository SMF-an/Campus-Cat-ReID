#!/usr/bin/env python3
"""Threshold tuning for Campus-Cat-ReID.

Reads the baseline predictions CSV, sweeps confirmed/uncertain threshold
combinations, and finds the optimal three-tier decision boundaries:

  - Confirmed:  score >= confirmed_threshold  →  return Top1
  - Uncertain:  uncertain_threshold <= score < confirmed_threshold  →  return Top3
  - Unknown:    score < uncertain_threshold  →  return "unknown"

Usage:
    PYTHONPATH=. uv run python scripts/evaluate_threshold_tuning.py
    PYTHONPATH=. uv run python scripts/evaluate_threshold_tuning.py \
        --predictions outputs/split_baseline/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> list[dict]:
    """Load predictions.csv from baseline evaluation."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "query_path": r["query_path"],
                "true_cat_id": r["true_cat_id"],
                "true_cat_name": r["true_cat_name"],
                "pred_cat_id": r["pred_cat_id"],
                "pred_cat_name": r["pred_cat_name"],
                "top1_score": float(r["top1_score"]),
                "top1_correct": int(r["top1_correct"]),
                "top3_correct": int(r["top3_correct"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def evaluate_thresholds(
    predictions: list[dict],
    confirmed_threshold: float,
    uncertain_threshold: float,
) -> dict:
    """Evaluate one (confirmed, uncertain) threshold pair."""
    total = len(predictions)
    if total == 0:
        return {}

    n_confirmed = 0
    confirmed_correct = 0
    n_uncertain = 0
    uncertain_top3_correct = 0
    n_unknown = 0

    for p in predictions:
        score = p["top1_score"]
        if score >= confirmed_threshold:
            n_confirmed += 1
            confirmed_correct += p["top1_correct"]
        elif score >= uncertain_threshold:
            n_uncertain += 1
            uncertain_top3_correct += p["top3_correct"]
        else:
            n_unknown += 1

    confirmed_precision = confirmed_correct / n_confirmed if n_confirmed else 0.0
    confirmed_coverage = n_confirmed / total
    uncertain_top3_hit = uncertain_top3_correct / n_uncertain if n_uncertain else 0.0
    uncertain_coverage = n_uncertain / total
    unknown_rate = n_unknown / total

    # Effective accuracy: confirmed correct + uncertain Top3 correct (human picks right)
    effective_correct = confirmed_correct + uncertain_top3_correct
    effective_accuracy = effective_correct / total

    return {
        "confirmed_threshold": round(confirmed_threshold, 4),
        "uncertain_threshold": round(uncertain_threshold, 4),
        "n_confirmed": n_confirmed,
        "confirmed_correct": confirmed_correct,
        "confirmed_precision": round(confirmed_precision, 4),
        "confirmed_coverage": round(confirmed_coverage, 4),
        "n_uncertain": n_uncertain,
        "uncertain_top3_correct": uncertain_top3_correct,
        "uncertain_top3_hit_rate": round(uncertain_top3_hit, 4),
        "uncertain_coverage": round(uncertain_coverage, 4),
        "n_unknown": n_unknown,
        "unknown_rate": round(unknown_rate, 4),
        "effective_correct": effective_correct,
        "effective_accuracy": round(effective_accuracy, 4),
    }


def parse_range(s: str) -> list[float]:
    """Parse a range string like '0.70:0.96:0.05' into a list of floats."""
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid range: {s!r} — expected start:stop:step")
    start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
    vals = []
    v = start
    while v < stop - 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals


# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------

def compute_score_distribution(predictions: list[dict], n_bins: int = 20) -> dict:
    """Compute histogram of top1_scores, split by correct/incorrect."""
    correct_scores = [p["top1_score"] for p in predictions if p["top1_correct"]]
    incorrect_scores = [p["top1_score"] for p in predictions if not p["top1_correct"]]

    all_scores = [p["top1_score"] for p in predictions]
    if not all_scores:
        return {}

    lo = min(all_scores)
    hi = max(all_scores)
    bins = np.linspace(lo, hi, n_bins + 1).tolist()

    correct_hist, _ = np.histogram(correct_scores, bins=bins)
    incorrect_hist, _ = np.histogram(incorrect_scores, bins=bins)

    return {
        "n_bins": n_bins,
        "bin_edges": [round(b, 4) for b in bins],
        "correct_counts": correct_hist.tolist(),
        "incorrect_counts": incorrect_hist.tolist(),
        "score_stats": {
            "all": {
                "mean": round(float(np.mean(all_scores)), 4),
                "std": round(float(np.std(all_scores)), 4),
                "min": round(float(np.min(all_scores)), 4),
                "max": round(float(np.max(all_scores)), 4),
                "median": round(float(np.median(all_scores)), 4),
            },
            "correct": {
                "mean": round(float(np.mean(correct_scores)), 4),
                "std": round(float(np.std(correct_scores)), 4),
                "median": round(float(np.median(correct_scores)), 4),
            } if correct_scores else {},
            "incorrect": {
                "mean": round(float(np.mean(incorrect_scores)), 4),
                "std": round(float(np.std(incorrect_scores)), 4),
                "median": round(float(np.median(incorrect_scores)), 4),
            } if incorrect_scores else {},
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(
    sweep_results: list[dict],
    optimal: dict,
    distribution: dict,
    total: int,
) -> str:
    """Generate a human-readable Markdown report."""
    lines = []
    lines.append("# Threshold Tuning Report\n")

    # Score distribution summary
    stats = distribution.get("score_stats", {})
    all_s = stats.get("all", {})
    corr_s = stats.get("correct", {})
    incorr_s = stats.get("incorrect", {})

    lines.append("## Score Distribution\n")
    lines.append(f"| Metric | All queries | Correct Top1 | Incorrect Top1 |")
    lines.append(f"|--------|-------------|--------------|----------------|")
    lines.append(f"| Mean   | {all_s.get('mean',0):.4f} | {corr_s.get('mean',0):.4f} | {incorr_s.get('mean',0):.4f} |")
    lines.append(f"| Median | {all_s.get('median',0):.4f} | {corr_s.get('median',0):.4f} | {incorr_s.get('median',0):.4f} |")
    lines.append(f"| Std    | {all_s.get('std',0):.4f} | {corr_s.get('std',0):.4f} | {incorr_s.get('std',0):.4f} |")
    lines.append("")

    # Optimal thresholds
    lines.append("## Optimal Thresholds\n")
    lines.append(f"- **Confirmed threshold**: {optimal['confirmed_threshold']}")
    lines.append(f"- **Uncertain threshold**: {optimal['uncertain_threshold']}")
    lines.append("")
    lines.append("### Metrics at Optimal\n")
    lines.append(f"| Tier | Count | Coverage | Precision / Hit Rate |")
    lines.append(f"|------|-------|----------|---------------------|")
    lines.append(f"| Confirmed | {optimal['n_confirmed']} | {optimal['confirmed_coverage']:.2%} | {optimal['confirmed_precision']:.2%} (Top1) |")
    lines.append(f"| Uncertain | {optimal['n_uncertain']} | {optimal['uncertain_coverage']:.2%} | {optimal['uncertain_top3_hit_rate']:.2%} (Top3) |")
    lines.append(f"| Unknown   | {optimal['n_unknown']} | {optimal['unknown_rate']:.2%} | — |")
    lines.append("")
    lines.append(f"- **Effective accuracy**: {optimal['effective_accuracy']:.2%}")
    lines.append(f"  (confirmed correct + uncertain Top3 correct) / {total}")
    lines.append("")

    # Top 10 sweep results table
    lines.append("## Top 10 Threshold Combinations\n")
    lines.append("Sorted by effective_accuracy (confirmed_precision ≥ 0.95).\n")
    lines.append("| # | Confirmed | Uncertain | Conf Prec | Conf Cov | Unc Top3 Hit | Unc Cov | Unknown | Eff Acc |")
    lines.append("|---|-----------|-----------|-----------|----------|--------------|---------|---------|---------|")
    for i, r in enumerate(sweep_results[:10], 1):
        lines.append(
            f"| {i} | {r['confirmed_threshold']:.2f} | {r['uncertain_threshold']:.2f} "
            f"| {r['confirmed_precision']:.2%} | {r['confirmed_coverage']:.2%} "
            f"| {r['uncertain_top3_hit_rate']:.2%} | {r['uncertain_coverage']:.2%} "
            f"| {r['unknown_rate']:.2%} | {r['effective_accuracy']:.2%} |"
        )
    lines.append("")

    lines.append("## Decision Logic\n")
    lines.append("```")
    lines.append("if top1_score >= confirmed_threshold:")
    lines.append("    return CONFIRMED  # Top1 answer, high confidence")
    lines.append("elif top1_score >= uncertain_threshold:")
    lines.append("    return UNCERTAIN  # Top3 candidates, human review")
    lines.append("else:")
    lines.append("    return UNKNOWN    # Not in database")
    lines.append("```\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Threshold tuning for cat ReID")
    parser.add_argument(
        "--predictions", default="outputs/split_baseline/predictions.csv", type=str,
        help="Path to baseline predictions CSV",
    )
    parser.add_argument(
        "--out-dir", default="outputs/threshold_tuning", type=str,
    )
    parser.add_argument(
        "--confirmed-range", default="0.70:0.96:0.05", type=str,
        help="Range for confirmed threshold (start:stop:step)",
    )
    parser.add_argument(
        "--uncertain-range", default="0.40:0.76:0.05", type=str,
        help="Range for uncertain threshold (start:stop:step)",
    )
    parser.add_argument(
        "--min-confirmed-precision", default=0.95, type=float,
        help="Minimum confirmed precision for optimal selection",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    predictions = load_predictions(Path(args.predictions))
    total = len(predictions)
    logger.info("Loaded %d predictions from %s", total, args.predictions)

    # Score distribution
    distribution = compute_score_distribution(predictions)
    with open(out_dir / "score_distribution.json", "w", encoding="utf-8") as f:
        json.dump(distribution, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", out_dir / "score_distribution.json")

    # Sweep thresholds
    confirmed_vals = parse_range(args.confirmed_range)
    uncertain_vals = parse_range(args.uncertain_range)
    logger.info("Sweeping %d confirmed × %d uncertain = %d combinations",
                len(confirmed_vals), len(uncertain_vals),
                len(confirmed_vals) * len(uncertain_vals))

    sweep_results = []
    for ct in confirmed_vals:
        for ut in uncertain_vals:
            if ut >= ct:
                continue  # uncertain must be < confirmed
            result = evaluate_thresholds(predictions, ct, ut)
            sweep_results.append(result)

    # Sort by effective_accuracy descending
    sweep_results.sort(key=lambda r: r["effective_accuracy"], reverse=True)

    with open(out_dir / "threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s (%d combos)", out_dir / "threshold_sweep.json", len(sweep_results))

    # Find optimal: max effective_accuracy subject to confirmed_precision >= min
    min_prec = args.min_confirmed_precision
    feasible = [r for r in sweep_results if r["confirmed_precision"] >= min_prec]
    if feasible:
        optimal = feasible[0]  # already sorted by effective_accuracy
    else:
        # Relax: pick the one with highest confirmed_precision among top effective_accuracy
        logger.warning(
            "No combo meets confirmed_precision >= %.2f; picking best available",
            min_prec,
        )
        optimal = sweep_results[0]

    optimal_output = {
        "confirmed_threshold": optimal["confirmed_threshold"],
        "uncertain_threshold": optimal["uncertain_threshold"],
        "min_confirmed_precision_constraint": min_prec,
        "metrics": optimal,
        "rationale": (
            f"Maximizes effective_accuracy={optimal['effective_accuracy']:.4f} "
            f"subject to confirmed_precision >= {min_prec}. "
            f"At confirmed_threshold={optimal['confirmed_threshold']}, "
            f"{optimal['n_confirmed']}/{total} queries are confirmed with "
            f"{optimal['confirmed_precision']:.2%} precision. "
            f"At uncertain_threshold={optimal['uncertain_threshold']}, "
            f"{optimal['n_uncertain']}/{total} queries need human review with "
            f"{optimal['uncertain_top3_hit_rate']:.2%} Top3 hit rate. "
            f"{optimal['n_unknown']}/{total} queries are unknown."
        ),
    }
    with open(out_dir / "optimal_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(optimal_output, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", out_dir / "optimal_thresholds.json")

    # Generate report
    report_md = generate_report(sweep_results, optimal, distribution, total)
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Wrote %s", out_dir / "report.md")

    # Summary
    logger.info("=" * 50)
    logger.info("THRESHOLD TUNING SUMMARY")
    logger.info("=" * 50)
    logger.info("Optimal confirmed_threshold: %.2f", optimal["confirmed_threshold"])
    logger.info("Optimal uncertain_threshold: %.2f", optimal["uncertain_threshold"])
    logger.info("Confirmed: %d queries (%.1f%%), precision=%.2f%%",
                optimal["n_confirmed"], optimal["confirmed_coverage"] * 100,
                optimal["confirmed_precision"] * 100)
    logger.info("Uncertain: %d queries (%.1f%%), Top3 hit=%.2f%%",
                optimal["n_uncertain"], optimal["uncertain_coverage"] * 100,
                optimal["uncertain_top3_hit_rate"] * 100)
    logger.info("Unknown:   %d queries (%.1f%%)",
                optimal["n_unknown"], optimal["unknown_rate"] * 100)
    logger.info("Effective accuracy: %.2f%%", optimal["effective_accuracy"] * 100)
    logger.info("Outputs written to: %s", out_dir)


if __name__ == "__main__":
    main()
