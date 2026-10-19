"""Phase 2: Validation against ground-truth CSV.

Given a ground-truth CSV filled in manually (zone, true_footfall, true_avg_dwell),
compare against computed values and report absolute and percentage error.

Also includes a template ground-truth CSV for the user to fill in.
"""

from __future__ import annotations

import csv
from pathlib import Path


def template_ground_truth_csv(path: str | Path) -> None:
    """Write a template ground-truth CSV file at the given path.

    The user fills in:
        zone, true_footfall, true_avg_dwell_sec

    One row per zone defined in config/zones.json.
    """
    template_path = Path(path)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    with open(template_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["zone", "true_footfall", "true_avg_dwell_sec"])
        # Note: user should replace the placeholder row(s) with actual counts
        writer.writerow(["entrance", "0", "0.0"])


def load_ground_truth(path: str | Path) -> dict[str, dict]:
    """Load ground-truth CSV into a dict keyed by zone name.

    Returns:
        dict mapping zone name -> {"true_footfall": int, "true_avg_dwell_sec": float}
    """
    lookup: dict[str, dict] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zone = row["zone"]
            lookup[zone] = {
                "true_footfall": int(row["true_footfall"]),
                "true_avg_dwell_sec": float(row["true_avg_dwell_sec"]),
            }
    return lookup


def validate_against_ground_truth(
    computed: dict[str, dict],
    ground_truth_path: str | Path,
) -> dict[str, dict]:
    """Compare computedZoneStats against ground-truth and report errors.

    Parameters
    ----------
    computed: output from vision.analytics.compute_zone_stats()
    ground_truth_path: path to CSV laid out by template_ground_truth_csv()

    Returns dict with per-zone absolute and percentage error:
        {
            "zone_name": {
                "absolute_error_footfall": int,
                "percentage_error_footfall_pct": float,
                "absolute_error_avg_dwell_sec": float,
                "percentage_error_avg_dwell_sec_pct": float,
            }
        }
    """
    gt_lookup = load_ground_truth(ground_truth_path)

    validation: dict[str, dict] = {}
    for zone_name, computed_stats in computed.items():
        if zone_name not in gt_lookup:
            print(f"Warning: zone '{zone_name}' not found in ground-truth file.")
            continue

        gt = gt_lookup[zone_name]

        # Footfall error
        abs_footfall = abs(computed_stats["footfall"] - gt["true_footfall"])
        pct_footfall = (
            (abs_footfall / gt["true_footfall"] * 100)
            if gt["true_footfall"] > 0
            else 0.0
        )

        # Average dwell error
        abs_dwell = abs(computed_stats["avg_dwell_sec"] - gt["true_avg_dwell_sec"])
        pct_dwell = (
            (abs_dwell / gt["true_avg_dwell_sec"] * 100)
            if gt["true_avg_dwell_sec"] > 0
            else 0.0
        )

        validation[zone_name] = {
            "absolute_error_footfall": abs_footfall,
            "percentage_error_footfall_pct": round(pct_footfall, 2),
            "absolute_error_avg_dwell_sec": round(abs_dwell, 4),
            "percentage_error_avg_dwell_sec_pct": round(pct_dwell, 2),
        }

    return validation


def print_validation_report(validation: dict[str, dict]) -> None:
    """Print a human-readable validation error report."""
    print(f"{'Zone':<15} {'Footfall Abs':>13} {'Footfall Pct':>11} {'Dwell Abs':>11} {'Dwell Pct':>11}")
    print("-" * 60)
    for zone_name, errors in sorted(validation.items()):
        print(
            f"{zone_name:<15} "
            f"{errors['absolute_error_footfall']:>13} "
            f"{errors['percentage_error_footfall_pct']:>11.2f} "
            f"{errors['absolute_error_avg_dwell_sec']:>11.4f} "
            f"{errors['percentage_error_avg_dwell_sec_pct']:>11.2f}"
        )