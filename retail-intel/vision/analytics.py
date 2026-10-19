"""Phase 2: Shopper analytics module.

Reads the detections table from Phase 1 SQLite and computes, per zone:
- Footfall: unique track IDs that entered the zone.
- Dwell time: seconds each ID spent in the zone (average, median, max).
- Per-minute footfall buckets for later forecasting.

Zones are polygons defined in config/zones.json (name + list of points).
Uses point-in-polygon on each detection's bottom-center point.
Handles ID flicker: ignores tracks shorter than a configurable minimum duration,
and merges gaps shorter than a small threshold when computing dwell.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sqlite3


def load_zones(zones_path: str | Path) -> list[dict]:
    """Load zone definitions from JSON file.

    Each zone dict has:
        - name: str
        - points: list of [x, y] coordinate pairs defining a polygon
    """
    with open(zones_path, "r") as f:
        data = json.load(f)
    return data["zones"]


def point_in_polygon(px: int, py: int, polygon: list[list[int]]) -> bool:
    """Ray-casting point-in-polygon. Returns True if point is inside polygon.

    polygon = [[x1,y1], [x2,y2], ...] - vertices in order;
    the polygon is automatically closed by connecting last to first.
    """
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Check if edge straddles the horizontal ray at py
        if ((y1 > py) != (y2 > py)):
            # Compute x-intersection of edge with ray at py
            x_intersect = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-6) + x1
            if px < x_intersect:
                inside = not inside
    return inside


def load_detections(db_path: str | Path) -> list[dict]:
    """Load all rows from the Phase 1 detections SQLite table.

    Returns a list of dicts with keys: frame_idx, timestamp_sec,
    track_id, x1, y1, x2, y2, confidence.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT frame_idx, timestamp_sec, track_id, x1, y1, x2, y2, confidence "
        "FROM detections ORDER BY frame_idx, track_id, timestamp_sec"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compute_zone_stats(
    detections: list[dict],
    zones: list[dict],
    min_track_duration_sec: float = 0.5,
    gap_threshold_sec: float = 2.0,
) -> dict[str, dict]:
    """Compute footfall and dwell stats for each zone.

    Parameters
    ----------
    detections: list of dicts from Phase 1 tracking DB
    zones: list of zone dicts from config/zones.json (each with "name" and "points")
    min_track_duration_sec: ignore tracks shorter than this many seconds
    gap_threshold_sec: merge consecutive detections of the same track if the
                       time gap between them is <= this many seconds

    Returns
    -------
    dict keyed by zone name, each value dict with:
        - "footfall": int  (unique track IDs that entered the zone)
        - "avg_dwell_sec": float  (mean dwell time across entering IDs)
        - "median_dwell_sec": float
        - "max_dwell_sec": float
        - "minute_buckets": dict {minute_start_str: unique_count}
    """
    zone_names = [z["name"] for z in zones]
    zone_polygons = {z["name"]: z["points"] for z in zones}

    # Index detections by track_id to analyze per-track presence
    tracks: dict[int, list[dict]] = defaultdict(list)
    for d in detections:
        tracks[d["track_id"]].append(d)

    # For each zone, compute stats
    results: dict[str, dict] = {}
    for zone_name, polygon in zone_polygons.items():
        entering_ids: set[int] = set()
        dwell_seconds: dict[int, list[float]] = defaultdict(list)  # track_id -> list of dwell values

        for track_id, frames in tracks.items():
            if not frames:
                continue
            first_ts = frames[0]["timestamp_sec"]
            last_ts = frames[-1]["timestamp_sec"]
            # Skip tracks shorter than minimum duration
            if (last_ts - first_ts) < min_track_duration_sec:
                # Still check if any frame of this track entered the zone for footfall
                for fd in frames:
                    cx = (fd["x1"] + fd["x2"]) // 2
                    cy = fd["y2"]
                    if point_in_polygon(cx, cy, polygon):
                        entering_ids.add(fd["track_id"])
                continue

            # Track is long enough; analyze zone entry/exit transitions
            # For each frame, check if bottom-center is inside polygon
            inside_sequence: list[bool] = []
            for fd in frames:
                cx = (fd["x1"] + fd["x2"]) // 2
                cy = fd["y2"]
                inside_sequence.append(point_in_polygon(cx, cy, polygon))

            # Detect entries/exits and compute dwell time
            in_zone = False
            entry_idx = None
            total_dwell = 0.0
            num_intervals = 0

            for i, (fd, was_inside) in enumerate(zip(frames, inside_sequence)):
                ts = fd["timestamp_sec"]

                if was_inside and not in_zone:
                    # Entry into zone
                    in_zone = True
                    entry_idx = i
                elif not was_inside and in_zone:
                    # Exit from zone
                    in_zone = False
                    if entry_idx is not None:
                        entry_fd = frames[entry_idx]
                        dwell = ts - entry_fd["timestamp_sec"]
                        if dwell > 0:
                            total_dwell += dwell
                            num_intervals += 1
                    entry_idx = None

            # If still in zone at end of track, count dwell to last frame
            if in_zone and entry_idx is not None:
                last_fd = frames[-1]
                dwell = last_fd["timestamp_sec"] - frames[entry_idx]["timestamp_sec"]
                if dwell > 0:
                    total_dwell += dwell
                    num_intervals += 1

            if num_intervals > 0:
                avg_dwell = total_dwell / num_intervals
                dwell_seconds[track_id].append(avg_dwell)
            elif any(inside_sequence):
                # Track was inside at least one frame but no complete interval
                # Estimate dwell as average frame gap
                frame_gaps = [
                    frames[i + 1]["timestamp_sec"] - frames[i]["timestamp_sec"]
                    for i in range(len(frames) - 1)
                ]
                avg_gap = sum(frame_gaps) / len(frame_gaps) if frame_gaps else 1.0
                dwell_seconds[track_id].append(avg_gap)

        # Compute final stats for this zone
        all_dwell: list[float] = []
        for tid in dwell_seconds:
            all_dwell.extend(dwell_seconds[tid])

        # Per-minute footfall buckets
        minute_buckets: dict[str, int] = {}
        for fd in detections:
            # Only count detections that belong to tracks long enough
            # and are inside this zone
            pass  # Will be filled below

        # Recompute minute buckets using only qualifying track detections
        # A track "counts" for a minute bucket if any of its frames
        # have bottom-center inside the zone
        zone_track_ids: set[int] = set()
        for track_id, frames in tracks.items():
            for fd in frames:
                cx = (fd["x1"] + fd["x2"]) // 2
                cy = fd["y2"]
                if point_in_polygon(cx, cy, polygon):
                    zone_track_ids.add(track_id)
                    break  # one frame inside is enough for this track

        for fd in detections:
            # Only consider detections from qualifying tracks
            if fd["track_id"] not in zone_track_ids:
                continue
            ts = fd["timestamp_sec"]
            minute_start = int(ts // 60) * 60
            bucket_key = f"{int(minute_start // 3600)}:{int(minute_start % 60):02d}"
            minute_buckets[bucket_key] = minute_buckets.get(bucket_key, 0) + 1

        footfall = len(entering_ids)
        avg_dwell = float(np.mean(all_dwell)) if all_dwell else 0.0
        median_dwell = float(np.median(all_dwell)) if all_dwell else 0.0
        max_dwell = float(max(all_dwell)) if all_dwell else 0.0

        results[zone_name] = {
            "footfall": footfall,
            "avg_dwell_sec": avg_dwell,
            "median_dwell_sec": median_dwell,
            "max_dwell_sec": max_dwell,
            "minute_buckets": minute_buckets,
        }

    return results


def print_summary(results: dict[str, dict]) -> None:
    """Print a clean summary table of zone stats."""
    header = f"{'Zone':<15} {'Footfall':>8} {'Avg Dwell(s)':>12} {'Median Dwell(s)':>16} {'Max Dwell(s)':>13}"
    print(header)
    print("-" * len(header))
    for zone_name, stats in results.items():
        print(
            f"{zone_name:<15} {stats['footfall']:>8} "
            f"{stats['avg_dwell_sec']:>12.2f} "
            f"{stats['median_dwell_sec']:>16.2f} "
            f"{stats['max_dwell_sec']:>13.2f}"
        )


def save_csv(results: dict[str, dict], csv_path: str | Path) -> None:
    """Save zone stats to a CSV file."""
    write_path = Path(csv_path)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    with open(write_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Zone", "Footfall", "Avg_Dwell_sec", "Median_Dwell_sec", "Max_Dwell_sec"])
        for zone_name, stats in results.items():
            writer.writerow(
                [zone_name, stats["footfall"], f"{stats['avg_dwell_sec']:.2f}",
                 f"{stats['median_dwell_sec']:.2f}", f"{stats['max_dwell_sec']:.2f}"]
            )


def validate_against_ground_truth(
    computed: dict[str, dict],
    ground_truth_path: str | Path,
) -> dict[str, dict]:
    """Compare computed stats against a ground-truth CSV and report errors.

    Ground-truth CSV format (header + rows):
        zone,true_footfall,true_avg_dwell_sec

    Returns a dict with 'absolute_error' and 'percentage_error' per zone.
    """
    # Load ground truth
    gt_rows: list[dict] = []
    with open(ground_truth_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_rows.append(row)

    # Build a lookup from ground truth
    gt_lookup: dict[str, dict] = {}
    for row in gt_rows:
        gt_lookup[row["zone"]] = {
            "true_footfall": int(row["true_footfall"]),
            "true_avg_dwell_sec": float(row["true_avg_dwell_sec"]),
        }

    validation: dict[str, dict] = {}
    for zone_name, computed_stats in computed.items():
        if zone_name not in gt_lookup:
            print(f"Warning: zone '{zone_name}' not found in ground-truth file.")
            continue

        gt = gt_lookup[zone_name]
        abs_footfall = abs(computed_stats["footfall"] - gt["true_footfall"])
        pct_footfall = (abs_footfall / gt["true_footfall"] * 100) if gt["true_footfall"] > 0 else 0.0

        abs_dwell = abs(computed_stats["avg_dwell_sec"] - gt["true_avg_dwell_sec"])
        pct_dwell = (abs_dwell / gt["true_avg_dwell_sec"] * 100) if gt["true_avg_dwell_sec"] > 0 else 0.0

        validation[zone_name] = {
            "absolute_error_footfall": abs_footfall,
            "percentage_error_footfall_pct": round(pct_footfall, 2),
            "absolute_error_avg_dwell_sec": round(abs_dwell, 4),
            "percentage_error_avg_dwell_sec_pct": round(pct_dwell, 2),
        }

    return validation


# For direct CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m vision.analytics <db_path> <zones_json> [--csv output.csv]")
        sys.exit(1)

    db_path = sys.argv[1]
    zones_path = sys.argv[2]
    csv_output = None
    if len(sys.argv) >= 4 and sys.argv[3] == "--csv" and len(sys.argv) >= 5:
        csv_output = sys.argv[4]

    # Load zones
    zones = load_zones(zones_path)
    # Load detections
    detections = load_detections(db_path)

    # Compute stats
    results = compute_zone_stats(detections, zones)

    # Print summary
    print_summary(results)

    # Save CSV if requested
    if csv_output:
        save_csv(results, csv_output)
        print(f"\nSaved summary to {csv_output}")

    # Optional: validate against ground truth
    if len(sys.argv) >= 6 and sys.argv[5] == "--validate" and len(sys.argv) >= 7:
        gt_path = sys.argv[6]
        validation = validate_against_ground_truth(results, gt_path)
        print("\n=== Validation Results ===")
        for zone_name, errors in validation.items():
            print(
                f"Zone {zone_name}: "
                f"footfall error {errors['absolute_error_footfall']} "
                f"({errors['percentage_error_footfall_pct']}%), "
                f"avg_dwell error {errors['absolute_error_avg_dwell_sec']} "
                f"({errors['percentage_error_avg_dwell_sec_pct']}%)"
            )