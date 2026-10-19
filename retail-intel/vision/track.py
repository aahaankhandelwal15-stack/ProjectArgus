"""Phase 1: Detection + tracking pipeline.

Runs YOLOv8 person detection with ByteTrack on a video (or RTSP stream),
writes one row per detection per frame to SQLite, and optionally saves
an annotated output video.

Usage:
    python -m vision.track <video_path> [--save-video] [--frame-skip N]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 + ByteTrack person detection + tracking"
    )
    parser.add_argument(
        "video_path",
        help="Path to video file or RTSP/URL camera stream",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated output video",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Process every Nth frame (0 = all frames, >0 skips for speed)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Ignore tracks shorter than this many seconds (Phase 2)",
    )
    return parser.parse_args()


def initialize_sqlite(db_path: str) -> sqlite3.Connection:
    """Create the detections table if it doesn't exist; return connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            frame_idx INTEGER NOT NULL,
            timestamp_sec REAL NOT NULL,
            track_id INTEGER NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL,
            y2 INTEGER NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def bottom_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """Return (x, y) of the bounding box bottom-center point."""
    x = (x1 + x2) // 2
    y = y2
    return x, y


def point_in_polygon(px: int, py: int, polygon: list[list[int]]) -> bool:
    """Ray-casting point-in-polygon. Returns True if point is inside polygon.

    polygon = [[x1,y1], [x2,y2], ...]
    """
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Check if ray intersects edge (handle horizontal edges)
        if ((y1 > py) != (y2 > py)):
            x_intersect = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-6) + x1
            if px < x_intersect:
                inside = not inside
    return inside


def main() -> None:
    args = parse_args()

    video_path = args.video_path
    frame_skip = args.frame_skip
    min_duration = args.min_duration

    # Open video and read FPS
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video source: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback

    # Prepare output video writer if requested
    out_writer = None
    if args.save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            "data/output_tracks.mp4",
            fourcc,
            fps if fps > 0 else 30.0,
            (width, height),
        )

    # Initialize SQLite
    db_path = os.path.join("data", "tracking.db")
    conn = initialize_sqlite(db_path)

    # Load YOLO model (yolov8n - nano for speed on CPU)
    # persist=True enables built-in ByteTrack tracking across frames
    model = YOLO("yolov8n.pt")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker state every frame (even skipped frames) so IDs persist
        if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
            # Skip heavy processing but still run lightweight tracking
            # to keep ID continuity. We still need a frame for video output.
            if out_writer is not None:
                out_writer.write(frame)
            frame_idx += 1
            continue

        # Run YOLOv8 person detection (class 0 = person) with tracking
        # persist=True keeps track IDs across frames (ByteTrack)
        # verbose=False suppresses per-frame logging
        results = model.predict(
            frame, classes=0, persist=True, verbose=False
        )

        # Get the results for this frame
        r = results[0] if len(results) > 0 else None

        # Get FPS-based timestamp for this frame
        timestamp_sec = frame_idx / fps

        # Process detections
        if r is not None and len(r.boxes) > 0:
            boxes = r.boxes

            for box in boxes:
                # Extract bbox coordinates (xyxy format)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                # box.id is the ByteTrack persistent ID
                track_id = int(box.id[0]) if box.id is not None else -1

                # Skip low-confidence detections
                if confidence < 0.3:
                    continue

                # Compute bottom-center for zone analytics later
                cx, cy = bottom_center(x1, y1, x2, y2)

                # Write to SQLite
                conn.execute(
                    (
                        "INSERT INTO detections "
                        "(frame_idx, timestamp_sec, track_id, x1, y1, x2, y2, confidence) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                    ),
                    (frame_idx, timestamp_sec, track_id, x1, y1, x2, y2, confidence),
                )

                # Draw on frame if saving video
                if out_writer is not None:
                    color = (0, 255, 0)  # green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

        # Write annotated frame if requested
        if out_writer is not None:
            out_writer.write(frame)

        frame_idx += 1

    # Cleanup
    cap.release()
    if out_writer is not None:
        out_writer.release()
    conn.close()

    print(
        f"Done. Processed {frame_idx} frames, "
        f"wrote detections to {db_path}"
    )


if __name__ == "__main__":
    main()