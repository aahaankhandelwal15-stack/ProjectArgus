"""Phase 2: Interactive zone definition tool.

Allows the user to click points on a video frame to define zones
and saves them to config/zones.json.

Usage:
    python -m vision.draw_zones <video_path>

Controls:
    - Left-click: add a point to the current zone
    - Right-click: finish current zone and start a new one
    - Press 's': save zones and exit
    - Press 'q': quit without saving
"""

from __future__ import annotations

import json
import numpy as np
import os
from pathlib import Path

import cv2


# Default zones structure if user saves before drawing any zones
DEFAULT_ZONES: dict = {"zones": [{"name": "zone_1", "points": [[100, 100], [500, 100], [500, 200], [100, 200]]}]}


class ZoneDrawer:
    """Interactive zone drawer using OpenCV."""

    def __init__(self, video_path: str, zones_path: str = "config/zones.json"):
        self.video_path = video_path
        self.zones_path = zones_path
        self.zones: list[dict] = DEFAULT_ZONES["zones"]
        self.current_zone_name = "zone_1"
        self.current_points: list[list[int]] = []
        self.window_name = "Draw Zones - Click points (Right-click to finish zone)"
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: cannot open video: {video_path}")
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Get first frame for reference
        self.ref_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ref_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Load existing zones if the file exists
        if os.path.exists(self.zones_path):
            with open(self.zones_path, "r") as f:
                existing = json.load(f)
            # Merge: keep existing zones, don't overwrite with defaults
            if "zones" in existing and existing["zones"]:
                self.zones = existing["zones"]
            # Ensure each zone has a name
            for z in self.zones:
                if "name" not in z:
                    z["name"] = f"zone_{len(self.zones) + 1}"

        # State
        self.drawing = False
        self.frame = None

    def on_mouse(self, event, x, y, flags, param):
        """Handle mouse events for point collection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_points.append([x, y])
            self._draw_current_frame()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current zone: if we have at least 3 points, finalize it
            if len(self.current_points) >= 3:
                # Close the polygon by connecting last to first
                zone_entry = {"name": self.current_zone_name, "points": self.current_points}
                self.zones.append(zone_entry)
                print(f"Zone '{self.current_zone_name}' finalized with {len(self.current_points)} points.")
                # Reset for next zone
                self.current_zone_name = f"zone_{len(self.zones) + 1}"
                self.current_points = []
            self._draw_current_frame()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_points:
                self._draw_current_frame()

    def _draw_current_frame(self) -> None:
        """Render the current frame with points and zone preview."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize if too large for display
        disp = frame.copy()
        # Draw current points
        for i, pt in enumerate(self.current_points):
            cv2.circle(disp, tuple(pt), 5, (0, 255, 255), -1)
            cv2.putText(disp, str(i), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Draw completed zones semi-transparently
        for zone in self.zones:
            pts = zone["points"]
            if len(pts) >= 3:
                pts_np = np.array(pts, dtype=np.int32)
                # Draw polygon outline
                cv2.polylines(disp, [pts_np], isClosed=True, color=(255, 0, 0), thickness=2)
                # Fill semi-transparent
                overlay = disp.copy()
                cv2.fillPoly(overlay, [pts_np], color=(255, 0, 0))
                cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)

        # Add instructions
        cv2.putText(disp, "Left-click: add point  |  Right-click: finish zone  |  Press 's' to save  |  Press 'q' to quit",
                    (20, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.window_name, disp)

    def run(self) -> None:
        """Main loop: show frame, handle mouse callbacks."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        # Read first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, self.frame = self.cap.read()
        if not ret:
            print("Warning: could not read first frame")
            self.frame = None

        self._draw_current_frame()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting without saving.")
                break
            elif key == ord("s"):
                self._save_zones()
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _save_zones(self) -> None:
        """Save zones to JSON file, adding any zones that were drawn."""
        # Finalize any zone currently being drawn
        if len(self.current_points) >= 3:
            zone_entry = {"name": self.current_zone_name, "points": self.current_points}
            self.zones.append(zone_entry)
            print(f"Zone '{self.current_zone_name}' saved with {len(self.current_points)} points.")
        elif len(self.current_points) > 0:
            # Not enough points; still add what we have
            zone_entry = {"name": self.current_zone_name, "points": self.current_points}
            self.zones.append(zone_entry)
            print(f"Zone '{self.current_zone_name}' saved with {len(self.current_points)} points (incomplete).")

        # Ensure all zones have names
        for i, z in enumerate(self.zones):
            if "name" not in z:
                z["name"] = f"zone_{i + 1}"

        # Write to file
        os.makedirs(os.path.dirname(self.zones_path) or ".", exist_ok=True)
        with open(self.zones_path, "w") as f:
            json.dump({"zones": self.zones}, f, indent=2)

        print(f"Zones saved to {self.zones_path}")


def main() -> None:
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_sample.mp4"
    drawer = ZoneDrawer(video_path)
    drawer.run()


if __name__ == "__main__":
    main()