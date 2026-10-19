# Retail Intelligence Platform

AI-powered system that turns existing CCTV footage into shopper analytics.

## Overview

Three phases built sequentially:

| Phase | Focus | Key Files |
|-------|-------|-----------|
| 1 | Detection + Tracking | `vision/track.py` |
| 2 | Shopper Analytics | `vision/analytics.py`, `vision/draw_zones.py`, `vision/validate.py` |
| 3 | Backend + Storage | `app/main.py`, `app/schemas.py`, `app/db/` |

## Phase 1: Detection + Tracking Pipeline

### `vision/track.py`

Runs YOLOv8n person detection (class 0 only) with built-in ByteTrack tracking.

**Input**: Video file path or RTSP/IP camera URL

**Output**: SQLite at `data/tracking.db` with table `detections`:
- Columns: `frame_idx, timestamp_sec, track_id, x1, y1, x2, y2, confidence`
- Position: bottom-center of bounding box
- Persistent track IDs via ByteTrack (`persist=True`)

**Flags**:
- `--save-video`: save annotated output video (`data/output_tracks.mp4`)
- `--frame-skip N`: process every Nth frame (0 = all) for speed
- `--min-duration N`: ignore tracks shorter than N seconds (Phase 2)

### Run Phase 1

```bash
# Basic usage
python -m vision.track data/test_sample.mp4

# Save annotated output
python -m vision.track data/test_sample.mp4 --save-video

# Skip frames for speed
python -m vision.track data/test_sample.mp4 --frame-skip 3
```

### Test Verification Checklist

- [ ] `data/tracking.db` exists with `detections` table
- [ ] Rows have columns: frame_idx, timestamp_sec, track_id, x1, y1, x2, y2, confidence
- [ ] Track IDs are persistent across frames (ByteTrack)
- [ ] Annotated video saved if `--save-video` used
- [ ] Low-confidence detections filtered (< 0.3)

## Phase 2: Shopper Analytics Module

### `vision/analytics.py`

Computes per-zone stats from Phase 1 detections:

- **Footfall**: unique track IDs that entered the zone
- **Dwell time**: average/median/max seconds each ID spent in the zone
- **Per-minute footfall buckets** for forecasting

Uses ray-casting point-in-polygon on each detection's bottom-center point.
Handles ID flicker: ignores tracks shorter than configurable minimum duration,
merges gaps shorter than threshold when computing dwell.

### `vision/draw_zones.py`

Interactive OpenCV tool for defining zones:

- Left-click: add point to current zone
- Right-click: finish current zone, start new one
- Press 's': save zones to `config/zones.json`
- Press 'q': quit without saving

### `vision/validate.py`

Compares computed stats against a manually-filled ground-truth CSV.

```bash
# Compute and print zone stats
python -m vision.analytics data/tracking.db config/zones.json --csv output.csv

# Validate against ground truth
python -m vision.analytics data/tracking.db config/zones.json --validate tests/ground_truth.csv

# Generate template ground-truth CSV
python -m vision.validate template  # creates tests/ground_truth_template.csv
```

### Run Phase 2 Examples

```bash
# Compute zone stats from tracking DB
python -m vision.analytics data/tracking.db config/zones.json

# Save to CSV
python -m vision.analytics data/tracking.db config/zones.json --csv output.csv

# Validate against ground truth
python -m vision.analytics data/tracking.db config/zones.json --validate tests/ground_truth.csv
```

## Phase 3: Backend + Storage

### FastAPI app at `app/`

**PostgreSQL via SQLAlchemy**: connection string from `.env` file

**Models**:
- `Zone`: id, name, points (JSON)
- `Detection`: frame_idx, timestamp_sec, track_id, x1, y1, x2, y2, confidence
- `ZoneStat`: zone_id, window_start, window_end, footfall, avg_dwell_sec

**Endpoints**:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST /zones` | Create zone |
| `GET /zones` | List zones |
| `POST /ingest` | Load detections from Phase 1 SQLite |
| `GET /stats/footfall` | Footfall per zone over time buckets (hour/day) |
| `GET /stats/dwell` | Dwell time per zone over time range |
| `GET /health` | Health check |

**Auto-generated docs**: `/docs` (Swagger UI), `/redoc` (ReDoc)

### Docker Compose (PostgreSQL)

```yaml
version: "3.8"
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: retail_intel
      POSTGRES_USER: retail_user
      POSTGRES_PASSWORD: retail_pass
    ports: ["5432:5432"]
    volumes: [pg_data:/var/lib/postgresql/data]

  app:
    build: .
    depends_on: [postgres]
    environment:
      DATABASE_URL: postgres://retail_user:retail_pass@postgres:5432/retail_intel
    ports: ["8000:8000"]
volumes:
  pg_data:
```

### Alembic Migration

```bash
cd app
alembic init migrations
# Then: alembic revision --autogenerate -m "initial schema"
#       alembic upgrade head
```

### Pytest Tests

```bash
cd app
pytest tests/ -v
```

## Requirements

```
opencv-python-headless
ultralytics
fastapi
sqlalchemy
psycopg2-binary
email-validator
python-dotenv
pytest
pytest-asyncio
```

## Notes & Design Decisions

- **Zone position**: bottom-center of bounding box (used for point-in-polygon)
- **Minimum track duration**: 0.5 seconds (configurable via `--min-duration` in Phase 1, `min_track_duration_sec` in analytics)
- **Gap threshold**: 2.0 seconds for merging track gaps when computing dwell time (configurable via `gap_threshold_sec`)
- **Bucket granularity**: "hour" or "day" for footfall stats (Phase 3 API)
- **Design forward-looking**: database schema accommodates future forecasting, agent, and frontend features without requiring migrations
- **CPU-only**: YOLOv8n runs on laptop CPU; GPU optional for speed
- **All functions typed and docstringed** for viva explainability