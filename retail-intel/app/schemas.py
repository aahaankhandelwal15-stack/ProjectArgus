"""Phase 3: Pydantic schemas for request/response bodies."""

from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


# === Zone schemas ===

class ZoneCreate(BaseModel):
    """Input for POST /zones."""
    name: str = Field(..., description="Unique zone name")
    points: List[List[int]] = Field(
        ..., description="Polygon vertices as [[x1,y1],[x2,y2],...]"
    )


class ZoneResponse(BaseModel):
    """Output for GET /zones."""
    id: int
    name: str
    points: List[List[int]]

    class Config:
        from_attributes = True


# === Ingest schemas ===

class IngestSource(BaseModel):
    """Source of detections for POST /ingest."""
    sqlite_db: str = Field(
        ..., description="Path to Phase 1 SQLite output (data/tracking.db)"
    )


class IngestResponse(BaseModel):
    """Output confirming ingestion."""
    records_loaded: int
    message: str


# === Stats schemas ===

class FootfallBucket(BaseModel):
    """A per-minute or per-hour footfall bucket."""
    bucket_start: str  # ISO-style minute/hour string, e.g. "2026-10-12T10:00:00" or "minute:00"
    footfall: int


class FootfallStats(BaseModel):
    zone: str
    start: str
    end: str
    bucket: str  # "hour" or "day"
    footfall_buckets: List[FootfallBucket]


class DwellStats(BaseModel):
    zone: str
    start: str
    end: str
    avg_dwell_sec: float
    median_dwell_sec: float
    max_dwell_sec: float


# === Health schema ===

class HealthResponse(BaseModel):
    status: str = "ok"
    database: str