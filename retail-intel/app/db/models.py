"""Phase 3: SQLAlchemy models for the retail intelligence backend."""

from __future__ import annotations

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    func,
    Table,
    MetaData,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Zone(Base):
    """Zone definition model."""

    __tablename__ = "zones"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    # Store polygon points as JSON
    points = Column(String, nullable=False)  # JSON string: [[x1,y1],[x2,y2],...]

    # Relationships
    zone_stats = relationship("ZoneStat", back_populates="zone", cascade="all, delete-orphen")


class Detection(Base):
    """Individual detection from Phase 1 tracking."""

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    frame_idx = Column(Integer, nullable=False, index=True)
    timestamp_sec = Column(Float, nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)


class ZoneStat(Base):
    """Pre-computed per-zone statistics aggregated by time window."""

    __tablename__ = "zone_stats"

    id = Column(Integer, primary_key=True, index=True)
    zone_id = Column(Integer, nullable=False, index=True)
    # Time window: bucket start (e.g. "2026-10-12 10:00:00" or integer epoch)
    window_start = Column(String, nullable=False, index=True)
    window_end = Column(String, nullable=False, index=True)
    footfall = Column(Integer, nullable=False)
    avg_dwell_sec = Column(Float, nullable=False)

    # Relationship
    zone = relationship("Zone", back_populates="zone_stats")