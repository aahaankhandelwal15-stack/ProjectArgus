"""Phase 3: Database initialization and migration support."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

# Default: use SQLite for local dev; overlay with POSTGRES_URL from .env in production
import os
from pathlib import Path

# Load .env file manually (simple key=value)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./data/retail_intel.db",
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_all() -> None:
    """Create all tables (simple migration for local dev)."""
    Base.metadata.create_all(bind=engine)


def drop_all() -> None:
    """Drop all tables (useful for testing)."""
    Base.metadata.drop_all(bind=engine)