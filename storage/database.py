"""SQLite storage (Day 11, PROJECT_SPEC.md Section 29's `storage/`
folder). Deliberately lightweight -- a single local SQLite file, no
separate database server to run or deploy, per the user's explicit
"lightweight local DB" choice for report history.

Sync SQLAlchemy engine -- async endpoints dispatch repository calls via
`asyncio.to_thread()` (see `storage/repositories.py` callers) rather
than adding an async SQLite driver as a second dependency.
"""
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from storage.models import Base

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "app.db"


def _resolve_database_url() -> str:
    """`DATABASE_URL` env var (Section 22 config-not-hardcoded
    convention) overrides the default local file -- e.g. a persistent
    volume path in a container deployment. Empty/unset falls back to
    `storage/app.db`."""
    configured = get_settings().database_url
    return configured if configured else f"sqlite:///{DEFAULT_DB_PATH}"


DATABASE_URL = _resolve_database_url()

# check_same_thread=False: SQLite connections are per-thread by default,
# but asyncio.to_thread() may dispatch to a different worker thread each
# call. SessionLocal() still opens a fresh connection per call below, so
# this doesn't introduce cross-thread sharing of one connection/session
# -- it just lets the same engine be reused across threads.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Creates tables if they don't exist. Safe to call on every app
    startup -- `create_all` is a no-op for existing tables."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Callers are responsible for closing the returned session (use
    `with get_session() as session:` -- SQLAlchemy's Session is a
    context manager). Tests override this function directly (via
    monkeypatch) to point at an isolated temp database, rather than
    reconfiguring the module-level engine after import."""
    return SessionLocal()
