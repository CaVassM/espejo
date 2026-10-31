"""Acceso a la tabla de análisis (Postgres en Render, SQLite en tu PC y en los tests)."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, LargeBinary, String, Text, create_engine, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

ID_LENGTH = 32
STATUS_LENGTH = 20
STAGE_LENGTH = 60
TOPIC_LENGTH = 300
IN_MEMORY_SQLITE = "sqlite://"


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    """Un análisis: su estado, lo que entregó el worker (VER) y el reporte (INTERPRETAR)."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    topic: Mapped[str | None] = mapped_column(String(TOPIC_LENGTH))
    status: Mapped[str] = mapped_column(String(STATUS_LENGTH))
    stage: Mapped[str] = mapped_column(String(STAGE_LENGTH))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    detections: Mapped[Any | None] = mapped_column(JSON)
    audio_raw: Mapped[Any | None] = mapped_column(JSON)
    heatmap_png: Mapped[bytes | None] = mapped_column(LargeBinary)
    report: Mapped[Any | None] = mapped_column(JSON)


def create_db_engine(database_url: str) -> Engine:
    """Un SQLite en memoria comparte una sola conexión (tests); el resto, el pool normal."""
    if database_url == IN_MEMORY_SQLITE:
        return create_engine(
            database_url, connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    return create_engine(database_url, pool_pre_ping=True)


def create_tables(engine: Engine) -> None:
    """Sin migraciones por ahora: crea la tabla si no existe."""
    Base.metadata.create_all(engine)


class AnalysisRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def add(self, record: AnalysisRecord) -> None:
        with self.session_factory() as session, session.begin():
            session.add(record)

    def get(self, analysis_id: str) -> AnalysisRecord | None:
        with self.session_factory() as session:
            record = session.get(AnalysisRecord, analysis_id)
            if record is not None:
                session.expunge(record)
            return record

    def update_fields(self, analysis_id: str, **fields: Any) -> None:
        with self.session_factory() as session, session.begin():
            record = session.get(AnalysisRecord, analysis_id)
            if record is None:
                return
            for name, value in fields.items():
                setattr(record, name, value)

    def claim(self, analysis_id: str, from_statuses: list[str], **fields: Any) -> bool:
        """Cambia el estado solo si sigue en uno de from_statuses. True si ganó este llamado.

        Evita interpretar dos veces cuando llegan a la vez el aviso del worker y una consulta.
        """
        statement = (
            update(AnalysisRecord)
            .where(AnalysisRecord.id == analysis_id, AnalysisRecord.status.in_(from_statuses))
            .values(**fields, updated_at=utc_now())
        )
        with self.session_factory() as session, session.begin():
            result = session.execute(statement)
            return result.rowcount == 1
