"""Acceso a la tabla de análisis (Postgres en Render, SQLite en tu PC y en los tests)."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    LargeBinary,
    String,
    Text,
    create_engine,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

ID_LENGTH = 32
STATUS_LENGTH = 20
STAGE_LENGTH = 60
TOPIC_LENGTH = 300
KEY_HASH_LENGTH = 64
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
    # Hash de la clave del docente (nunca la clave): agrupa los análisis que ve su asistente
    teacher_key_hash: Mapped[str | None] = mapped_column(String(KEY_HASH_LENGTH), index=True)


def create_db_engine(database_url: str) -> Engine:
    """Un SQLite en memoria comparte una sola conexión (tests); el resto, el pool normal."""
    if database_url == IN_MEMORY_SQLITE:
        return create_engine(
            database_url, connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
    return create_engine(database_url, pool_pre_ping=True)


def create_tables(engine: Engine) -> None:
    """Sin migraciones por ahora: crea la tabla si no existe y agrega columnas nuevas."""
    Base.metadata.create_all(engine)
    add_missing_columns(engine)


def add_missing_columns(engine: Engine) -> None:
    """La base de Render ya existía sin teacher_key_hash: create_all no agrega columnas."""
    columns = set()
    for column in inspect(engine).get_columns(AnalysisRecord.__tablename__):
        columns.add(column["name"])
    if "teacher_key_hash" in columns:
        return
    with engine.begin() as connection:
        connection.execute(
            text(f"ALTER TABLE analyses ADD COLUMN teacher_key_hash VARCHAR({KEY_HASH_LENGTH})")
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_analyses_teacher_key_hash "
                "ON analyses (teacher_key_hash)"
            )
        )


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

    def key_exists(self, teacher_key_hash: str) -> bool:
        statement = select(AnalysisRecord.id).where(
            AnalysisRecord.teacher_key_hash == teacher_key_hash
        )
        with self.session_factory() as session:
            return session.scalars(statement.limit(1)).first() is not None

    def list_by_key(self, teacher_key_hash: str) -> list[AnalysisRecord]:
        """Los análisis de una clave, del más reciente al más antiguo."""
        statement = (
            select(AnalysisRecord)
            .where(AnalysisRecord.teacher_key_hash == teacher_key_hash)
            .order_by(AnalysisRecord.created_at.desc())
        )
        with self.session_factory() as session:
            records = list(session.scalars(statement).all())
            for record in records:
                session.expunge(record)
            return records

    def get_owned(self, analysis_id: str, teacher_key_hash: str) -> AnalysisRecord | None:
        """None si no existe o si es de otra clave: las dos cosas se ven igual desde afuera."""
        record = self.get(analysis_id)
        if record is None or record.teacher_key_hash != teacher_key_hash:
            return None
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
