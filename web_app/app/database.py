"""База данных для пользователей (PostgreSQL)"""

import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Enum, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


class TransactionLog(Base):
    __tablename__ = "transaction_logs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    transaction_id = Column(String(100), nullable=False)
    amount = Column(Float, nullable=False)  # Изменено с Integer на Float
    status = Column(String(50), nullable=False)
    kafka_partition = Column(Integer, nullable=True)
    kafka_offset = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Настройка подключения к PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://pipeline_user:pipeline_pass@postgres:5432/transactions_db"
)

# Создаем engine с настройками для PostgreSQL
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


async def init_db():
    """Инициализация базы данных"""
    # В PostgreSQL нужно создать базу данных, если её нет
    # Для простоты используем существующую ml_pipeline
    Base.metadata.create_all(bind=engine)


def get_db():
    """Получение сессии БД"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()