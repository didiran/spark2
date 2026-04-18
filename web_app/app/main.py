"""
FastAPI веб-приложение для отправки транзакций в Kafka
Интеграция с ML pipeline для fraud detection
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from contextlib import asynccontextmanager
import logging
import os

# Импорты из нашего приложения
from app.database import init_db, get_db
from app.routers import auth, transactions
from app.kafka_producer import get_kafka_producer
from app.auth import get_current_user

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Starting web application...")
    await init_db()
    await get_kafka_producer()  # Инициализация Kafka producer
    logger.info("Web application started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down web application...")
    from app.kafka_producer import kafka_producer
    if kafka_producer:
        await kafka_producer.stop()


app = FastAPI(
    title="Fraud Detection Transaction API",
    description="API для отправки транзакций в Kafka для ML пайплайна",
    version="1.0.0",
    lifespan=lifespan
)

# Подключаем роутеры
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(transactions.router, prefix="/api/transactions", tags=["transactions"])

# Статика и шаблоны (с проверкой существования папок)
static_dir = "static"
templates_dir = "templates"

# Создаем папки если их нет
import os
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

# Монтируем статику только если папка не пустая
if os.path.exists(static_dir) and os.listdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/metrics")
async def get_metrics(current_user = Depends(get_current_user)):
    """Получить метрики отправленных транзакций (только для авторизованных)"""
    from app.kafka_producer import get_stats
    return get_stats()
    
@app.get("/health")
async def health_check():
    """Health check для мониторинга"""
    return {"status": "healthy", "service": "fraud-transaction-api"}


@app.get("/metrics")
async def get_metrics(current_user = Depends(get_current_user)):
    """Получить метрики отправленных транзакций (только для авторизованных)"""
    from app.kafka_producer import get_stats
    return get_stats()