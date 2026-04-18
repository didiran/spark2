"""Kafka producer для отправки транзакций"""

from kafka import KafkaProducer as SyncKafkaProducer
from kafka.errors import KafkaError
import json
import asyncio
from datetime import datetime
import random
import uuid
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Глобальный producer
_kafka_producer = None
_stats = {"total_sent": 0, "total_failed": 0, "last_transaction": None}


class KafkaProducerWrapper:
    """Обертка для Kafka producer"""
    
    def __init__(self, bootstrap_servers: str = "kafka-1:29092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        
    def start(self):
        """Запуск producer"""
        try:
            # Исправленная конфигурация для kafka-python
            self.producer = SyncKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                # acks='all' - допустимые значения: 0, 1, 'all'
                acks='all',
                # retries - допустимое значение
                retries=3,
                # max_in_flight_requests_per_connection - допустимо
                max_in_flight_requests_per_connection=1,
                # enable_idempotence НЕ поддерживается в kafka-python, убираем
                # Также убираем request_timeout_ms, если есть проблемы
                request_timeout_ms=30000,
                metadata_max_age_ms=300000
            )
            logger.info(f"Kafka producer connected to {self.bootstrap_servers}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def send_transaction(self, topic: str, transaction: Dict[str, Any]) -> tuple:
        """Отправка транзакции, возвращает (partition, offset) или None"""
        if not self.producer:
            raise RuntimeError("Kafka producer not started")
        
        try:
            future = self.producer.send(topic, value=transaction)
            record_metadata = future.get(timeout=10)
            return record_metadata.partition, record_metadata.offset
        except KafkaError as e:
            logger.error(f"Failed to send to Kafka: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def stop(self):
        """Остановка producer"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer stopped")


# Синглтон
_kafka_wrapper = None


async def get_kafka_producer() -> KafkaProducerWrapper:
    """Получение экземпляра Kafka producer"""
    global _kafka_wrapper
    if _kafka_wrapper is None:
        _kafka_wrapper = KafkaProducerWrapper()
        # Запускаем в отдельном потоке, чтобы не блокировать asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _kafka_wrapper.start)
    return _kafka_wrapper


def get_stats() -> Dict[str, Any]:
    """Получить статистику отправки"""
    global _stats
    return _stats


def update_stats(success: bool, transaction_id: str = None):
    """Обновить статистику"""
    global _stats
    if success:
        _stats["total_sent"] += 1
        _stats["last_transaction"] = transaction_id
    else:
        _stats["total_failed"] += 1


# Для обратной совместимости
kafka_producer = _kafka_wrapper