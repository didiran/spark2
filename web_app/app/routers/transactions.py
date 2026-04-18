"""Эндпойнты для отправки транзакций"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
import random

from app.models import TransactionRequest, TransactionResponse
from app.database import get_db, TransactionLog
from app.auth import get_current_user
from app.database import User
from app.kafka_producer import get_kafka_producer, update_stats

router = APIRouter()


@router.post("/send", response_model=TransactionResponse)
async def send_transaction(
    transaction: TransactionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Отправка транзакции в Kafka для анализа ML пайплайном
    
    - Транзакция отправляется в топик 'fraud-transactions'
    - ML пайплайн анализирует и определяет вероятность мошенничества
    """
    
    # Генерируем ID транзакции
    transaction_id = f"tx_{uuid.uuid4().hex[:12]}"
    
    # Формируем сообщение для Kafka
    kafka_message = {
        "transaction_id": transaction_id,
        "timestamp": datetime.now().isoformat(),
        "user_id": current_user.username,
        "amount": transaction.amount,
        "merchant_category": transaction.merchant_category,
        "card_type": transaction.card_type,
        "is_international": 1 if transaction.is_international else 0,
        "is_online": 1 if transaction.is_online else 0,
        "is_fraud": 0,  # Будет определено ML моделью
    }
    
    # Отправка в Kafka
    try:
        kafka = await get_kafka_producer()
        partition, offset = kafka.send_transaction("fraud-transactions", kafka_message)
        
        # Логируем в БД
        log = TransactionLog(
            user_id=current_user.id,
            transaction_id=transaction_id,
            amount=int(transaction.amount),
            status="sent",
            kafka_partition=partition,
            kafka_offset=offset
        )
        db.add(log)
        db.commit()
        
        update_stats(success=True, transaction_id=transaction_id)
        
        # Генерируем "предсказание" (в реальном пайплайне будет от модели)
        # Пока эмулируем случайный fraud score
        fraud_score = random.random() * 100
        
        return TransactionResponse(
            transaction_id=transaction_id,
            timestamp=datetime.now(),
            status="sent_to_kafka",
            predicted_fraud_score=round(fraud_score, 2),
            message=f"Transaction sent to Kafka (partition={partition}, offset={offset})"
        )
        
    except Exception as e:
        update_stats(success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send transaction to Kafka: {str(e)}"
        )


@router.get("/history")
async def get_transaction_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50
):
    """Получить историю транзакций пользователя"""
    
    logs = db.query(TransactionLog).filter(
        TransactionLog.user_id == current_user.id
    ).order_by(TransactionLog.created_at.desc()).limit(limit).all()
    
    return [
        {
            "transaction_id": log.transaction_id,
            "amount": log.amount,
            "status": log.status,
            "kafka_partition": log.kafka_partition,
            "kafka_offset": log.kafka_offset,
            "created_at": log.created_at
        }
        for log in logs
    ]