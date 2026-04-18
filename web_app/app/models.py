"""Pydantic модели для валидации данных"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class TransactionRequest(BaseModel):
    """Модель для отправки транзакции"""
    amount: float = Field(..., gt=0, le=100000, description="Сумма транзакции")
    merchant_category: str = Field(..., description="Категория мерчанта")
    card_type: str = Field(..., description="Тип карты")
    is_international: bool = Field(False, description="Международная транзакция")
    is_online: bool = Field(False, description="Онлайн транзакция")
    
    @field_validator('merchant_category')
    @classmethod
    def validate_merchant_category(cls, v: str) -> str:
        valid_categories = ["grocery", "gas_station", "restaurant", "online_retail", "travel", "entertainment"]
        if v not in valid_categories:
            raise ValueError(f'Merchant category must be one of {valid_categories}')
        return v
    
    @field_validator('card_type')
    @classmethod
    def validate_card_type(cls, v: str) -> str:
        valid_types = ["visa", "mastercard", "amex", "discover"]
        if v not in valid_types:
            raise ValueError(f'Card type must be one of {valid_types}')
        return v


class TransactionResponse(BaseModel):
    transaction_id: str
    timestamp: datetime
    status: str
    predicted_fraud_score: Optional[float] = None
    message: str