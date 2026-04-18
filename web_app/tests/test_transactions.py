import pytest
from unittest.mock import patch, MagicMock


def test_send_transaction_unauthorized(client):
    response = client.post(
        "/api/transactions/send",
        json={
            "amount": 100.50,
            "merchant_category": "grocery",
            "card_type": "visa",
            "is_international": False,
            "is_online": True
        }
    )
    assert response.status_code == 401


def test_send_transaction_success(client, auth_headers):
    # Мокаем Kafka producer
    with patch('app.routers.transactions.get_kafka_producer') as mock_kafka:
        mock_producer = MagicMock()
        mock_producer.send_transaction.return_value = (0, 12345)
        mock_kafka.return_value = mock_producer
        
        response = client.post(
            "/api/transactions/send",
            headers=auth_headers,
            json={
                "amount": 150.75,
                "merchant_category": "restaurant",
                "card_type": "mastercard",
                "is_international": False,
                "is_online": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert data["status"] == "sent_to_kafka"
        assert "predicted_fraud_score" in data


def test_send_transaction_invalid_amount(client, auth_headers):
    response = client.post(
        "/api/transactions/send",
        headers=auth_headers,
        json={
            "amount": -10,  # Negative amount
            "merchant_category": "grocery",
            "card_type": "visa",
            "is_international": False,
            "is_online": True
        }
    )
    assert response.status_code == 422  # Validation error


def test_send_transaction_invalid_category(client, auth_headers):
    response = client.post(
        "/api/transactions/send",
        headers=auth_headers,
        json={
            "amount": 100,
            "merchant_category": "invalid_category",
            "card_type": "visa",
            "is_international": False,
            "is_online": True
        }
    )
    assert response.status_code == 422


def test_get_transaction_history(client, auth_headers):
    # Сначала отправляем несколько транзакций
    with patch('app.routers.transactions.get_kafka_producer') as mock_kafka:
        mock_producer = MagicMock()
        mock_producer.send_transaction.return_value = (0, 1)
        mock_kafka.return_value = mock_producer
        
        for _ in range(3):
            client.post(
                "/api/transactions/send",
                headers=auth_headers,
                json={
                    "amount": 100,
                    "merchant_category": "grocery",
                    "card_type": "visa",
                    "is_international": False,
                    "is_online": False
                }
            )
    
    # Получаем историю
    response = client.get("/api/transactions/history", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 3
    assert data[0]["status"] == "sent"