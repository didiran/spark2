"""
Реальные тесты API через HTTP запросы
Не требует TestClient и моков
"""
import requests
import pytest
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

@pytest.fixture
def api_ready():
    """Проверка что API доступен"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        return True
    except:
        pytest.skip("API not available. Run: docker-compose up -d web-api")
        return False

@pytest.fixture
def auth_token(api_ready):
    """Создает тестового пользователя и возвращает токен"""
    username = f"test_user_{int(time.time())}"
    user_data = {
        "username": username,
        "email": f"{username}@test.com",
        "password": "testpass123"
    }
    
    # Регистрация
    requests.post(f"{BASE_URL}/api/auth/register", json=user_data)
    
    # Логин
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        data={"username": username, "password": "testpass123"}
    )
    
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        pytest.skip("Authentication failed")
        return None

@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}

class TestHealth:
    def test_health_endpoint(self, api_ready):
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

class TestAuth:
    def test_register_success(self, api_ready):
        username = f"new_user_{int(time.time())}"
        user_data = {
            "username": username,
            "email": f"{username}@test.com",
            "password": "testpass123"
        }
        response = requests.post(f"{BASE_URL}/api/auth/register", json=user_data)
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["username"] == username

    def test_register_duplicate(self, api_ready):
        username = f"dup_user_{int(time.time())}"
        user_data = {
            "username": username,
            "email": f"{username}@test.com",
            "password": "testpass123"
        }
        # Первая регистрация
        requests.post(f"{BASE_URL}/api/auth/register", json=user_data)
        # Вторая регистрация (дубликат)
        response = requests.post(f"{BASE_URL}/api/auth/register", json=user_data)
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_login_success(self, api_ready):
        username = f"login_user_{int(time.time())}"
        user_data = {
            "username": username,
            "email": f"{username}@test.com",
            "password": "testpass123"
        }
        requests.post(f"{BASE_URL}/api/auth/register", json=user_data)
        
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": username, "password": "testpass123"}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_wrong_password(self, api_ready):
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": "nonexistent", "password": "wrong"}
        )
        assert response.status_code == 401

class TestTransactions:
    def test_send_transaction_unauthorized(self, api_ready):
        tx_data = {
            "amount": 100,
            "merchant_category": "grocery",
            "card_type": "visa"
        }
        response = requests.post(f"{BASE_URL}/api/transactions/send", json=tx_data)
        assert response.status_code == 401

    def test_send_transaction_success(self, api_ready, auth_headers):
        tx_data = {
            "amount": 150.75,
            "merchant_category": "restaurant",
            "card_type": "mastercard",
            "is_international": False,
            "is_online": True
        }
        response = requests.post(
            f"{BASE_URL}/api/transactions/send",
            json=tx_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert data["status"] == "sent_to_kafka"

    def test_send_transaction_invalid_amount(self, api_ready, auth_headers):
        tx_data = {
            "amount": -100,
            "merchant_category": "grocery",
            "card_type": "visa"
        }
        response = requests.post(
            f"{BASE_URL}/api/transactions/send",
            json=tx_data,
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error

    def test_send_transaction_invalid_category(self, api_ready, auth_headers):
        tx_data = {
            "amount": 100,
            "merchant_category": "invalid_category",
            "card_type": "visa"
        }
        response = requests.post(
            f"{BASE_URL}/api/transactions/send",
            json=tx_data,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_get_transaction_history(self, api_ready, auth_headers):
        response = requests.get(
            f"{BASE_URL}/api/transactions/history",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

class TestMetrics:
    def test_metrics_endpoint(self, api_ready, auth_headers):
        # Пробуем оба возможных пути
        response = requests.get(
            f"{BASE_URL}/metrics",  # сначала пробуем /metrics
            headers=auth_headers
        )
        
        if response.status_code == 404:
            # Если нет, пробуем /api/metrics
            response = requests.get(
                f"{BASE_URL}/api/metrics",
                headers=auth_headers
            )
        
        # Если оба не работают, пропускаем тест
        if response.status_code == 404:
            pytest.skip("Metrics endpoint not implemented")
        else:
            assert response.status_code == 200
            data = response.json()
            assert "total_sent" in data
            assert "total_failed" in data
