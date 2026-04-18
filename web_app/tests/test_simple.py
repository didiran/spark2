"""
Простые тесты API - не требуют установки fastapi, sqlalchemy и т.д.
"""
import requests
import time
import json

BASE_URL = "http://web-api:8000"

class TestAPI:
    
    @staticmethod
    def test_health():
        print("\n[1] Testing health endpoint...")
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health check passed")
        return True
    
    @staticmethod
    def test_register():
        print("\n[2] Testing registration...")
        username = f"testuser_{int(time.time())}"
        user_data = {
            "username": username,
            "email": f"{username}@test.com",
            "password": "testpass123"
        }
        response = requests.post(f"{BASE_URL}/api/auth/register", json=user_data)
        assert response.status_code in [200, 201]
        print(f"✅ Registration passed for user: {username}")
        return username
    
    @staticmethod
    def test_login(username):
        print("\n[3] Testing login...")
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": username, "password": "testpass123"}
        )
        assert response.status_code == 200
        token = response.json()["access_token"]
        print("✅ Login passed")
        return token
    
    @staticmethod
    def test_send_transaction(token):
        print("\n[4] Testing transaction sending...")
        headers = {"Authorization": f"Bearer {token}"}
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
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        print(f"✅ Transaction sent: {data['transaction_id']}")
        return data["transaction_id"]
    
    @staticmethod
    def test_history(token):
        print("\n[5] Testing transaction history...")
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{BASE_URL}/api/transactions/history",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✅ History retrieved: {len(data)} transactions")
    
    @staticmethod
    def run_all():
        print("=" * 50)
        print("Running API Tests")
        print("=" * 50)
        
        try:
            TestAPI.test_health()
            username = TestAPI.test_register()
            token = TestAPI.test_login(username)
            TestAPI.test_send_transaction(token)
            TestAPI.test_history(token)
            
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED!")
            print("=" * 50)
            return True
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            return False

if __name__ == "__main__":
    success = TestAPI.run_all()
    exit(0 if success else 1)