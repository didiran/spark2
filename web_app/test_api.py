#!/usr/bin/env python3
"""
Простой тест API - работает без pytest
"""
import requests
import time
import sys

BASE_URL = "http://web-api:8000"
PASSED = 0
FAILED = 0

def test(name, condition):
    global PASSED, FAILED
    if condition:
        print(f"  ✅ {name} - PASSED")
        PASSED += 1
        return True
    else:
        print(f"  ❌ {name} - FAILED")
        FAILED += 1
        return False

def test_health():
    print("\n[1] Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return test("Health endpoint", response.status_code == 200 and response.json().get("status") == "healthy")
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False

def test_register():
    print("\n[2] User Registration")
    username = f"testuser_{int(time.time())}"
    user_data = {
        "username": username,
        "email": f"{username}@test.com",
        "password": "test123"
    }
    try:
        response = requests.post(f"{BASE_URL}/api/auth/register", json=user_data, timeout=5)
        return test("Registration", response.status_code in [200, 201])
    except Exception as e:
        print(f"  ❌ Registration failed: {e}")
        return False

def test_login():
    print("\n[3] User Login")
    username = f"loginuser_{int(time.time())}"
    user_data = {
        "username": username,
        "email": f"{username}@test.com",
        "password": "test123"
    }
    try:
        # Register first
        requests.post(f"{BASE_URL}/api/auth/register", json=user_data, timeout=5)
        # Login
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": username, "password": "test123"},
            timeout=5
        )
        if response.status_code == 200 and "access_token" in response.json():
            test("Login", True)
            return response.json()["access_token"]
        else:
            test("Login", False)
            return None
    except Exception as e:
        print(f"  ❌ Login failed: {e}")
        return None

def test_transaction(token):
    print("\n[4] Send Transaction")
    if not token:
        test("Send Transaction (skipped - no token)", False)
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    tx_data = {
        "amount": 150.75,
        "merchant_category": "restaurant",
        "card_type": "mastercard",
        "is_international": False,
        "is_online": True
    }
    try:
        response = requests.post(
            f"{BASE_URL}/api/transactions/send",
            json=tx_data,
            headers=headers,
            timeout=5
        )
        return test("Send Transaction", response.status_code == 200 and "transaction_id" in response.json())
    except Exception as e:
        print(f"  ❌ Transaction failed: {e}")
        return False

def test_history(token):
    print("\n[5] Transaction History")
    if not token:
        test("Transaction History (skipped - no token)", False)
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(f"{BASE_URL}/api/transactions/history", headers=headers, timeout=5)
        return test("Transaction History", response.status_code == 200 and isinstance(response.json(), list))
    except Exception as e:
        print(f"  ❌ History failed: {e}")
        return False

def main():
    print("=" * 50)
    print("API Tests")
    print("=" * 50)
    
    test_health()
    test_register()
    token = test_login()
    test_transaction(token)
    test_history(token)
    
    print("\n" + "=" * 50)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 50)
    
    return FAILED == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)