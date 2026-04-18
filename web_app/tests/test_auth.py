import pytest
from app.main import app
from app.database import User


def test_register_success(client, test_user):
    response = client.post("/api/auth/register", json=test_user)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == test_user["username"]
    assert data["email"] == test_user["email"]
    assert "id" in data


def test_register_duplicate_username(client, test_user):
    client.post("/api/auth/register", json=test_user)
    response = client.post("/api/auth/register", json=test_user)
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"]


def test_register_invalid_email(client):
    response = client.post("/api/auth/register", json={
        "username": "user",
        "email": "invalid-email",
        "password": "pass123"
    })
    assert response.status_code == 422  # Validation error


def test_login_success(client, test_user):
    # Сначала регистрируем
    client.post("/api/auth/register", json=test_user)
    
    # Пытаемся залогиниться
    response = client.post(
        "/api/auth/login",
        data={"username": test_user["username"], "password": test_user["password"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0


def test_login_wrong_password(client, test_user):
    client.post("/api/auth/register", json=test_user)
    response = client.post(
        "/api/auth/login",
        data={"username": test_user["username"], "password": "wrongpass"}
    )
    assert response.status_code == 401


def test_login_nonexistent_user(client):
    response = client.post(
        "/api/auth/login",
        data={"username": "nobody", "password": "pass"}
    )
    assert response.status_code == 401