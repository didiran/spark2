"""
Нагрузочное тестирование с эмуляцией нагрузки (аналог send_data.py)
"""

import pytest
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
import statistics


class TestLoadSimulation:
    """Нагрузочные тесты для API"""
    
    @pytest.fixture
    def base_url(self):
        return "http://localhost:8000"  # Адрес запущенного приложения
    
    def test_concurrent_transactions(self, base_url):
        """Тест множества одновременных транзакций"""
        
        def send_transaction(session, transaction_data, token):
            """Отправка одной транзакции"""
            headers = {"Authorization": f"Bearer {token}"}
            response = session.post(
                f"{base_url}/api/transactions/send",
                json=transaction_data,
                headers=headers
            )
            return response.status_code
        
        # Сначала получаем токен
        with aiohttp.ClientSession() as session:
            # Логинимся (один раз для всех потоков)
            login_response = session.post(
                f"{base_url}/api/auth/login",
                data={"username": "testuser", "password": "testpass123"}
            )
            token = login_response.json()["access_token"]
        
        # Генерируем данные для отправки
        import random
        merchants = ["grocery", "restaurant", "online_retail", "travel"]
        card_types = ["visa", "mastercard", "amex"]
        
        transactions = [
            {
                "amount": random.uniform(10, 5000),
                "merchant_category": random.choice(merchants),
                "card_type": random.choice(card_types),
                "is_international": random.choice([True, False]),
                "is_online": random.choice([True, False])
            }
            for _ in range(100)  # 100 транзакций
        ]
        
        # Отправляем конкурентно
        with ThreadPoolExecutor(max_workers=10) as executor:
            with aiohttp.ClientSession() as session:
                futures = []
                start_time = time.time()
                
                for tx in transactions:
                    future = executor.submit(send_transaction, session, tx, token)
                    futures.append(future)
                
                results = [f.result() for f in futures]
                end_time = time.time()
        
        # Анализируем результаты
        success_count = sum(1 for r in results if r == 200)
        duration = end_time - start_time
        
        print(f"\n=== Load Test Results ===")
        print(f"Total transactions: {len(transactions)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(transactions) - success_count}")
        print(f"Duration: {duration:.2f}s")
        print(f"Throughput: {len(transactions) / duration:.2f} tx/s")
        
        assert success_count > len(transactions) * 0.95  # 95% успешных
    
    def test_rate_limit(self, base_url):
        """Тест ограничения скорости отправки"""
        
        # TODO: Добавить rate limiting middleware
        # Сейчас просто проверяем, что большое количество запросов не ломает сервер
        
        import asyncio
        
        async def burst_send():
            async with aiohttp.ClientSession() as session:
                # Логин
                login_data = {"username": "testuser", "password": "testpass123"}
                async with session.post(f"{base_url}/api/auth/login", data=login_data) as resp:
                    token = (await resp.json())["access_token"]
                
                headers = {"Authorization": f"Bearer {token}"}
                
                # Пачка запросов
                tasks = []
                for i in range(50):
                    tx_data = {
                        "amount": 100 + i,
                        "merchant_category": "grocery",
                        "card_type": "visa",
                        "is_international": False,
                        "is_online": True
                    }
                    tasks.append(session.post(f"{base_url}/api/transactions/send", json=tx_data, headers=headers))
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                return responses
        
        responses = asyncio.run(burst_send())
        success_count = sum(1 for r in responses if hasattr(r, 'status') and r.status == 200)
        
        print(f"\n=== Rate Limit Test ===")
        print(f"Total: {len(responses)}")
        print(f"Successful: {success_count}")
        
        # Сервер должен выдержать 50 быстрых запросов
        assert success_count > 40