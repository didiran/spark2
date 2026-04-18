import json
import random
import time
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ===========================================================
# НАСТРОЙКИ ПРОДЮСЕРА ДЛЯ НАДЁЖНОЙ ДОСТАВКИ (At least once)
# ===========================================================
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    acks='all',                    # ждём подтверждения от всех реплик в ISR
    retries=5,                     # повторяем при ошибке
    enable_idempotence=True,       # ← ДОБАВИТЬ ЭТУ СТРОКУ (включает идемпотентность)
    max_in_flight_requests_per_connection=1,  # сохраняем порядок
    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
)

TOPICS = ["ml-training-data", "fraud-transactions", "iot-sensors", "financial-trades"]

# ===========================================================
# ФУНКЦИИ ГЕНЕРАЦИИ ДАННЫХ
# ===========================================================

def generate_transaction():
    """Генерирует транзакцию для fraud-detection"""
    return {
        "transaction_id": f"tx_{random.randint(10000, 99999)}",
        "timestamp": datetime.now().isoformat(),
        "user_id": f"user_{random.randint(1, 100)}",
        "amount": round(random.uniform(1, 10000), 2),
        "merchant_category": random.choice(["grocery", "gas_station", "restaurant", "online_retail"]),
        "is_fraud": 1 if random.random() < 0.02 else 0,
        "card_type": random.choice(["visa", "mastercard", "amex"]),
        "is_international": 1 if random.random() < 0.1 else 0
    }

def generate_sensor_data():
    """Генерирует данные IoT сенсора"""
    return {
        "reading_id": f"sensor_{random.randint(1000, 9999)}",
        "timestamp": datetime.now().isoformat(),
        "sensor_id": f"SENSOR_{random.randint(1, 50)}",
        "temperature": round(random.uniform(15, 35), 1),
        "humidity": round(random.uniform(30, 80), 1),
        "pressure": round(random.uniform(980, 1020), 1),
        "is_anomaly": 1 if random.random() < 0.05 else 0
    }

def generate_trade():
    """Генерирует финансовую транзакцию"""
    return {
        "trade_id": f"trade_{random.randint(10000, 99999)}",
        "timestamp": datetime.now().isoformat(),
        "account_id": f"ACC_{random.randint(1, 200)}",
        "instrument": random.choice(["equity", "bond", "option", "forex"]),
        "side": random.choice(["buy", "sell"]),
        "quantity": random.randint(1, 1000),
        "price": round(random.uniform(10, 500), 2),
        "is_suspicious": 1 if random.random() < 0.01 else 0
    }

# ===========================================================
# CALLBACK ФУНКЦИИ ДЛЯ ОТСЛЕЖИВАНИЯ СТАТУСА
# ===========================================================

def on_send_success(record_metadata):
    print(f"  ✓ Подтверждено: топик={record_metadata.topic}, партиция={record_metadata.partition}, offset={record_metadata.offset}")

def on_send_error(excp):
    print(f"  ✗ Ошибка: {excp}")

# ===========================================================
# ОСНОВНОЙ ЦИКЛ
# ===========================================================

print("=" * 50)
print("Kafka Data Simulator (At least once mode)")
print("=" * 50)
print(f"Connected to Kafka at localhost:9092")
print(f"Available topics: {', '.join(TOPICS)}")
print("Настройки: acks=all, retries=5")
print("\nSending data... Press Ctrl+C to stop\n")

message_count = 0
batch_size = 10

try:
    while True:
        topic = random.choice(TOPICS)
        
        # Генерируем данные в зависимости от топика
        if topic == "fraud-transactions":
            data = generate_transaction()
        elif topic == "iot-sensors":
            data = generate_sensor_data()
        elif topic == "financial-trades":
            data = generate_trade()
        else:  # ml-training-data
            data = generate_transaction()
            data["feature_values"] = [random.random() for _ in range(5)]
        
        # Асинхронная отправка (не блокирует)
        future = producer.send(topic, value=data)
        future.add_callback(on_send_success).add_errback(on_send_error)
        
        message_count += 1
        msg_id = data.get('transaction_id', data.get('reading_id', data.get('trade_id', 'unknown')))
        print(f"[{message_count:4d}] Sent to '{topic}': {msg_id}")
        
        # Каждые 10 сообщений принудительно отправляем (flush)
        if message_count % batch_size == 0:
            producer.flush()
            print(f"  >>> Flushed {batch_size} messages")
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print(f"\n\nStopping. Total sent: {message_count}")
    producer.flush()  # отправляем всё, что осталось
    producer.close()
    print("Done!")