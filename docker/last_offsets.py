from kafka import KafkaConsumer, TopicPartition
from kafka.admin import KafkaAdminClient
import json
from datetime import datetime

bootstrap_servers = 'localhost:9092'
admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
all_topics = admin.list_topics()

# Фильтруем топики
topics = [t for t in all_topics if t not in ['__consumer_offsets', '_schemas']]

print("=" * 80)
print("СТАТИСТИКА ПО ТОПИКАМ И ПАРТИЦИЯМ")
print("=" * 80)

consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)

# Для хранения глобального последнего сообщения
global_last = {
    'topic': None,
    'partition': None,
    'offset': -1,
    'timestamp': None,
    'value': None
}

for topic in topics:
    print(f"\n📁 ТОПИК: {topic}")
    print("-" * 60)
    
    partitions = consumer.partitions_for_topic(topic)
    topic_total = 0
    
    for partition in sorted(partitions):
        tp = TopicPartition(topic, partition)
        consumer.assign([tp])
        
        # Получаем первый и последний offset
        beginning_offset = consumer.beginning_offsets([tp])[tp]
        end_offset = consumer.end_offsets([tp])[tp]
        count = end_offset - beginning_offset
        topic_total += count
        
        # Читаем последнее сообщение
        last_msg = None
        last_value = None
        if count > 0:
            consumer.seek(tp, end_offset - 1)
            for msg in consumer:
                last_msg = msg
                try:
                    last_value = json.loads(msg.value.decode('utf-8'))
                except:
                    last_value = msg.value.decode('utf-8')
                break
        
        # Определяем ID сообщения
        msg_id = 'N/A'
        if last_value and isinstance(last_value, dict):
            msg_id = last_value.get('trade_id') or last_value.get('transaction_id') or last_value.get('sensor_id') or last_value.get('reading_id') or f"offset_{last_msg.offset}"
        elif last_value:
            msg_id = str(last_value)[:30]
        
        # Выводим строку таблицы
        print(f"  Partition {partition}: {count:3d} сообщений | Последний offset: {end_offset-1 if count>0 else 0} | ID: {msg_id}")
        
        # Обновляем глобальное последнее сообщение
        if last_msg and last_msg.offset > global_last['offset']:
            global_last['topic'] = topic
            global_last['partition'] = partition
            global_last['offset'] = last_msg.offset
            global_last['timestamp'] = last_msg.timestamp
            global_last['value'] = last_value
    
    print(f"  {'─' * 55}")
    print(f"  ИТОГО в топике: {topic_total} сообщений")

consumer.close()

# Выводим глобальное последнее сообщение
print("\n" + "=" * 80)
print("🏆 САМОЕ ПОСЛЕДНЕЕ СООБЩЕНИЕ ВО ВСЕХ ТОПИКАХ")
print("=" * 80)
print(f"  Топик:        {global_last['topic']}")
print(f"  Партиция:     {global_last['partition']}")
print(f"  Offset:       {global_last['offset']}")
if global_last['timestamp']:
    dt = datetime.fromtimestamp(global_last['timestamp']/1000)
    print(f"  Время:        {dt}")
print(f"  Данные:       {json.dumps(global_last['value'], indent=2, ensure_ascii=False)[:500]}")
print("=" * 80)