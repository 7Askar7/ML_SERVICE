import os
from celery import Celery
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from ..configs import redis_url, backend_url


print(f"Celery using Broker: {redis_url}")
print(f"Celery using Backend: {backend_url}")

celery_app = Celery(
    "worker",
    broker=redis_url,
    backend=backend_url,
    include=['app.services.asr', 'app.services.llm', 'app.services.tts']
)

CeleryInstrumentor().instrument()

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
    task_create_missing_queues=True,
)