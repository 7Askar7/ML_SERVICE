import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_tracer_initialized = False

def configure_tracer(service_name="mlserv_app"):
    """Настраивает глобальный TracerProvider OpenTelemetry."""
    global _tracer_initialized
    if _tracer_initialized:
        logger.info("Tracer already initialized.")
        return

    logger.info(f"Initializing tracer for service: {service_name}")
    
    resource = Resource(attributes={
        "service.name": service_name
    })

    provider = TracerProvider(resource=resource)

    otlp_exporter = OTLPSpanExporter()

    processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(processor)

    # Устанавливаем глобальный провайдер
    trace.set_tracer_provider(provider)
    _tracer_initialized = True
    logger.info("Tracer initialized successfully.")
