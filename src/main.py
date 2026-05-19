from fastapi import FastAPI
from starlette.requests import Request as StarletteRequest
from src.api import router
from contextlib import asynccontextmanager
from src.pipeline import run_pipeline
import logging
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
# from src.limiter import limiter


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../app.log"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to console
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing...")
    app.state.retriever = run_pipeline()  # Run the pipeline before handling requests
    logger.info("Pipeline initialized successfully.")
    yield
    # logger.info("Shutting down...")

app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Middleware to log requests & responses
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

app.include_router(router)
