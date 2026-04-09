from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys
import time
import os

# --- Logging setup ---
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
logger.add("logs/app.log", rotation="1 MB", level="DEBUG")

# --- App setup ---
app = FastAPI(title="AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Log every request ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    ms = round((time.time() - start) * 1000, 2)
    logger.info(f"← {response.status_code} | {ms}ms")
    return response

@app.get("/")
def root():
    return {"status": "AI Backend running"}

# --- Register routes ---
from api.routes import router
app.include_router(router)