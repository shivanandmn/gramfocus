from fastapi import FastAPI
import subprocess
import sys
import logging
import logging.handlers
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            'app.log',
            maxBytes=10000000,  # 10MB
            backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Check if ffmpeg is available
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    logger.info("✅ ffmpeg is available")
except subprocess.CalledProcessError as e:
    logger.error("❌ Error: ffmpeg check failed")
    logger.error(f"Error output: {e.stderr}")
    sys.exit(1)
except FileNotFoundError:
    logger.error("❌ Error: ffmpeg is not installed")
    logger.error("Please install ffmpeg to enable audio format conversion")
    sys.exit(1)

app = FastAPI(title=settings.APP_NAME, version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME} API"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}
