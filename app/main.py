from fastapi import FastAPI
import subprocess
import sys
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()

# Check if ffmpeg is available
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    print("✅ ffmpeg is available")
except subprocess.CalledProcessError:
    print("❌ Error: ffmpeg is not installed")
    print("Please install ffmpeg to enable audio format conversion")
    sys.exit(1)
except FileNotFoundError:
    print("❌ Error: ffmpeg is not installed")
    print("Please install ffmpeg to enable audio format conversion")
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
