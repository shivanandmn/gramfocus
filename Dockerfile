# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set build arguments
ARG OPENAI_API_KEY
ARG GOOGLE_API_KEY
ARG GOOGLE_APPLICATION_CREDENTIALS

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENAI_API_KEY=${OPENAI_API_KEY} \
    GOOGLE_API_KEY=${GOOGLE_API_KEY} \
    GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create uploads directory
RUN mkdir -p uploads && chmod 777 uploads

# Expose port
EXPOSE 8000

# Set the default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
