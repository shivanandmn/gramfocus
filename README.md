# GramFocus

GramFocus is an innovative application that helps users improve their English grammar through voice recordings. Users can record themselves speaking on any topic, and the application provides detailed grammar analysis and corrections.

## Features

- Voice recording upload and real-time recording
- Speech-to-text transcription (OpenAI Whisper or Google Speech-to-Text)
- Basic grammar error detection and correction (OpenAI GPT-4 or Google Gemini)
- Detailed explanations for grammar corrections
- User-friendly interface

## Technical Stack

- Backend: FastAPI + Python
- Speech-to-Text: OpenAI Whisper / Google Speech-to-Text
- Grammar Analysis: OpenAI GPT-4 / Google Gemini
- Database: SQLite
- Deployment: Docker + Google Cloud Run

## Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gramfocus.git
cd gramfocus
```

2. Create and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run with Docker Compose:
```bash
docker-compose up --build
```

4. Access the API at `http://localhost:8000`

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest
```

## Deployment

The application is configured for deployment to Google Cloud Run using GitHub Actions.

### Prerequisites

1. Set up Google Cloud:
   ```bash
   # Create a new project (if needed)
   gcloud projects create gramfocus-project
   
   # Set the project
   gcloud config set project gramfocus-project
   
   # Enable required APIs
   gcloud services enable \
     run.googleapis.com \
     artifactregistry.googleapis.com
   
   # Create Artifact Registry repository
   gcloud artifacts repositories create gramfocus \
     --repository-format=docker \
     --location=your-region
   ```

### Service Account Setup

1. **GitHub Actions Deployment Account**:
   ```bash
   # Create a service account for GitHub Actions
   gcloud iam service-accounts create github-actions \
       --description="Service account for GitHub Actions" \
       --display-name="GitHub Actions"

   # Grant necessary permissions
   gcloud projects add-iam-policy-binding your-project-id \
       --member="serviceAccount:github-actions@your-project-id.iam.gserviceaccount.com" \
       --role="roles/run.admin"

   gcloud projects add-iam-policy-binding your-project-id \
       --member="serviceAccount:github-actions@your-project-id.iam.gserviceaccount.com" \
       --role="roles/artifactregistry.admin"

   # Create and download the key
   gcloud iam service-accounts keys create github-actions-key.json \
       --iam-account=github-actions@your-project-id.iam.gserviceaccount.com
   ```

2. **Speech-to-Text Service Account**:
   ```bash
   # Create a service account for Speech-to-Text
   gcloud iam service-accounts create speech-to-text \
       --description="Service account for Speech-to-Text API" \
       --display-name="Speech to Text"

   # Grant Speech-to-Text permissions
   gcloud projects add-iam-policy-binding your-project-id \
       --member="serviceAccount:speech-to-text@your-project-id.iam.gserviceaccount.com" \
       --role="roles/speech.client"

   # Create and download the key
   gcloud iam service-accounts keys create speech-to-text-key.json \
       --iam-account=speech-to-text@your-project-id.iam.gserviceaccount.com
   ```

3. **Set up GitHub Secrets**:
   - Open `github-actions-key.json` and copy its content to a GitHub secret named `GCP_SA_KEY`
   - Open `speech-to-text-key.json` and copy its content to a GitHub secret named `GOOGLE_APPLICATION_CREDENTIALS`
   - Add other required secrets:
     - `GCP_PROJECT_ID`: Your Google Cloud project ID
     - `GCP_REGION`: Your preferred region (e.g., us-central1)
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `GOOGLE_API_KEY`: Your Google API key for Gemini

### Deployment Process

1. Push to main branch triggers automatic deployment
2. GitHub Actions will:
   - Run tests
   - Build Docker image
   - Push to Google Artifact Registry
   - Deploy to Cloud Run

### Manual Deployment

You can also deploy manually using gcloud:

```bash
# Build and push the image
docker build -t gcr.io/your-project/gramfocus .
docker push gcr.io/your-project/gramfocus

# Deploy to Cloud Run
gcloud run deploy gramfocus \
  --image gcr.io/your-project/gramfocus \
  --platform managed \
  --region your-region \
  --allow-unauthenticated
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

MIT
