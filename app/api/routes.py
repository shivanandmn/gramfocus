from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pathlib import Path
import aiofiles
import os
import json
from app.services.transcription import get_transcription_service
from app.services.grammar_analysis import GrammarAnalysisService
from app.core.config import LLMProvider, TranscriptionProvider, get_settings
from app.models.grammar import (
    GrammarAnalysis,
    TranscriptAnalysisRequest,
    AudioAnalysisResponse,
    ProviderInfo,
    ProvidersInfo,
    TranscriptAnalysisResponse,
)

router = APIRouter()
transcription_service = get_transcription_service()
grammar_service = GrammarAnalysisService()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(
    audio_file: UploadFile = File(...),
    transcription_provider: TranscriptionProvider = Query(default=None),
    transcription_model: str = Query(default=None),
    analysis_provider: LLMProvider = Query(default=None),
    analysis_model: str = Query(default=None),
):
    """
    Endpoint to analyze audio file for grammar mistakes

    Args:
        audio_file: The audio file to analyze
        transcription_provider: Provider for transcription (openai/google)
        transcription_model: Model name for transcription
        analysis_provider: Provider for grammar analysis (openai/gemini)
        analysis_model: Model name for analysis
    """
    try:
        settings = get_settings()

        # Use provided values or fallback to defaults from settings
        trans_provider = transcription_provider or settings.TRANSCRIPTION_PROVIDER
        trans_model = transcription_model or (
            settings.OPENAI_WHISPER_MODEL
            if trans_provider == TranscriptionProvider.OPENAI
            else None  # Add Google model default if needed
        )

        anal_provider = analysis_provider or settings.LLM_PROVIDER
        anal_model = analysis_model or (
            settings.OPENAI_CHAT_MODEL
            if anal_provider == LLMProvider.OPENAI
            else settings.GEMINI_MODEL
        )

        # Save audio file
        file_path = UPLOAD_DIR / audio_file.filename
        async with aiofiles.open(file_path, "wb") as out_file:
            content = await audio_file.read()
            await out_file.write(content)

        # Get transcription service with specified provider and model
        transcription = await transcription_service.transcribe_audio(
            file_path, model=trans_model
        )
        if not transcription:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")

        # Analyze grammar with specified provider and model
        analysis = await grammar_service.analyze_text(
            transcription, provider=anal_provider, model=anal_model
        )

        # Clean up - delete the audio file
        os.remove(file_path)

        return {
            "transcription": transcription,
            "analysis": analysis,
            "providers": {
                "transcription": {"provider": trans_provider, "model": trans_model},
                "analysis": {"provider": anal_provider, "model": anal_model},
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/analyze-audio", response_model=AudioAnalysisResponse)
async def test_analyze_audio():
    """
    Test endpoint that returns a sample audio analysis response

    Returns:
        AudioAnalysisResponse: A sample response loaded from analyze_audio_response.json
    """
    try:
        # Path to the sample response file
        response_file = Path("data/results/analyze_audio_response.json")

        # Read and return the sample response
        with open(response_file, "r") as f:
            sample_response = json.load(f)
            return sample_response

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Sample response file not found")
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, detail="Error parsing sample response file"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-transcript", response_model=TranscriptAnalysisResponse)
async def analyze_transcript(request: TranscriptAnalysisRequest):
    """
    Endpoint to analyze transcript text directly for grammar mistakes

    Args:
        request: TranscriptAnalysisRequest containing:
            - transcript: The text to analyze
            - llm_provider: Which LLM provider to use (openai/gemini)
            - model_name: Optional specific model name to use

    Returns:
        TranscriptAnalysisResponse: Analysis results including corrections, improved version, and provider info
    """
    try:
        settings = get_settings()

        # Get the model name based on provider
        model_name = request.model_name
        if not model_name:
            # Use default models based on provider
            model_name = (
                settings.OPENAI_CHAT_MODEL
                if request.llm_provider == LLMProvider.OPENAI
                else settings.GEMINI_MODEL
            )

        # Perform grammar analysis
        analysis_result = await grammar_service.analyze_text(
            request.transcript, provider=request.llm_provider, model=model_name
        )

        # Create response with provider info
        return TranscriptAnalysisResponse(
            analysis=analysis_result,
            provider_info=ProviderInfo(
                provider=request.llm_provider.value, model=model_name
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
