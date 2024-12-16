from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import aiofiles
import os
from ..services.transcription import get_transcription_service
from ..services.grammar_analysis import GrammarAnalysisService

router = APIRouter()
transcription_service = get_transcription_service()
grammar_service = GrammarAnalysisService()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/analyze-audio")
async def analyze_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint to analyze audio file for grammar mistakes
    
    1. Save uploaded audio file
    2. Transcribe audio to text using configured provider
    3. Analyze text for grammar mistakes
    4. Return analysis results
    """
    try:
        # Save audio file
        file_path = UPLOAD_DIR / audio_file.filename
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await audio_file.read()
            await out_file.write(content)

        # Transcribe audio using configured service
        transcription = await transcription_service.transcribe_audio(file_path)
        if not transcription:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")

        # Analyze grammar
        analysis = await grammar_service.analyze_text(transcription)

        # Clean up - delete the audio file
        os.remove(file_path)

        return {
            "transcription": transcription,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
