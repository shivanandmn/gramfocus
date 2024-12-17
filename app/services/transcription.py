import openai
from google.cloud import speech
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from .transcription_base import TranscriptionService
from ..core.config import get_settings, TranscriptionProvider
from openai import AsyncOpenAI
import tempfile
from fastapi import UploadFile

load_dotenv()
settings = get_settings()

class OpenAITranscriptionService(TranscriptionService):
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)

    async def transcribe_audio(self, audio_file: UploadFile, model: str = None) -> Optional[str]:
        """
        Transcribe audio file using OpenAI Whisper API
        
        Args:
            audio_file: Audio file to transcribe
            model: Optional model name to use (defaults to settings)
        """
        try:
            # Save the uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            try:
                contents = await audio_file.read()
                temp_file.write(contents)
                temp_file.close()

                with open(temp_file.name, "rb") as audio:
                    transcript = await self.client.audio.transcriptions.create(
                        model=model or self.settings.OPENAI_WHISPER_MODEL,
                        file=audio
                    )
                return transcript.text
            finally:
                os.remove(temp_file.name)
        except Exception as e:
            print(f"OpenAI Transcription Error: {str(e)}")
            return None

class GoogleTranscriptionService(TranscriptionService):
    def __init__(self):
        self.client = speech.SpeechClient()

    async def transcribe_audio(self, audio_file: UploadFile, model: str = None) -> Optional[str]:
        """
        Transcribe audio file using Google Speech-to-Text
        
        Args:
            audio_file: Audio file to transcribe
            model: Optional model name to use (e.g., 'phone_call', 'video', 'command_and_search')
        """
        try:
            # Save the uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            try:
                contents = await audio_file.read()
                temp_file.write(contents)
                temp_file.close()

                # Read the audio file
                with open(temp_file.name, "rb") as audio_file:
                    content = audio_file.read()

                # Configure the recognition
                audio = speech.RecognitionAudio(content=content)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                    enable_automatic_punctuation=True,
                    model=model if model else "default"  # Use specified model or default
                )

                # Perform the transcription
                response = self.client.recognize(config=config, audio=audio)

                # Combine all transcriptions
                transcript = ""
                for result in response.results:
                    transcript += result.alternatives[0].transcript + " "

                return transcript.strip()
            finally:
                os.remove(temp_file.name)
        except Exception as e:
            print(f"Google Transcription Error: {str(e)}")
            return None

def get_transcription_service(provider: TranscriptionProvider = None) -> TranscriptionService:
    """
    Factory function to get the configured transcription service
    
    Args:
        provider: Optional provider override (defaults to settings)
    """
    provider = provider or settings.TRANSCRIPTION_PROVIDER
    
    if provider == TranscriptionProvider.OPENAI:
        return OpenAITranscriptionService()
    elif provider == TranscriptionProvider.GOOGLE:
        return GoogleTranscriptionService()
    else:
        raise ValueError(f"Unsupported transcription provider: {provider}")
