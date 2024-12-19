import openai
from google.cloud import speech
import os
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv
from .transcription_base import TranscriptionService
from ..core.config import get_settings, TranscriptionProvider
from openai import AsyncOpenAI
import tempfile
from fastapi import UploadFile
from pydub import AudioSegment
import io

load_dotenv()
settings = get_settings()

async def convert_audio_to_wav(audio_file: Union[UploadFile, Path]) -> tuple[bytes, str]:
    """
    Convert any audio format to WAV format using pydub
    
    Args:
        audio_file: Audio file to convert (can be UploadFile or Path)
        
    Returns:
        tuple: (converted audio bytes, original format)
    """
    try:
        # Handle different input types
        if isinstance(audio_file, Path):
            # For Path objects, read the file directly
            with open(audio_file, 'rb') as f:
                content = f.read()
            original_format = audio_file.suffix[1:].lower()  # Remove the dot from extension
        else:
            # For UploadFile objects
            content = await audio_file.read()
            original_format = audio_file.filename.split('.')[-1].lower()
            await audio_file.seek(0)  # Reset file pointer for UploadFile
        
        # Convert the audio using pydub
        audio = AudioSegment.from_file(io.BytesIO(content), format=original_format)
        
        # Export as WAV to a bytes buffer
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format='wav')
        wav_buffer.seek(0)
        
        return wav_buffer.getvalue(), original_format
    except Exception as e:
        print(f"Audio conversion error: {str(e)}")
        raise ValueError(f"Failed to convert audio format: {str(e)}")

class OpenAITranscriptionService(TranscriptionService):
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)

    async def transcribe_audio(self, audio_file: Union[UploadFile, Path], model: str = None) -> Optional[str]:
        """
        Transcribe audio file using OpenAI Whisper API
        
        Args:
            audio_file: Audio file to transcribe (can be UploadFile or Path)
            model: Optional model name to use (defaults to settings)
        """
        try:
            # Convert audio to WAV format if it's not already
            wav_data, original_format = await convert_audio_to_wav(audio_file)
            
            # Save the converted WAV data temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            try:
                temp_file.write(wav_data)
                temp_file.close()

                with open(temp_file.name, "rb") as audio:
                    transcript = await self.client.audio.transcriptions.create(
                        model=model or self.settings.OPENAI_WHISPER_MODEL,
                        file=audio
                    )
                return transcript.text
            finally:
                os.remove(temp_file.name)
        except ValueError as ve:
            print(f"Audio conversion error: {str(ve)}")
            return None
        except Exception as e:
            print(f"OpenAI Transcription Error: {str(e)}")
            return None

class GoogleTranscriptionService(TranscriptionService):
    def __init__(self):
        self.client = speech.SpeechClient()

    async def transcribe_audio(self, audio_file: Union[UploadFile, Path], model: str = None) -> Optional[str]:
        """
        Transcribe audio file using Google Speech-to-Text
        
        Args:
            audio_file: Audio file to transcribe (can be UploadFile or Path)
            model: Optional model name to use (e.g., 'phone_call', 'video', 'command_and_search')
        """
        try:
            # Convert audio to WAV format if it's not already
            wav_data, original_format = await convert_audio_to_wav(audio_file)
            
            # Save the converted WAV data temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            try:
                temp_file.write(wav_data)
                temp_file.close()

                # Read the audio file
                with open(temp_file.name, "rb") as audio_file:
                    content = audio_file.read()

                audio = speech.RecognitionAudio(content=content)

                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,  # Standard sample rate
                    language_code="en-US",
                    model=model if model else "default",
                )

                # Detect speech in the audio file
                response = self.client.recognize(config=config, audio=audio)
                
                # Combine all transcriptions
                transcript = " ".join(
                    result.alternatives[0].transcript
                    for result in response.results
                    if result.alternatives
                )
                
                return transcript.strip()
            finally:
                os.remove(temp_file.name)
        except ValueError as ve:
            print(f"Audio conversion error: {str(ve)}")
            return None
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
