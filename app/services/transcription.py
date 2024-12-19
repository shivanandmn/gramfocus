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
from pydub import AudioSegment
import io

load_dotenv()
settings = get_settings()

async def convert_audio_to_wav(audio_file: UploadFile) -> tuple[bytes, str]:
    """
    Convert any audio format to WAV format using pydub
    
    Args:
        audio_file: Audio file to convert
        
    Returns:
        tuple: (converted audio bytes, original format)
    """
    try:
        # Read the content of the uploaded file
        content = await audio_file.read()
        
        # Get the original format from the filename
        original_format = audio_file.filename.split('.')[-1].lower()
        
        # Convert the audio using pydub
        audio = AudioSegment.from_file(io.BytesIO(content), format=original_format)
        
        # Export as WAV to a bytes buffer
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format='wav')
        
        # Reset the original file's cursor for potential future reads
        await audio_file.seek(0)
        
        return wav_buffer.getvalue(), original_format
    except Exception as e:
        print(f"Audio conversion error: {str(e)}")
        raise ValueError(f"Failed to convert audio format: {str(e)}")

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

    async def transcribe_audio(self, audio_file: UploadFile, model: str = None) -> Optional[str]:
        """
        Transcribe audio file using Google Speech-to-Text
        
        Args:
            audio_file: Audio file to transcribe
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
