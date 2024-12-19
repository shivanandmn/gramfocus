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
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Log the conversion attempt
        file_name = audio_file.filename if isinstance(audio_file, UploadFile) else str(audio_file)
        logger.info(f"Starting audio conversion for file: {file_name}")

        # Handle different input types
        if isinstance(audio_file, Path):
            # For Path objects, read the file directly
            logger.info(f"Reading from Path object: {audio_file}")
            with open(audio_file, 'rb') as f:
                content = f.read()
            original_format = audio_file.suffix[1:].lower()  # Remove the dot from extension
        else:
            # For UploadFile objects
            logger.info(f"Reading from UploadFile object: {audio_file.filename}")
            content = await audio_file.read()
            original_format = audio_file.filename.split('.')[-1].lower()
            await audio_file.seek(0)  # Reset file pointer for UploadFile
        
        logger.info(f"Detected format: {original_format}")
        
        # Save content to a temporary file for ffmpeg processing
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{original_format}")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            # Write content to temp file
            temp_input.write(content)
            temp_input.close()
            temp_output.close()

            # Use ffmpeg directly for more control and better error messages
            cmd = [
                'ffmpeg',
                '-i', temp_input.name,
                '-acodec', 'pcm_s16le',  # Standard WAV codec
                '-ar', '44100',          # Standard sample rate
                '-ac', '2',              # Stereo
                '-y',                    # Overwrite output file
                temp_output.name
            ]
            
            logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg and capture output
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"ffmpeg conversion failed with return code: {process.returncode}")
                logger.error(f"ffmpeg stderr: {process.stderr}")
                logger.error(f"ffmpeg stdout: {process.stdout}")
                raise ValueError(f"Audio conversion failed: {process.stderr}")

            # Read the converted WAV file
            with open(temp_output.name, 'rb') as f:
                wav_data = f.read()

            logger.info(f"Successfully converted {original_format} to WAV format")
            return wav_data, original_format

        finally:
            # Clean up temporary files
            os.remove(temp_input.name)
            os.remove(temp_output.name)
            logger.info("Cleaned up temporary files")

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg process error: {str(e)}")
        logger.error(f"ffmpeg stderr: {e.stderr}")
        logger.error(f"ffmpeg stdout: {e.stdout}")
        raise ValueError(f"ffmpeg process failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}", exc_info=True)
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
