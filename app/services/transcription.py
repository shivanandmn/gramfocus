import openai
from google.cloud import speech
import os
from pathlib import Path
from typing import Optional, Union, TypeVar, Callable, Awaitable
from dotenv import load_dotenv
from .transcription_base import TranscriptionService, convert_audio_to_wav
from ..core.config import get_settings, TranscriptionProvider
from openai import AsyncOpenAI
import tempfile
from fastapi import UploadFile
import io
import logging
import pydub
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()
settings = get_settings()

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

class AsyncWorkerPool:
    """
    Generic async worker pool for parallel processing tasks
    """
    def __init__(self, max_workers: int = 3):
        """
        Initialize worker pool
        
        Args:
            max_workers: Maximum number of concurrent workers (default: 3)
        """
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        
    async def process_item(self, 
                         item: T, 
                         process_func: Callable[[T], Awaitable[R]], 
                         item_id: int = None) -> R:
        """
        Process a single item with semaphore control
        
        Args:
            item: Item to process
            process_func: Async function to process the item
            item_id: Optional identifier for logging
        """
        try:
            async with self.semaphore:
                logger.info(f"Processing item {item_id} in worker pool")
                return await process_func(item)
        except Exception as e:
            logger.error(f"Error processing item {item_id}: {str(e)}", exc_info=True)
            raise
            
    async def process_batch(self, 
                          items: list[T], 
                          process_func: Callable[[T], Awaitable[R]]) -> list[R]:
        """
        Process a batch of items concurrently
        
        Args:
            items: List of items to process
            process_func: Async function to process each item
        """
        tasks = [
            self.process_item(item, process_func, idx) 
            for idx, item in enumerate(items)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

async def split_audio_into_chunks(wav_data: bytes, chunk_size_mb: int = 10) -> list[bytes]:
    """
    Split audio data into chunks of specified size
    
    Args:
        wav_data: WAV format audio data
        chunk_size_mb: Maximum size of each chunk in MB (default: 10)
        
    Returns:
        List of audio chunks in bytes
    """
    try:
        # Convert bytes to AudioSegment
        audio_segment = pydub.AudioSegment.from_wav(io.BytesIO(wav_data))
        
        # Calculate chunk duration based on size
        # Approximate calculation: 1 minute of WAV ~ 10MB (stereo, 44.1kHz)
        chunk_duration = (chunk_size_mb * 60 * 1000) // 10  # in milliseconds
        
        chunks = []
        for i in range(0, len(audio_segment), chunk_duration):
            chunk = audio_segment[i:i + chunk_duration]
            # Convert chunk to WAV bytes
            chunk_buffer = io.BytesIO()
            chunk.export(chunk_buffer, format='wav')
            chunks.append(chunk_buffer.getvalue())
            
        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to split audio: {str(e)}")

class OpenAITranscriptionService(TranscriptionService):
    def __init__(self, max_workers: int = 3):
        super().__init__(max_workers=max_workers)
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)

    async def prepare_audio_chunk(self, chunk: bytes, **kwargs) -> bytes:
        """
        OpenAI accepts WAV format directly, so no preparation needed
        """
        return chunk

    async def transcribe_chunk(self, chunk: bytes, **kwargs) -> Optional[str]:
        """
        Transcribe a single chunk using OpenAI Whisper API
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            temp_file.write(chunk)
            temp_file.close()

            with open(temp_file.name, "rb") as audio:
                transcript = await self.client.audio.transcriptions.create(
                    model=kwargs.get('model') or self.settings.OPENAI_WHISPER_MODEL,
                    file=audio
                )
            return transcript.text
        finally:
            os.remove(temp_file.name)

    async def transcribe_audio(self, audio_file: Union[UploadFile, Path], model: str = None) -> Optional[str]:
        """
        Transcribe audio file using OpenAI Whisper API with parallel chunk processing
        """
        try:
            # Convert audio to WAV format
            wav_data, original_format = await convert_audio_to_wav(audio_file)
            
            # Split into chunks
            chunks = await split_audio_into_chunks(wav_data)
            logger.info(f"Split audio into {len(chunks)} chunks for parallel processing")
            
            # Process chunks in parallel
            async def process_chunk(chunk: bytes) -> Optional[str]:
                prepared_chunk = await self.prepare_audio_chunk(chunk)
                return await self.transcribe_chunk(prepared_chunk, model=model)
            
            transcriptions = await self.worker_pool.process_batch(chunks, process_chunk)
            
            # Handle errors
            errors = [t for t in transcriptions if isinstance(t, Exception)]
            if errors:
                error_msg = "; ".join(str(e) for e in errors)
                logger.error(f"Errors during parallel transcription: {error_msg}")
                return None
            
            # Combine successful transcriptions
            valid_transcriptions = [t for t in transcriptions if isinstance(t, str)]
            final_transcript = " ".join(valid_transcriptions)
            
            logger.info("Successfully transcribed all audio chunks in parallel")
            return final_transcript
            
        except ValueError as ve:
            logger.error(f"Audio conversion error: {str(ve)}")
            return None
        except Exception as e:
            logger.error(f"OpenAI Transcription Error: {str(e)}", exc_info=True)
            return None

class GoogleTranscriptionService(TranscriptionService):
    def __init__(self, max_workers: int = 3):
        super().__init__(max_workers=max_workers)
        self.client = speech.SpeechClient()

    async def prepare_audio_chunk(self, chunk: bytes, **kwargs) -> bytes:
        """
        Convert audio chunk to format required by Google Speech-to-Text
        """
        # Google requires specific audio encoding
        audio_segment = pydub.AudioSegment.from_wav(io.BytesIO(chunk))
        buffer = io.BytesIO()
        audio_segment.export(buffer, format='wav', parameters=[
            '-acodec', 'pcm_s16le',  # Linear PCM encoding
            '-ar', '16000'           # 16kHz sample rate
        ])
        return buffer.getvalue()

    async def transcribe_chunk(self, chunk: bytes, **kwargs) -> Optional[str]:
        """
        Transcribe a single chunk using Google Speech-to-Text
        """
        try:
            # Create the recognition audio object
            audio = speech.RecognitionAudio(content=chunk)

            # Configure recognition settings
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=kwargs.get('language_code', 'en-US'),
                enable_automatic_punctuation=True,
                model=kwargs.get('model', 'default')
            )

            # Perform the transcription
            operation = self.client.recognize(config=config, audio=audio)
            
            # Extract the transcription text
            if operation.results:
                return " ".join(
                    alternative.transcript
                    for result in operation.results
                    for alternative in result.alternatives
                )
            return None

        except Exception as e:
            logger.error(f"Google transcription error: {str(e)}", exc_info=True)
            return None

    async def transcribe_audio(self, audio_file: Union[UploadFile, Path], model: str = None) -> Optional[str]:
        """
        Transcribe audio file using Google Speech-to-Text with parallel chunk processing
        """
        try:
            # Convert audio to WAV format
            wav_data, original_format = await convert_audio_to_wav(audio_file)
            
            # Split into chunks
            chunks = await split_audio_into_chunks(wav_data)
            logger.info(f"Split audio into {len(chunks)} chunks for parallel processing")
            
            # Process chunks in parallel
            async def process_chunk(chunk: bytes) -> Optional[str]:
                prepared_chunk = await self.prepare_audio_chunk(chunk)
                return await self.transcribe_chunk(prepared_chunk, model=model)
            
            transcriptions = await self.worker_pool.process_batch(chunks, process_chunk)
            
            # Handle errors
            errors = [t for t in transcriptions if isinstance(t, Exception)]
            if errors:
                error_msg = "; ".join(str(e) for e in errors)
                logger.error(f"Errors during parallel transcription: {error_msg}")
                return None
            
            # Combine successful transcriptions
            valid_transcriptions = [t for t in transcriptions if isinstance(t, str)]
            final_transcript = " ".join(valid_transcriptions)
            
            logger.info("Successfully transcribed all audio chunks in parallel")
            return final_transcript
            
        except ValueError as ve:
            logger.error(f"Audio conversion error: {str(ve)}")
            return None
        except Exception as e:
            logger.error(f"Google Transcription Error: {str(e)}", exc_info=True)
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
