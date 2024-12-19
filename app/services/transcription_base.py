from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List, TypeVar, Callable, Awaitable
import asyncio
import logging
import tempfile
from fastapi import UploadFile
import io
import subprocess
from pydub import AudioSegment
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

class AsyncWorkerPool:
    """
    Generic async worker pool for parallel processing tasks
    """
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        
    async def process_item(self, 
                         item: T, 
                         process_func: Callable[[T], Awaitable[R]], 
                         item_id: int = None) -> R:
        """Process a single item with semaphore control"""
        try:
            async with self.semaphore:
                logger.info(f"Processing item {item_id} in worker pool")
                return await process_func(item)
        except Exception as e:
            logger.error(f"Error processing item {item_id}: {str(e)}", exc_info=True)
            raise
            
    async def process_batch(self, 
                          items: List[T], 
                          process_func: Callable[[T], Awaitable[R]]) -> List[R]:
        """Process a batch of items concurrently"""
        tasks = [
            self.process_item(item, process_func, idx) 
            for idx, item in enumerate(items)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

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
            logger.info(f"Reading from Path object: {audio_file}")
            with open(audio_file, 'rb') as f:
                content = f.read()
            original_format = audio_file.suffix[1:].lower()
        else:
            logger.info(f"Reading from UploadFile object: {audio_file.filename}")
            content = await audio_file.read()
            original_format = audio_file.filename.split('.')[-1].lower()
            await audio_file.seek(0)
        
        # Save content to temporary files
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{original_format}")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            temp_input.write(content)
            temp_input.close()
            temp_output.close()

            # Convert using ffmpeg
            cmd = [
                'ffmpeg',
                '-i', temp_input.name,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-y',
                temp_output.name
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                raise ValueError(f"Audio conversion failed: {process.stderr}")

            with open(temp_output.name, 'rb') as f:
                wav_data = f.read()

            logger.info(f"Successfully converted {original_format} to WAV format")
            return wav_data, original_format

        finally:
            for temp_file in [temp_input.name, temp_output.name]:
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to convert audio format: {str(e)}")

class TranscriptionService(ABC):
    """Base class for transcription services with parallel processing support"""
    
    def __init__(self, max_workers: int = 3):
        self.worker_pool = AsyncWorkerPool(max_workers=max_workers)
    
    @abstractmethod
    async def prepare_audio_chunk(self, chunk: bytes, **kwargs) -> bytes:
        """Prepare audio chunk for transcription (convert format if needed)"""
        pass
    
    @abstractmethod
    async def transcribe_chunk(self, chunk: bytes, **kwargs) -> Optional[str]:
        """Transcribe a single audio chunk"""
        pass
    
    async def split_audio(self, audio_data: bytes, chunk_size_mb: int = 10) -> List[bytes]:
        """Split audio data into chunks"""
        try:
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            chunk_duration = (chunk_size_mb * 60 * 1000) // 10
            
            chunks = []
            for i in range(0, len(audio_segment), chunk_duration):
                chunk = audio_segment[i:i + chunk_duration]
                chunk_buffer = io.BytesIO()
                chunk.export(chunk_buffer, format='wav')
                chunks.append(chunk_buffer.getvalue())
                
            logger.info(f"Split audio into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting audio: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to split audio: {str(e)}")
    
    async def transcribe_audio(self, 
                             audio_file: Union[UploadFile, Path], 
                             chunk_size_mb: int = 10,
                             **kwargs) -> Optional[str]:
        """Transcribe audio file with parallel chunk processing"""
        try:
            # Convert audio to WAV format
            wav_data, original_format = await convert_audio_to_wav(audio_file)
            
            # Split into chunks
            chunks = await self.split_audio(wav_data, chunk_size_mb)
            logger.info(f"Processing {len(chunks)} chunks in parallel")
            
            # Process chunks in parallel
            async def process_chunk(chunk: bytes) -> Optional[str]:
                prepared_chunk = await self.prepare_audio_chunk(chunk, **kwargs)
                return await self.transcribe_chunk(prepared_chunk, **kwargs)
            
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
            
            logger.info("Successfully transcribed all audio chunks")
            return final_transcript
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            return None
