from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class TranscriptionService(ABC):
    """Base class for transcription services"""
    
    @abstractmethod
    async def transcribe_audio(self, audio_file_path: Path) -> Optional[str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_file_path (Path): Path to the audio file
            
        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        pass
