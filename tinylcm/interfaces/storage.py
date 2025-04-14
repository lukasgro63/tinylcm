"""Abstract interfaces for storage mechanisms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, TypeVar, Generic

T = TypeVar('T')

class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save(self, data: Any, path: str) -> str:
        """
        Save data at the specified path.
        
        Args:
            data: Data to save
            path: Storage path
            
        Returns:
            Actual path where the data was saved
            
        Raises:
            StorageError: If there are issues with saving
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> Any:
        """
        Load data from the specified path.
        
        Args:
            path: Path of the data to load
            
        Returns:
            Loaded data
            
        Raises:
            StorageError: If there are issues with loading
        """
        pass
    
    @abstractmethod
    def delete(self, path: str) -> bool:
        """
        Delete data from the specified path.
        
        Args:
            path: Path of the data to delete
            
        Returns:
            True if successfully deleted
            
        Raises:
            StorageError: If there are issues with deletion
        """
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Check if data exists at the specified path.
        
        Args:
            path: Path to check
            
        Returns:
            True if data exists
        """
        pass


class ModelStorageBackend(StorageBackend):
    """Abstract base class for model storage backends."""
    
    @abstractmethod
    def get_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model metadata
            
        Raises:
            ModelNotFoundError: If the model was not found
        """
        pass
    
    @abstractmethod
    def list_models(self, **filters) -> List[Dict[str, Any]]:
        """
        List available models, optionally filtered.
        
        Args:
            **filters: Filter criteria
            
        Returns:
            List of model metadata
        """
        pass


class StreamingStorageBackend(StorageBackend):
    """Abstract base class for streaming storage backends."""
    
    @abstractmethod
    def stream_read(self, path: str, chunk_size: int = 1024) -> Generator[Any, None, None]:
        """
        Read data in chunks from the specified path.
        
        Args:
            path: Path of the data to read
            chunk_size: Size of chunks to read
            
        Yields:
            Data chunks
            
        Raises:
            StorageError: If there are issues with reading
        """
        pass
    
    @abstractmethod
    def stream_write(self, data_generator: Any, path: str) -> str:
        """
        Write data from a generator to the specified path.
        
        Args:
            data_generator: Generator providing data
            path: Storage path
            
        Returns:
            Actual path where the data was saved
            
        Raises:
            StorageError: If there are issues with writing
        """
        pass