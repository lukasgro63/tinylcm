"""Model manager for TinyLCM.

Provides functionality for managing machine learning models, including
versioning, storing, retrieving, and tracking metadata.
"""

import json
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Protocol, TypeVar, cast

from tinylcm.constants import DEFAULT_ACTIVE_MODEL_LINK, DEFAULT_MODELS_DIR
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.file_utils import ensure_dir, load_json, save_json
from tinylcm.utils.versioning import calculate_file_hash, create_version_info


class ModelFormat(str, Enum):
    """Supported model formats."""
    
    TFLITE = "tflite"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    PICKLE = "pkl"
    JSON = "json"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, format_str: str) -> "ModelFormat":
        """Convert string to ModelFormat enum."""
        try:
            return cls(format_str.lower())
        except ValueError:
            return cls.UNKNOWN


class ModelStorageStrategy(ABC):
    """Abstract strategy for model storage."""
    
    @abstractmethod
    def save_model(
        self, 
        model_path: Union[str, Path], 
        model_id: str,
        target_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Save a model file to the target directory.
        
        Args:
            model_path: Path to the model file
            model_id: Unique identifier for the model
            target_dir: Target directory to save the model
            
        Returns:
            Dict[str, Any]: Model storage metadata
        """
        pass
    
    @abstractmethod
    def load_model(
        self, 
        model_id: str,
        models_dir: Union[str, Path]
    ) -> str:
        """
        Get the path to the model file.
        
        Args:
            model_id: Unique identifier for the model
            models_dir: Base directory for models
            
        Returns:
            str: Path to the model file
        """
        pass
    
    @abstractmethod
    def delete_model(
        self, 
        model_id: str,
        models_dir: Union[str, Path]
    ) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: Unique identifier for the model
            models_dir: Base directory for models
            
        Returns:
            bool: True if successful
        """
        pass


class FileSystemModelStorage(ModelStorageStrategy):
    """File system implementation of model storage."""
    
    def save_model(
        self, 
        model_path: Union[str, Path], 
        model_id: str,
        target_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Save a model file to the target directory.
        
        Args:
            model_path: Path to the model file
            model_id: Unique identifier for the model
            target_dir: Target directory to save the model
            
        Returns:
            Dict[str, Any]: Model storage metadata
        """
        source_path = Path(model_path)
        model_dir = Path(target_dir) / model_id
        
        # Create model directory if it doesn't exist
        ensure_dir(model_dir)
        
        # Copy model file to target directory
        target_path = model_dir / source_path.name
        shutil.copy2(source_path, target_path)
        
        # Calculate hash for integrity check
        md5_hash = calculate_file_hash(target_path)
        
        return {
            "filename": source_path.name,
            "path": str(target_path),
            "md5_hash": md5_hash
        }
    
    def load_model(
        self, 
        model_id: str,
        models_dir: Union[str, Path]
    ) -> str:
        """
        Get the path to the model file.
        
        Args:
            model_id: Unique identifier for the model
            models_dir: Base directory for models
            
        Returns:
            str: Path to the model file
        """
        model_dir = Path(models_dir) / model_id
        
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        
        # Find model file (assuming there's only one file in the directory)
        model_files = [f for f in model_dir.iterdir() if f.is_file()]
        
        if not model_files:
            raise ValueError(f"No model file found in directory: {model_dir}")
        
        return str(model_files[0])
    
    def delete_model(
        self, 
        model_id: str,
        models_dir: Union[str, Path]
    ) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: Unique identifier for the model
            models_dir: Base directory for models
            
        Returns:
            bool: True if successful
        """
        model_dir = Path(models_dir) / model_id
        
        if not model_dir.exists():
            return False
        
        shutil.rmtree(model_dir)
        return True


class ModelMetadataProvider(ABC):
    """Abstract provider for model metadata."""
    
    @abstractmethod
    def save_metadata(
        self, 
        model_id: str, 
        metadata: Dict[str, Any],
        metadata_dir: Union[str, Path]
    ) -> None:
        """
        Save metadata for a model.
        
        Args:
            model_id: Unique identifier for the model
            metadata: Model metadata
            metadata_dir: Directory to save metadata
        """
        pass
    
    @abstractmethod
    def load_metadata(
        self, 
        model_id: str,
        metadata_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load metadata for a model.
        
        Args:
            model_id: Unique identifier for the model
            metadata_dir: Directory containing metadata
            
        Returns:
            Dict[str, Any]: Model metadata
        """
        pass
    
    @abstractmethod
    def list_metadata(
        self,
        metadata_dir: Union[str, Path],
        filter_func: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        List metadata for all models.
        
        Args:
            metadata_dir: Directory containing metadata
            filter_func: Optional function to filter metadata
            
        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        pass
    
    @abstractmethod
    def delete_metadata(
        self, 
        model_id: str,
        metadata_dir: Union[str, Path]
    ) -> bool:
        """
        Delete metadata for a model.
        
        Args:
            model_id: Unique identifier for the model
            metadata_dir: Directory containing metadata
            
        Returns:
            bool: True if successful
        """
        pass


class JSONFileMetadataProvider(ModelMetadataProvider):
    """JSON file implementation of model metadata provider."""
    
    def save_metadata(
        self, 
        model_id: str, 
        metadata: Dict[str, Any],
        metadata_dir: Union[str, Path]
    ) -> None:
        """
        Save metadata for a model as JSON.
        
        Args:
            model_id: Unique identifier for the model
            metadata: Model metadata
            metadata_dir: Directory to save metadata
        """
        metadata_path = Path(metadata_dir) / f"{model_id}.json"
        save_json(metadata, metadata_path)
    
    def load_metadata(
        self, 
        model_id: str,
        metadata_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load metadata for a model from JSON.
        
        Args:
            model_id: Unique identifier for the model
            metadata_dir: Directory containing metadata
            
        Returns:
            Dict[str, Any]: Model metadata
        """
        metadata_path = Path(metadata_dir) / f"{model_id}.json"
        
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")
        
        return load_json(metadata_path)
    
    def list_metadata(
        self,
        metadata_dir: Union[str, Path],
        filter_func: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        List metadata for all models from JSON files.
        
        Args:
            metadata_dir: Directory containing metadata
            filter_func: Optional function to filter metadata
            
        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        metadata_list = []
        metadata_dir_path = Path(metadata_dir)
        
        if not metadata_dir_path.exists():
            return []
        
        for metadata_file in metadata_dir_path.glob("*.json"):
            try:
                metadata = load_json(metadata_file)
                
                # Apply filter if provided
                if filter_func is None or filter_func(metadata):
                    metadata_list.append(metadata)
            except Exception as e:
                # Log but continue
                print(f"Error loading metadata from {metadata_file}: {str(e)}")
        
        return metadata_list
    
    def delete_metadata(
        self, 
        model_id: str,
        metadata_dir: Union[str, Path]
    ) -> bool:
        """
        Delete metadata for a model.
        
        Args:
            model_id: Unique identifier for the model
            metadata_dir: Directory containing metadata
            
        Returns:
            bool: True if successful
        """
        metadata_path = Path(metadata_dir) / f"{model_id}.json"
        
        if not metadata_path.exists():
            return False
        
        metadata_path.unlink()
        return True


class ModelManager:
    """
    Model manager for TinyLCM.
    
    Manages model storage, metadata, and versioning.
    """
    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        storage_strategy: Optional[ModelStorageStrategy] = None,
        metadata_provider: Optional[ModelMetadataProvider] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the model manager.
        
        Args:
            storage_dir: Directory for model storage (default from config)
            storage_strategy: Strategy for model storage (default: FileSystemModelStorage)
            metadata_provider: Provider for model metadata (default: JSONFileMetadataProvider)
            config: Configuration (default: global config)
        """
        self.config = config or get_config()
        
        # Get storage directory from arguments or config
        self.storage_dir = Path(storage_dir or self.config.get("model_manager", "storage_dir", DEFAULT_MODELS_DIR))
        
        # Initialize storage directories
        self.models_dir = ensure_dir(self.storage_dir / "models")
        self.metadata_dir = ensure_dir(self.storage_dir / "metadata")
        
        # Set storage strategy and metadata provider
        self.storage_strategy = storage_strategy or FileSystemModelStorage()
        self.metadata_provider = metadata_provider or JSONFileMetadataProvider()
    
    def save_model(
        self,
        model_path: Union[str, Path],
        model_format: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        set_active: bool = False
    ) -> str:
        """
        Save a model to the storage.
        
        Args:
            model_path: Path to the model file
            model_format: Format of the model (e.g., "tflite", "onnx")
            version: Model version (default: auto-generated)
            description: Model description
            tags: List of tags for the model
            metrics: Performance metrics for the model
            set_active: Whether to set this model as active
            
        Returns:
            str: Unique identifier for the saved model
        """
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        
        # Save model file
        storage_meta = self.storage_strategy.save_model(
            model_path=model_path,
            model_id=model_id,
            target_dir=self.models_dir
        )
        
        # Prepare metadata
        metadata = {
            "model_id": model_id,
            "model_format": model_format,
            "version": version or f"v_{int(time.time())}",
            "description": description or "",
            "tags": tags or [],
            "metrics": metrics or {},
            "timestamp": time.time(),
            "is_active": set_active,
            "md5_hash": storage_meta["md5_hash"],
            "filename": storage_meta["filename"]
        }
        
        # Save metadata
        self.metadata_provider.save_metadata(
            model_id=model_id,
            metadata=metadata,
            metadata_dir=self.metadata_dir
        )
        
        # Set as active if requested
        if set_active:
            self.set_active_model(model_id)
        
        return model_id
    
    def load_model(self, model_id: Optional[str] = None) -> str:
        """
        Load a model from storage.
        
        Args:
            model_id: Unique identifier for the model (default: active model)
            
        Returns:
            str: Path to the model file
            
        Raises:
            ValueError: If model not found
        """
        # If no model_id provided, use active model
        if model_id is None:
            active_link = Path(self.storage_dir) / DEFAULT_ACTIVE_MODEL_LINK
            
            if not active_link.exists() or not active_link.is_symlink():
                raise ValueError("No active model set")
            
            model_id = os.path.basename(os.readlink(active_link))
        
        return self.storage_strategy.load_model(
            model_id=model_id,
            models_dir=self.models_dir
        )
    
    def get_model_metadata(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_id: Unique identifier for the model (default: active model)
            
        Returns:
            Dict[str, Any]: Model metadata
            
        Raises:
            ValueError: If model not found
        """
        # If no model_id provided, use active model
        if model_id is None:
            active_link = Path(self.storage_dir) / DEFAULT_ACTIVE_MODEL_LINK
            
            if not active_link.exists() or not active_link.is_symlink():
                raise ValueError("No active model set")
            
            model_id = os.path.basename(os.readlink(active_link))
        
        return self.metadata_provider.load_metadata(
            model_id=model_id,
            metadata_dir=self.metadata_dir
        )
    
    def list_models(
        self,
        tag: Optional[str] = None,
        model_format: Optional[str] = None,
        version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all models, optionally filtered.
        
        Args:
            tag: Filter by tag
            model_format: Filter by model format
            version: Filter by version
            
        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        def filter_func(metadata: Dict[str, Any]) -> bool:
            """Filter function for metadata."""
            if tag is not None and tag not in metadata.get("tags", []):
                return False
            
            if model_format is not None and metadata.get("model_format") != model_format:
                return False
            
            if version is not None and metadata.get("version") != version:
                return False
            
            return True
        
        return self.metadata_provider.list_metadata(
            metadata_dir=self.metadata_dir,
            filter_func=filter_func
        )
    
    def set_active_model(self, model_id: str) -> bool:
        """
        Set a model as the active model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If model not found
        """
        # Check if model exists
        model_dir = Path(self.models_dir) / model_id
        
        if not model_dir.exists():
            raise ValueError(f"Model not found: {model_id}")
        
        # Update metadata for all models
        all_models = self.list_models()
        
        for metadata in all_models:
            current_id = metadata["model_id"]
            metadata["is_active"] = (current_id == model_id)
            
            # Save updated metadata
            self.metadata_provider.save_metadata(
                model_id=current_id,
                metadata=metadata,
                metadata_dir=self.metadata_dir
            )
        
        # Create/update symbolic link
        active_link = Path(self.storage_dir) / DEFAULT_ACTIVE_MODEL_LINK
        
        # Remove existing link if any
        if active_link.exists():
            if active_link.is_symlink():
                active_link.unlink()
            else:
                raise ValueError(f"Active model path exists but is not a symlink: {active_link}")
        
        # Create new link
        os.symlink(model_dir, active_link, target_is_directory=True)
        
        return True
    
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: Unique identifier for the model
            force: If True, delete even if it's the active model
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If trying to delete active model without force
        """
        # Check if it's the active model
        try:
            metadata = self.get_model_metadata(model_id)
            is_active = metadata.get("is_active", False)
            
            if is_active and not force:
                raise ValueError(
                    f"Cannot delete active model (model_id={model_id}). "
                    "Set force=True to override."
                )
            
            # Delete model files
            deleted_files = self.storage_strategy.delete_model(
                model_id=model_id,
                models_dir=self.models_dir
            )
            
            # Delete metadata
            deleted_metadata = self.metadata_provider.delete_metadata(
                model_id=model_id,
                metadata_dir=self.metadata_dir
            )
            
            # Remove active model link if needed
            if is_active:
                active_link = Path(self.storage_dir) / DEFAULT_ACTIVE_MODEL_LINK
                if active_link.exists() and active_link.is_symlink():
                    active_link.unlink()
            
            return deleted_files and deleted_metadata
            
        except Exception as e:
            if "No active model set" in str(e):
                # Not the active model, continue with deletion
                pass
            else:
                # Re-raise other exceptions
                raise
        
        # Delete model files
        deleted_files = self.storage_strategy.delete_model(
            model_id=model_id,
            models_dir=self.models_dir
        )
        
        # Delete metadata
        deleted_metadata = self.metadata_provider.delete_metadata(
            model_id=model_id,
            metadata_dir=self.metadata_dir
        )
        
        return deleted_files and deleted_metadata
    
    def add_tag(self, model_id: str, tag: str) -> bool:
        """
        Add a tag to a model.
        
        Args:
            model_id: Unique identifier for the model
            tag: Tag to add
            
        Returns:
            bool: True if successful
        """
        try:
            metadata = self.get_model_metadata(model_id)
            
            if "tags" not in metadata:
                metadata["tags"] = []
            
            if tag not in metadata["tags"]:
                metadata["tags"].append(tag)
                
                # Save updated metadata
                self.metadata_provider.save_metadata(
                    model_id=model_id,
                    metadata=metadata,
                    metadata_dir=self.metadata_dir
                )
            
            return True
            
        except Exception as e:
            print(f"Error adding tag to model {model_id}: {str(e)}")
            return False
    
    def remove_tag(self, model_id: str, tag: str) -> bool:
        """
        Remove a tag from a model.
        
        Args:
            model_id: Unique identifier for the model
            tag: Tag to remove
            
        Returns:
            bool: True if successful
        """
        try:
            metadata = self.get_model_metadata(model_id)
            
            if "tags" in metadata and tag in metadata["tags"]:
                metadata["tags"].remove(tag)
                
                # Save updated metadata
                self.metadata_provider.save_metadata(
                    model_id=model_id,
                    metadata=metadata,
                    metadata_dir=self.metadata_dir
                )
            
            return True
            
        except Exception as e:
            print(f"Error removing tag from model {model_id}: {str(e)}")
            return False
    
    def update_metrics(self, model_id: str, metrics: Dict[str, float]) -> bool:
        """
        Update metrics for a model.
        
        Args:
            model_id: Unique identifier for the model
            metrics: Dictionary of metrics to update
            
        Returns:
            bool: True if successful
        """
        try:
            metadata = self.get_model_metadata(model_id)
            
            if "metrics" not in metadata:
                metadata["metrics"] = {}
            
            # Update metrics
            metadata["metrics"].update(metrics)
            
            # Save updated metadata
            self.metadata_provider.save_metadata(
                model_id=model_id,
                metadata=metadata,
                metadata_dir=self.metadata_dir
            )
            
            return True
            
        except Exception as e:
            print(f"Error updating metrics for model {model_id}: {str(e)}")
            return False
    
    def verify_model_integrity(self, model_id: str) -> bool:
        """
        Verify the integrity of a model by comparing its hash.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            bool: True if the model is valid
        """
        try:
            # Get metadata with stored hash
            metadata = self.get_model_metadata(model_id)
            stored_hash = metadata.get("md5_hash")
            
            if not stored_hash:
                return False
            
            # Get model file path
            model_path = self.load_model(model_id)
            
            # Calculate current hash
            current_hash = calculate_file_hash(model_path)
            
            # Compare hashes
            return current_hash == stored_hash
            
        except Exception as e:
            print(f"Error verifying model integrity for {model_id}: {str(e)}")
            return False