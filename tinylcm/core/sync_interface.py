"""Synchronization interface for TinyLCM.

Provides functionality for preparing data for transmission between
edge devices and central servers, with support for packaging,
compression, and serialization.
"""

import gzip
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tinylcm.constants import DEFAULT_SYNC_DIR
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.errors import SyncError
from tinylcm.utils.file_utils import ensure_dir, load_json, save_json
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.versioning import calculate_file_hash


class SyncPackage:
    """
    Represents a package of data being prepared for synchronization.
    
    A SyncPackage contains files, metadata, and configuration for
    transmission between edge devices and servers.
    """
    
    def __init__(
        self,
        package_id: str,
        device_id: str,
        package_type: str,
        work_dir: Union[str, Path],
        compression: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Initialize a sync package.
        
        Args:
            package_id: Unique identifier for the package
            device_id: Identifier of the device creating the package
            package_type: Type of package (e.g., "models", "metrics", "logs")
            work_dir: Working directory for package assembly
            compression: Compression method ("gzip", "zip", or None)
            description: Optional description
        """
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        
        self.package_id = package_id
        self.device_id = device_id
        self.package_type = package_type
        self.description = description
        self.work_dir = Path(work_dir)
        self.compression = compression
        
        # Initialize package state
        self.creation_time = time.time()
        self.files = []
        self.is_finalized = False
        
        # Create working directory
        ensure_dir(self.work_dir)
        self.logger.debug(f"Created sync package: {package_id}")
    
    def add_file(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a file to the package.
        
        Args:
            file_path: Path to the file
            file_type: Type of file (e.g., "model", "metrics", "logs")
            metadata: Additional metadata for the file
            
        Raises:
            FileNotFoundError: If file does not exist
            SyncError: If package is already finalized
        """
        if self.is_finalized:
            raise SyncError(f"Cannot add file to finalized package: {self.package_id}")
        
        # Validate file exists
        src_path = Path(file_path)
        if not src_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create entry in package
        rel_path = src_path.name
        dst_path = self.work_dir / rel_path
        
        # Copy file to package directory
        shutil.copy2(src_path, dst_path)
        
        # Add file entry
        file_entry = {
            "original_path": str(src_path),
            "package_path": rel_path,
            "file_type": file_type or "unknown",
            "size_bytes": os.path.getsize(dst_path),
            "timestamp": time.time(),
            "hash": calculate_file_hash(dst_path),
            "metadata": metadata or {}
        }
        
        self.files.append(file_entry)
        self.logger.debug(f"Added file to package {self.package_id}: {rel_path}")
    
    def finalize(self, output_path: Union[str, Path]) -> str:
        """
        Finalize the package for transmission.
        
        Args:
            output_path: Path where to save the finalized package
            
        Returns:
            str: Path to the finalized package file
            
        Raises:
            SyncError: If package is already finalized or no files added
        """
        if self.is_finalized:
            raise SyncError(f"Package already finalized: {self.package_id}")
        
        if not self.files:
            raise SyncError(f"Cannot finalize empty package: {self.package_id}")
        
        # Create output directory if needed
        out_path = Path(output_path)
        ensure_dir(out_path.parent)
        
        # Determine compression method
        if self.compression == "gzip":
            self._finalize_gzip(out_path)
        elif self.compression == "zip":
            self._finalize_zip(out_path)
        else:
            # No compression, just create a tar archive
            self._finalize_tar(out_path)
        
        self.is_finalized = True
        self.logger.info(f"Finalized package {self.package_id} to {out_path}")
        
        return str(out_path)
    
    def _finalize_gzip(self, output_path: Path) -> None:
        """
        Finalize package using gzip compression.
        
        Args:
            output_path: Path where to save the package
        """
        import tarfile
        
        # Create a tar.gz archive
        with tarfile.open(output_path, "w:gz") as tar:
            for file_entry in self.files:
                file_path = self.work_dir / file_entry["package_path"]
                arcname = file_entry["package_path"]
                tar.add(file_path, arcname=arcname)
    
    def _finalize_zip(self, output_path: Path) -> None:
        """
        Finalize package using zip compression.
        
        Args:
            output_path: Path where to save the package
        """
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_entry in self.files:
                file_path = self.work_dir / file_entry["package_path"]
                zipf.write(file_path, arcname=file_entry["package_path"])
    
    def _finalize_tar(self, output_path: Path) -> None:
        """
        Finalize package using uncompressed tar.
        
        Args:
            output_path: Path where to save the package
        """
        import tarfile
        
        # Create an uncompressed tar archive
        with tarfile.open(output_path, "w") as tar:
            for file_entry in self.files:
                file_path = self.work_dir / file_entry["package_path"]
                arcname = file_entry["package_path"]
                tar.add(file_path, arcname=arcname)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get package metadata.
        
        Returns:
            Dict[str, Any]: Package metadata
        """
        return {
            "package_id": self.package_id,
            "device_id": self.device_id,
            "package_type": self.package_type,
            "description": self.description,
            "creation_time": self.creation_time,
            "files": self.files,
            "compression": self.compression,
            "is_finalized": self.is_finalized
        }


class SyncInterface:
    """
    Interface for synchronizing data between edge devices and servers.
    
    Provides functionality for creating, managing, and tracking
    packages of data for synchronization.
    """
    
    def __init__(
        self,
        sync_dir: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the sync interface.
        
        Args:
            sync_dir: Directory for sync-related files
            config: Configuration object
        """
        self.config = config or get_config()
        component_config = self.config.get_component_config("sync")
        
        # Set up logger
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize directory structure
        self.sync_dir = Path(sync_dir or component_config.get("storage_dir", DEFAULT_SYNC_DIR))
        self.packages_dir = ensure_dir(self.sync_dir / "packages")
        self.history_dir = ensure_dir(self.sync_dir / "history")
        
        # Track active packages
        self.active_packages = {}
        
        self.logger.info(f"Initialized sync interface with storage at: {self.sync_dir}")
    
    def create_package(
        self,
        device_id: str,
        package_type: str,
        description: Optional[str] = None,
        compression: Optional[str] = None
    ) -> str:
        """
        Create a new sync package.
        
        Args:
            device_id: Identifier of the device creating the package
            package_type: Type of package (e.g., "models", "metrics", "logs")
            description: Optional description
            compression: Compression method ("gzip", "zip", or None)
            
        Returns:
            str: Package ID
        """
        # Generate package ID
        package_id = str(uuid.uuid4())
        
        # Create temporary directory for package assembly
        package_dir = ensure_dir(self.packages_dir / f"tmp_{package_id}")
        
        # Create package object
        package = SyncPackage(
            package_id=package_id,
            device_id=device_id,
            package_type=package_type,
            work_dir=package_dir,
            compression=compression,
            description=description
        )
        
        # Track active package
        self.active_packages[package_id] = package
        
        self.logger.debug(f"Created package: {package_id}")
        return package_id
    
    def add_file_to_package(
        self,
        package_id: str,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a file to a package.
        
        Args:
            package_id: Package identifier
            file_path: Path to the file
            file_type: Type of file (e.g., "model", "metrics", "logs")
            metadata: Additional metadata for the file
            
        Raises:
            SyncError: If package does not exist
            FileNotFoundError: If file does not exist
        """
        # Get package
        package = self._get_package(package_id)
        
        # Add file to package
        package.add_file(file_path, file_type, metadata)
    
    def add_directory_to_package(
        self,
        package_id: str,
        directory_path: Union[str, Path],
        recursive: bool = False,
        file_type: Optional[str] = None
    ) -> int:
        """
        Add all files in a directory to a package.
        
        Args:
            package_id: Package identifier
            directory_path: Path to the directory
            recursive: Whether to include files in subdirectories
            file_type: Type of file for all files in the directory
            
        Returns:
            int: Number of files added
            
        Raises:
            SyncError: If package does not exist
            FileNotFoundError: If directory does not exist
        """
        # Get package
        package = self._get_package(package_id)
        
        # Validate directory exists
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Add files
        count = 0
        
        if recursive:
            # Use walk to get all files recursively
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        package.add_file(file_path, file_type)
                        count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to add file {file_path}: {e}")
        else:
            # Just get files in the top directory
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    try:
                        package.add_file(file_path, file_type)
                        count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to add file {file_path}: {e}")
        
        self.logger.debug(f"Added {count} files from directory {directory_path} to package {package_id}")
        return count
    
    def create_package_from_components(
        self,
        device_id: str,
        model_manager=None,
        inference_monitor=None,
        data_logger=None,
        training_tracker=None,
        compression: Optional[str] = None
    ) -> str:
        """
        Create a package from TinyLCM components.
        
        Args:
            device_id: Identifier of the device creating the package
            model_manager: ModelManager instance (optional)
            inference_monitor: InferenceMonitor instance (optional)
            data_logger: DataLogger instance (optional)
            training_tracker: TrainingTracker instance (optional)
            compression: Compression method ("gzip", "zip", or None)
            
        Returns:
            str: Package ID
            
        Raises:
            SyncError: If no components provided
        """
        if not any([model_manager, inference_monitor, data_logger, training_tracker]):
            raise SyncError("At least one component must be provided")
        
        # Create package
        package_id = self.create_package(
            device_id=device_id,
            package_type="components",
            description="Package with component data",
            compression=compression
        )
        
        # Add model data if model_manager provided
        if model_manager:
            try:
                # Load active model (uses the active model if no model_id is provided)
                model_path = model_manager.load_model()
                
                # Get model metadata for the active model
                # Since the active model was just loaded, we can get it without a specific ID
                model_meta = model_manager.get_model_metadata()
                
                self.add_file_to_package(
                    package_id=package_id,
                    file_path=model_path,
                    file_type="model",
                    metadata=model_meta
                )
                
                self.logger.debug(f"Added model from ModelManager to package {package_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add model data to package: {e}")
        
        # Add metrics data if inference_monitor provided
        if inference_monitor:
            try:
                # Export metrics to file
                metrics_path = inference_monitor.export_metrics(format="json")
                
                self.add_file_to_package(
                    package_id=package_id,
                    file_path=metrics_path,
                    file_type="metrics"
                )
                
                self.logger.debug(f"Added metrics from InferenceMonitor to package {package_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add metrics data to package: {e}")
        
        # Add data logs if data_logger provided
        if data_logger:
            try:
                # Export data log to CSV
                log_path = data_logger.export_to_csv()
                
                self.add_file_to_package(
                    package_id=package_id,
                    file_path=log_path,
                    file_type="data_log"
                )
                
                self.logger.debug(f"Added data log from DataLogger to package {package_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add data log to package: {e}")
        
        # Add training data if training_tracker provided
        if training_tracker:
            try:
                # Get recent runs
                runs = training_tracker.list_runs()
                
                if runs:
                    # Create a temporary directory to export run data
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Export most recent run to MLflow format
                        most_recent_run = runs[0]  # Assuming list_runs returns sorted by recency
                        
                        export_dir = os.path.join(temp_dir, "runs")
                        training_tracker.export_to_mlflow_format(
                            most_recent_run["run_id"],
                            export_dir
                        )
                        
                        # Add the exported run directory to the package
                        self.add_directory_to_package(
                            package_id=package_id,
                            directory_path=export_dir,
                            recursive=True,
                            file_type="training_run"
                        )
                        
                        self.logger.debug(f"Added training run from TrainingTracker to package {package_id}")
            except Exception as e:
                self.logger.warning(f"Failed to add training data to package: {e}")
        
        return package_id
    
    def finalize_package(self, package_id: str) -> str:
        """
        Finalize a package for transmission.
        
        Args:
            package_id: Package identifier
            
        Returns:
            str: Path to the finalized package file
            
        Raises:
            SyncError: If package does not exist or is already finalized
        """
        # Get package
        package = self._get_package(package_id)
        
        # Create output path
        timestamp = int(time.time())
        output_file = f"{package_id}_{timestamp}"
        
        if package.compression == "gzip":
            output_file += ".tar.gz"
        elif package.compression == "zip":
            output_file += ".zip"
        else:
            output_file += ".tar"
            
        output_path = self.packages_dir / output_file
        
        # Finalize package
        package_path = package.finalize(output_path)
        
        # Save package metadata
        metadata_path = self.packages_dir / f"{package_id}.meta.json"
        save_json(package.get_metadata(), metadata_path)
        
        # Remove package from active packages
        del self.active_packages[package_id]
        
        # Clean up temporary directory
        package_dir = self.packages_dir / f"tmp_{package_id}"
        if package_dir.exists():
            shutil.rmtree(package_dir)
        
        self.logger.info(f"Finalized package {package_id} to {package_path}")
        return package_path
    
    def list_packages(
        self,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        include_synced: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List available packages.
        
        Args:
            filter_func: Optional function to filter packages
            include_synced: Whether to include packages marked as synced
            
        Returns:
            List[Dict[str, Any]]: List of package metadata
        """
        packages = []
        
        # Get metadata files
        metadata_files = list(self.packages_dir.glob("*.meta.json"))
        
        for meta_file in metadata_files:
            try:
                # Load metadata
                metadata = load_json(meta_file)
                
                # Check if package is synced
                package_id = metadata.get("package_id")
                if not package_id:
                    continue
                    
                # Check if synced and should be included
                synced_meta_path = self.history_dir / f"{package_id}.sync.json"
                if synced_meta_path.exists():
                    # Load sync metadata
                    sync_meta = load_json(synced_meta_path)
                    
                    # Add sync info to metadata
                    metadata["sync_status"] = sync_meta.get("status")
                    metadata["sync_time"] = sync_meta.get("sync_time")
                    metadata["server_id"] = sync_meta.get("server_id")
                    
                    # Skip if synced and not including synced
                    if not include_synced:
                        continue
                
                # Apply filter if provided
                if filter_func and not filter_func(metadata):
                    continue
                    
                packages.append(metadata)
                
            except Exception as e:
                self.logger.warning(f"Failed to load package metadata from {meta_file}: {e}")
        
        return packages
    
    def mark_as_synced(
        self,
        package_id: str,
        sync_time: float,
        server_id: str,
        status: str
    ) -> None:
        """
        Mark a package as synchronized with the server.
        
        Args:
            package_id: Package identifier
            sync_time: Time of synchronization
            server_id: Identifier of the server
            status: Sync status (e.g., "success", "error")
            
        Raises:
            SyncError: If package does not exist
        """
        # Check if package metadata exists
        meta_path = self.packages_dir / f"{package_id}.meta.json"
        if not meta_path.exists():
            raise SyncError(f"Package metadata not found: {package_id}")
        
        # Create sync metadata
        sync_meta = {
            "package_id": package_id,
            "sync_time": sync_time,
            "server_id": server_id,
            "status": status
        }
        
        # Save sync metadata
        sync_meta_path = self.history_dir / f"{package_id}.sync.json"
        save_json(sync_meta, sync_meta_path)
        
        self.logger.info(f"Marked package {package_id} as synced with server {server_id}")
    
    def _get_package(self, package_id: str) -> SyncPackage:
        """
        Get an active package by ID.
        
        Args:
            package_id: Package identifier
            
        Returns:
            SyncPackage: Package object
            
        Raises:
            SyncError: If package does not exist
        """
        if package_id not in self.active_packages:
            raise SyncError(f"Package not found or not active: {package_id}")
            
        return self.active_packages[package_id]