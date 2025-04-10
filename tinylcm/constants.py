"""
Constants for the TinyLCM library.

This module defines all constant values used throughout the library,
including version information, data types, file formats, and default
configuration values.
"""

from enum import Enum
from typing import Dict, Any, Final

# Version
VERSION: Final[str] = "0.1.0"

# Model formats
class ModelFormat(str, Enum):
    """Supported model file formats."""
    TFLITE = "tflite"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    PICKLE = "pkl"
    JSON = "json"
    UNKNOWN = "unknown"

# Export as constants for backwards compatibility
MODEL_FORMAT_TFLITE: Final[str] = ModelFormat.TFLITE.value
MODEL_FORMAT_ONNX: Final[str] = ModelFormat.ONNX.value
MODEL_FORMAT_PYTORCH: Final[str] = ModelFormat.PYTORCH.value
MODEL_FORMAT_PICKLE: Final[str] = ModelFormat.PICKLE.value

# Status values
class Status(str, Enum):
    """Process status indicators."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    PENDING = "PENDING"
    UNKNOWN = "UNKNOWN"

# Export as constants for backwards compatibility
STATUS_RUNNING: Final[str] = Status.RUNNING.value
STATUS_COMPLETED: Final[str] = Status.COMPLETED.value
STATUS_FAILED: Final[str] = Status.FAILED.value
STATUS_ABORTED: Final[str] = Status.ABORTED.value

# Data types
class DataType(str, Enum):
    """Supported input data types."""
    IMAGE = "image"
    TEXT = "text"
    SENSOR = "sensor"
    JSON = "json"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"

# Export as constants for backwards compatibility
DATA_TYPE_IMAGE: Final[str] = DataType.IMAGE.value
DATA_TYPE_TEXT: Final[str] = DataType.TEXT.value
DATA_TYPE_SENSOR: Final[str] = DataType.SENSOR.value
DATA_TYPE_JSON: Final[str] = DataType.JSON.value

# File formats
class FileFormat(str, Enum):
    """Supported file formats."""
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    YAML = "yaml"
    HDF5 = "h5"
    PICKLE = "pkl"

# Export as constants for backwards compatibility
FILE_FORMAT_JSON: Final[str] = FileFormat.JSON.value
FILE_FORMAT_CSV: Final[str] = FileFormat.CSV.value
FILE_FORMAT_TXT: Final[str] = FileFormat.TXT.value
FILE_FORMAT_PNG: Final[str] = FileFormat.PNG.value
FILE_FORMAT_JPG: Final[str] = FileFormat.JPG.value
FILE_FORMAT_JPEG: Final[str] = FileFormat.JPEG.value

# Default file paths
DEFAULT_BASE_DIR: Final[str] = "tinylcm_data"
DEFAULT_MODELS_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/models"
DEFAULT_TRAINING_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/training_runs"
DEFAULT_INFERENCE_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/inference_logs"
DEFAULT_DRIFT_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/drift_detector"
DEFAULT_DATA_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/data_logs"
DEFAULT_SYNC_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/sync"
DEFAULT_LOG_DIR: Final[str] = f"{DEFAULT_BASE_DIR}/logs"

# Default file names
DEFAULT_LOG_FILE: Final[str] = f"{DEFAULT_LOG_DIR}/tinylcm.log"
DEFAULT_CONFIG_FILE: Final[str] = "tinylcm_config.json"
DEFAULT_ACTIVE_MODEL_LINK: Final[str] = "active_model"

# Default buffer sizes and limits
DEFAULT_MAX_STORAGE: Final[int] = 1024 * 1024 * 1024  # 1GB
DEFAULT_MEMORY_ENTRIES: Final[int] = 1000
DEFAULT_LOG_INTERVAL: Final[int] = 100
DEFAULT_BUFFER_SIZE: Final[int] = 50

# Default configuration settings
DEFAULT_CONFIG: Final[Dict[str, Dict[str, Any]]] = {
    "storage": {
        "base_dir": DEFAULT_BASE_DIR,
        "max_storage_bytes": DEFAULT_MAX_STORAGE,
        "cleanup_threshold": 0.9  # Cleanup when storage is 90% full
    },
    "model_manager": {
        "storage_dir": DEFAULT_MODELS_DIR,
        "max_models": 10,
        "enable_integrity_check": True
    },
    "inference_monitor": {
        "storage_dir": DEFAULT_INFERENCE_DIR,
        "log_interval": DEFAULT_LOG_INTERVAL,
        "memory_window_size": DEFAULT_MEMORY_ENTRIES,
        "confidence_threshold": 0.3,  # Alert on confidence below this
        "latency_threshold_ms": 100  # Alert on latency above this (ms)
    },
    "data_logger": {
        "storage_dir": DEFAULT_DATA_DIR,
        "buffer_size": DEFAULT_BUFFER_SIZE,
        "max_entries": 10000,
        "image_format": "jpg"
    },
    "drift_detector": {
        "storage_dir": DEFAULT_DRIFT_DIR,
        "window_size": 100,
        "alert_threshold": 0.2,
        "enabled": True
    },
    "sync": {
        "storage_dir": DEFAULT_SYNC_DIR,
        "auto_sync": False,
        "sync_interval_seconds": 3600,
        "max_retry": 3
    },
    "training_tracker": {
        "storage_dir": DEFAULT_TRAINING_DIR,
        "log_artifacts": True
    },
    "logging": {
        "log_dir": DEFAULT_LOG_DIR,
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
