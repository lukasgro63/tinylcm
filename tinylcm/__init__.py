"""
TinyLCM - Lightweight lifecycle management for ML models on edge devices.

This package provides tools for managing machine learning models on resource-constrained
edge devices, with support for model versioning, inference monitoring, drift detection,
and synchronization with central servers.
"""

# Version information
__version__ = "0.1.0"

# Import core components for easy access
from tinylcm.core.model_manager import ModelManager
from tinylcm.core.data_logger import DataLogger
from tinylcm.core.training_tracker import TrainingTracker
from tinylcm.core.inference_monitor import InferenceMonitor
from tinylcm.core.drift_detector import DriftDetector
from tinylcm.core.sync_interface import SyncInterface, SyncPackage

__all__ = [
    # Version info
    "__version__",
    
    # Core components
    "ModelManager",
    "DataLogger",
    "TrainingTracker",
    "InferenceMonitor",
    "DriftDetector",
    "SyncInterface",
    "SyncPackage"
]