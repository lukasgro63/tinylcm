"""
TinyLCM: Lightweight Lifecycle Management for TinyML on Edge Devices.

This package provides tools for managing the lifecycle of machine learning models
in resource-constrained edge environments, focusing on model versioning, 
monitoring, and data management.
"""

from importlib.metadata import version, PackageNotFoundError

# Core components
from tinylcm.core.model_manager import ModelManager
from tinylcm.core.inference_monitor import InferenceMonitor
from tinylcm.core.data_logger import DataLogger

# Utility modules
from tinylcm.utils.config import Config, get_config, set_global_config, load_config

# Constants
from tinylcm.constants import VERSION

try:
    __version__ = version("tinylcm")
except PackageNotFoundError:
    __version__ = VERSION

# Public API
__all__ = [
    # Core components
    "ModelManager",
    "InferenceMonitor",
    "DataLogger",
    
    # Configuration
    "Config",
    "get_config",
    "set_global_config",
    "load_config",
    
    # Version
    "__version__",
]