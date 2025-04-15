"""Custom exceptions for TinyLCM.

This module defines all custom exception types used throughout the library,
providing more specific error handling and clearer error messages.
"""

class TinyLCMError(Exception):
    """Base class for all TinyLCM-specific exceptions."""
    pass


class ModelError(TinyLCMError):
    """Errors related to models."""
    pass


class ModelNotFoundError(ModelError):
    """Model was not found."""
    pass


class ModelIntegrityError(ModelError):
    """Model integrity is violated."""
    pass


class StorageError(TinyLCMError):
    """Errors related to storage."""
    pass


class StorageAccessError(StorageError):
    """Error accessing storage."""
    pass


class StorageWriteError(StorageError):
    """Error writing to storage."""
    pass


class ConfigError(TinyLCMError):
    """Errors related to configuration."""
    pass


class DataLoggerError(TinyLCMError):
    """Errors related to the data logger."""
    pass


class MonitoringError(TinyLCMError):
    """Errors related to monitoring."""
    pass


class InvalidInputError(TinyLCMError):
    """Invalid input data."""
    pass

class SyncError(TinyLCMError):
    """Errors related to synchronization."""
    pass