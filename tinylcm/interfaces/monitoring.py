"""Abstract interfaces for monitoring components."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection."""
    
    @abstractmethod
    def check_for_anomalies(
        self,
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check if a record contains anomalies.
        
        Args:
            record: Record to check
            context: Additional context for the check
            
        Returns:
            Tuple of (is_anomaly, list_of_reasons)
        """
        pass


class MetricsProvider(ABC):
    """Abstract base class for metrics providers."""
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        pass


class MetricsConsumer(ABC):
    """Abstract base class for metrics consumers."""
    
    @abstractmethod
    def consume_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Process metrics data.
        
        Args:
            metrics: Metrics to process
        """
        pass


class DataStreamProcessor(ABC):
    """Abstract base class for data stream processors."""
    
    @abstractmethod
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single record.
        
        Args:
            record: Record to process
            
        Returns:
            Processed record
        """
        pass
    
    @abstractmethod
    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of records.
        
        Args:
            records: Records to process
            
        Returns:
            Processed records
        """
        pass

# tinylcm/interfaces/monitoring.py - Ergänzung
class DriftDetector(ABC):
    """Abstract base class for drift detection."""
    
    @abstractmethod
    def create_reference_distribution(self, data: Dict[str, Any]) -> None:
        """
        Create a reference distribution from data.
        
        Args:
            data: Dictionary with reference data
        """
        pass
    
    @abstractmethod
    def update(self, record: Dict[str, Any]) -> bool:
        """
        Update the detector with a new record.
        
        Args:
            record: New data record
            
        Returns:
            bool: True if drift was detected after this update
        """
        pass
    
    @abstractmethod
    def check_for_drift(self) -> Dict[str, Any]:
        """
        Check if drift has occurred between reference and current distributions.
        
        Returns:
            Dict[str, Any]: Drift check results
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the detector state."""
        pass