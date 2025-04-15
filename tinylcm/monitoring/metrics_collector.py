"""Collection and calculation of metrics for monitoring."""

from collections import Counter, deque
from typing import Any, Dict, List, Optional

import numpy as np

from tinylcm.interfaces.monitoring import MetricsProvider
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.metrics import MetricsCalculator

logger = setup_logger(__name__)

class InferenceMetricsCollector(MetricsProvider):
    """Collects and calculates metrics for inference data."""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize the metrics collector.
        
        Args:
            window_size: Size of the sliding window for metrics
        """
        self.window_size = window_size
        self.latency_window = deque(maxlen=window_size)
        self.confidence_window = deque(maxlen=window_size)
        self.prediction_counts = Counter()
        self.total_inferences = 0
        self.ground_truth_correct = 0
        self.ground_truth_total = 0
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
    
    def add_record(self, record: Dict[str, Any]) -> None:
        """
        Add an inference record and update metrics.
        
        Args:
            record: Inference record
        """
        self.total_inferences += 1
        
        # Confidence
        if "confidence" in record and record["confidence"] is not None:
            self.confidence_window.append(record["confidence"])
            
        # Latency
        if "latency_ms" in record and record["latency_ms"] is not None:
            self.latency_window.append(record["latency_ms"])
            
        # Prediction
        if "prediction" in record:
            self.prediction_counts[record["prediction"]] += 1
            
        # Ground Truth (if available)
        if "ground_truth" in record and record["ground_truth"] is not None:
            self.ground_truth_total += 1
            if record["prediction"] == record["ground_truth"]:
                self.ground_truth_correct += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            "total_inferences": self.total_inferences,
            "prediction_distribution": dict(self.prediction_counts)
        }
        
        # Calculate latency statistics
        if self.latency_window:
            metrics["latency"] = MetricsCalculator.calculate_latency_stats(list(self.latency_window))
            
        # Calculate confidence statistics
        if self.confidence_window:
            metrics["confidence"] = MetricsCalculator.calculate_confidence_stats(list(self.confidence_window))
            
        # Calculate accuracy if ground truth available
        if self.ground_truth_total > 0:
            metrics["accuracy"] = self.ground_truth_correct / self.ground_truth_total
            
        return metrics
    
    def get_statistical_context(self) -> Dict[str, Any]:
        """
        Create statistical context for anomaly detection.
        
        Returns:
            Statistical context with distributions and statistics
        """
        context = {
            "prediction_distribution": dict(self.prediction_counts)
        }
        
        # Confidence statistics
        if self.confidence_window:
            confidence_array = np.array(list(self.confidence_window))
            context["confidence_stats"] = {
                "mean": np.mean(confidence_array),
                "std": np.std(confidence_array),
                "min": np.min(confidence_array),
                "max": np.max(confidence_array)
            }
            
        # Latency statistics
        if self.latency_window:
            latency_array = np.array(list(self.latency_window))
            context["latency_stats"] = {
                "mean": np.mean(latency_array),
                "std": np.std(latency_array),
                "min": np.min(latency_array),
                "max": np.max(latency_array)
            }
            
        return context
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.latency_window.clear()
        self.confidence_window.clear()
        self.prediction_counts = Counter()
        self.total_inferences = 0
        self.ground_truth_correct = 0
        self.ground_truth_total = 0
        self.logger.info("Metrics collector reset")