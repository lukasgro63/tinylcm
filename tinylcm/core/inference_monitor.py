"""Monitoring of inference metrics and anomaly detection."""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tinylcm.constants import DEFAULT_INFERENCE_DIR, DEFAULT_LOG_INTERVAL, DEFAULT_MEMORY_ENTRIES
from tinylcm.core.anomaly_detectors import (
    CompositeAnomalyDetector,
    StatisticalAnomalyDetector,
    ThresholdAnomalyDetector,
    anomaly_detector_registry
)
from tinylcm.core.metrics_collector import InferenceMetricsCollector
from tinylcm.interfaces.monitoring import AnomalyDetector, MetricsProvider
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.errors import MonitoringError
from tinylcm.utils.file_utils import ensure_dir, save_json, stream_write_jsonl
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)

class InferenceMonitor:
    """
    Monitor and analyze model inference performance.
    
    Tracks metrics like latency, confidence, and prediction distribution
    to identify performance issues or drift.
    """

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        memory_window_size: Optional[int] = None,
        log_interval: Optional[int] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the inference monitor.
        
        Args:
            storage_dir: Directory for storing logs and metrics
            memory_window_size: Number of recent inferences to keep in memory
            log_interval: Number of inferences before auto-saving logs
            anomaly_detector: Custom anomaly detector
            config: Configuration object
        """
        self.config = config or get_config()
        component_config = self.config.get_component_config("inference_monitor")
        
        # Set up logger
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

        # Set configuration values
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir", DEFAULT_INFERENCE_DIR))
        self.memory_window_size = memory_window_size or component_config.get("memory_window_size", DEFAULT_MEMORY_ENTRIES)
        self.log_interval = log_interval or component_config.get("log_interval", DEFAULT_LOG_INTERVAL)

        # Create storage directory
        ensure_dir(self.storage_dir)

        # Initialize state
        self.session_id = str(uuid.uuid4())
        self.logger.info(f"Starting inference monitor session: {self.session_id}")
        
        # Initialize metrics collector
        self.metrics_collector = InferenceMetricsCollector(window_size=self.memory_window_size)
        
        # Anomaly log file
        self._current_log_file = None
        self._record_count = 0

        # Initialize anomaly detector
        if anomaly_detector is None:
            # Create default composite detector
            threshold_detector = ThresholdAnomalyDetector(
                confidence_threshold=component_config.get("confidence_threshold", 0.3),
                latency_threshold_ms=component_config.get("latency_threshold_ms", 100.0)
            )
            statistical_detector = StatisticalAnomalyDetector()

            self.anomaly_detector = CompositeAnomalyDetector([
                threshold_detector,
                statistical_detector
            ])
        else:
            self.anomaly_detector = anomaly_detector

        # Initialize callbacks
        self.anomaly_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def register_anomaly_callback(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback for anomaly detection.
        
        Args:
            callback: Function to call when an anomaly is detected
        """
        self.anomaly_callbacks.append(callback)
        self.logger.debug(f"Registered anomaly callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")

    def track_inference(
        self,
        input_id: str,
        prediction: str,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track a single inference.
        Writes records directly to disk when they reach the log_interval.
        
        Args:
            input_id: Unique identifier for the input
            prediction: Model's prediction
            confidence: Confidence score (0.0 to 1.0)
            latency_ms: Inference latency in milliseconds
            ground_truth: Correct label (if known)
            metadata: Additional metadata
            
        Returns:
            Dict[str, Any]: Record of the inference
        """
        # Create record
        record = {
            "input_id": input_id,
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "ground_truth": ground_truth,
            "timestamp": time.time(),
            "session_id": self.session_id,
            "metadata": metadata or {}
        }

        # Update metrics collectors
        self.metrics_collector.add_record(record)

        # Check for anomalies
        context = self.metrics_collector.get_statistical_context()
        is_anomaly, reasons = self.anomaly_detector.check_for_anomalies(record, context)

        if is_anomaly:
            record["anomaly"] = True
            record["anomaly_reasons"] = reasons
            self.logger.warning(f"Anomaly detected in inference {input_id}: {', '.join(reasons)}")

            # Call anomaly callbacks
            for callback in self.anomaly_callbacks:
                try:
                    callback(record)
                except Exception as e:
                    self.logger.error(f"Error in anomaly callback: {str(e)}")

        # Write record to disk
        self._write_record(record)

        return record

    def _write_record(self, record: Dict[str, Any]) -> None:
        """
        Write a record directly to disk.
        
        Args:
            record: Record to write
        """
        # Increment counter
        self._record_count += 1
        
        # Check if a new file needs to be created
        if self._current_log_file is None:
            timestamp = int(time.time())
            log_path = self.storage_dir / f"inference_log_{timestamp}_{self.session_id}.jsonl"
            
            # Create directory if it doesn't exist
            ensure_dir(log_path.parent)
            
            self._current_log_file = log_path
            self.logger.info(f"Created new log file: {log_path}")
        
        # Write record in JSONL format
        try:
            with open(self._current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record))
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Error writing record to log file: {str(e)}")
            raise MonitoringError(f"Failed to write record to log file: {str(e)}")
        
        # Check if a new file should be created for the next batch
        if self._record_count >= self.log_interval:
            self._current_log_file = None
            self._record_count = 0
            self.logger.debug(f"Log interval reached ({self.log_interval} records). Next record will start a new log file.")

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current monitoring metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of current metrics
        """
        metrics = self.metrics_collector.get_metrics()
        
        metrics["session_id"] = self.session_id
        
        return metrics

    def export_metrics(self, format: str = "json") -> str:
        """
        Export current metrics to a file.
        
        Args:
            format: Output format ('json' or 'csv')
            
        Returns:
            str: Path to the exported file
            
        Raises:
            ValueError: If unsupported format
        """
        metrics = self.get_current_metrics()
        timestamp = int(time.time())
        
        # Add timestamp to metrics to ensure CSV export includes it
        metrics["timestamp"] = timestamp
        
        if format.lower() == "json":
            filename = f"metrics_{timestamp}_{self.session_id}.json"
            file_path = self.storage_dir / filename
            save_json(metrics, file_path)
        
        elif format.lower() == "csv":
            filename = f"metrics_{timestamp}_{self.session_id}.csv"
            file_path = self.storage_dir / filename
            
            # Flatten metrics for CSV
            flat_metrics = self._flatten_metrics(metrics)
            
            # Write CSV
            import csv
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=flat_metrics.keys())
                writer.writeheader()
                writer.writerow(flat_metrics)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported metrics to {file_path}")
        return str(file_path)

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Flatten nested metrics dictionaries for CSV export.
        
        Args:
            metrics: Metrics dictionary
            prefix: Prefix for flattened keys
            
        Returns:
            Dict[str, Any]: Flattened metrics dictionary
        """
        flattened = {}
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                nested_flat = self._flatten_metrics(value, f"{prefix}{key}_")
                flattened.update(nested_flat)
            else:
                flattened[f"{prefix}{key}"] = value
        
        return flattened

    def close(self) -> None:
        """
        Close the monitor and save remaining logs.
        """
        # Export final metrics
        try:
            self.export_metrics("json")
            self.logger.info("Exported final metrics on close")
        except Exception as e:
            self.logger.error(f"Error exporting final metrics: {str(e)}")
        
        # Close log
        self.logger.info(f"Closing inference monitor session: {self.session_id}")
    
    def __enter__(self):
        """Context Manager entry point."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager exit point."""
        self.close()
        return False  # Don't suppress exceptions