"""Inference monitoring for TinyLCM.

Provides functionality for tracking and analyzing model inference performance,
including latency, confidence, and prediction distributions.
"""

import csv
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from tinylcm.constants import DEFAULT_INFERENCE_DIR, DEFAULT_LOG_INTERVAL, DEFAULT_MEMORY_ENTRIES
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.file_utils import ensure_dir, save_json
from tinylcm.utils.metrics import MetricsCalculator


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection strategies."""

    @abstractmethod
    def check_for_anomalies(
        self,
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check if a record contains anomalies.

        Args:
            record: The inference record to check
            context: Additional context information

        Returns:
            Tuple[bool, List[str]]: (is_anomaly, reasons)
        """
        pass


class ThresholdAnomalyDetector(AnomalyDetector):
    """Anomaly detector based on thresholds."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        latency_threshold_ms: float = 100.0
    ):
        """
        Initialize the threshold anomaly detector.

        Args:
            confidence_threshold: Minimum acceptable confidence
            latency_threshold_ms: Maximum acceptable latency in ms
        """
        self.confidence_threshold = confidence_threshold
        self.latency_threshold_ms = latency_threshold_ms

    def check_for_anomalies(
        self,
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check if record exceeds thresholds.

        Args:
            record: The inference record to check
            context: Additional context information

        Returns:
            Tuple[bool, List[str]]: (is_anomaly, reasons)
        """
        reasons = []

        # Check confidence
        if (
            "confidence" in record and
            record["confidence"] is not None and
            record["confidence"] < self.confidence_threshold
        ):
            reasons.append(f"Low confidence: {record['confidence']:.4f} < {self.confidence_threshold:.4f}")

        # Check latency
        if (
            "latency_ms" in record and
            record["latency_ms"] is not None and
            record["latency_ms"] > self.latency_threshold_ms
        ):
            reasons.append(f"High latency: {record['latency_ms']:.2f}ms > {self.latency_threshold_ms:.2f}ms")

        # Check prediction correctness if ground truth available
        if (
            "ground_truth" in record and
            record["ground_truth"] is not None and
            "prediction" in record and
            record["prediction"] != record["ground_truth"]
        ):
            reasons.append(f"Incorrect prediction: {record['prediction']} != {record['ground_truth']}")

        return bool(reasons), reasons


class StatisticalAnomalyDetector(AnomalyDetector):
    """Anomaly detector based on statistical measures."""

    def __init__(
        self,
        confidence_z_threshold: float = -2.0,
        latency_z_threshold: float = 2.0
    ):
        """
        Initialize the statistical anomaly detector.

        Args:
            confidence_z_threshold: Z-score threshold for confidence (negative means below average)
            latency_z_threshold: Z-score threshold for latency (positive means above average)
        """
        self.confidence_z_threshold = confidence_z_threshold
        self.latency_z_threshold = latency_z_threshold

    def check_for_anomalies(
        self,
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check if record is statistically anomalous.

        Args:
            record: The inference record to check
            context: Additional context information with statistical data

        Returns:
            Tuple[bool, List[str]]: (is_anomaly, reasons)
        """
        reasons = []

        # Check confidence
        if (
            "confidence" in record and
            record["confidence"] is not None and
            "confidence_stats" in context
        ):
            stats = context["confidence_stats"]
            if stats.get("std", 0) > 0:
                z_score = (record["confidence"] - stats.get("mean", 0)) / stats.get("std", 1)
                if z_score < self.confidence_z_threshold:
                    reasons.append(
                        f"Statistically low confidence: {record['confidence']:.4f}, "
                        f"z-score: {z_score:.2f} < {self.confidence_z_threshold:.2f}"
                    )

        # Check latency
        if (
            "latency_ms" in record and
            record["latency_ms"] is not None and
            "latency_stats" in context
        ):
            stats = context["latency_stats"]
            if stats.get("std", 0) > 0:
                z_score = (record["latency_ms"] - stats.get("mean", 0)) / stats.get("std", 1)
                if z_score > self.latency_z_threshold:
                    reasons.append(
                        f"Statistically high latency: {record['latency_ms']:.2f}ms, "
                        f"z-score: {z_score:.2f} > {self.latency_z_threshold:.2f}"
                    )

        # Check prediction distribution
        if (
            "prediction" in record and
            "prediction_distribution" in context
        ):
            distribution = context["prediction_distribution"]
            total = sum(distribution.values())
            if total > 0:
                prediction = record["prediction"]
                if prediction in distribution:
                    freq = distribution[prediction] / total
                    if freq < 0.05:  # Very rare prediction
                        reasons.append(
                            f"Rare prediction: '{prediction}' "
                            f"(frequency: {freq:.4f})"
                        )

        return bool(reasons), reasons


class CompositeAnomalyDetector(AnomalyDetector):
    """Composite anomaly detector that combines multiple detectors."""

    def __init__(self, detectors: List[AnomalyDetector]):
        """
        Initialize with list of detectors.

        Args:
            detectors: List of anomaly detectors
        """
        self.detectors = detectors

    def check_for_anomalies(
        self,
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check using all contained detectors.

        Args:
            record: The inference record to check
            context: Additional context information

        Returns:
            Tuple[bool, List[str]]: (is_anomaly, combined_reasons)
        """
        is_anomaly = False
        all_reasons = []

        for detector in self.detectors:
            detector_result, detector_reasons = detector.check_for_anomalies(record, context)

            if detector_result:
                is_anomaly = True
                all_reasons.extend(detector_reasons)

        return is_anomaly, all_reasons


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

        # Set configuration values
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir", DEFAULT_INFERENCE_DIR))
        self.memory_window_size = memory_window_size or component_config.get("memory_window_size", DEFAULT_MEMORY_ENTRIES)
        self.log_interval = log_interval or component_config.get("log_interval", DEFAULT_LOG_INTERVAL)

        # Create storage directory
        ensure_dir(self.storage_dir)

        # Initialize state
        self.session_id = str(uuid.uuid4())
        self.total_inferences = 0
        self.inference_records: List[Dict[str, Any]] = []
        self.latency_window: List[float] = []
        self.confidence_window: List[float] = []
        self.prediction_counts: Dict[str, int] = Counter()

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

        # Update internal state
        self.total_inferences += 1
        self.inference_records.append(record)

        if confidence is not None:
            self.confidence_window.append(confidence)
            # Keep window at specified size
            if len(self.confidence_window) > self.memory_window_size:
                self.confidence_window.pop(0)

        if latency_ms is not None:
            self.latency_window.append(latency_ms)
            # Keep window at specified size
            if len(self.latency_window) > self.memory_window_size:
                self.latency_window.pop(0)

        # Update prediction distribution
        self.prediction_counts[prediction] += 1

        # Check for anomalies
        context = self._get_statistical_context()
        is_anomaly, reasons = self.anomaly_detector.check_for_anomalies(record, context)

        if is_anomaly:
            record["anomaly"] = True
            record["anomaly_reasons"] = reasons

            # Call anomaly callbacks
            for callback in self.anomaly_callbacks:
                try:
                    callback(record)
                except Exception as e:
                    print(f"Error in anomaly callback: {str(e)}")

        # Check if we need to save logs
        if self.total_inferences % self.log_interval == 0:
            self._save_inference_logs()

        # Keep record list at specified size
        if len(self.inference_records) > self.memory_window_size:
            self.inference_records.pop(0)

        return record

    def _get_statistical_context(self) -> Dict[str, Any]:
        """
        Get statistical context for anomaly detection.

        Returns:
            Dict[str, Any]: Statistical context
        """
        context = {
            "prediction_distribution": dict(self.prediction_counts)
        }

        # Add confidence stats if available
        if self.confidence_window:
            context["confidence_stats"] = {
                "mean": np.mean(self.confidence_window),
                "std": np.std(self.confidence_window),
                "min": np.min(self.confidence_window),
                "max": np.max(self.confidence_window)
            }

        # Add latency stats if available
        if self.latency_window:
            context["latency_stats"] = {
                "mean": np.mean(self.latency_window),
                "std": np.std(self.latency_window),
                "min": np.min(self.latency_window),
                "max": np.max(self.latency_window)
            }

        return context

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current monitoring metrics.

        Returns:
            Dict[str, Any]: Dictionary of current metrics
        """
        metrics = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "total_inferences": self.total_inferences,
            "prediction_distribution": dict(self.prediction_counts)
        }

        # Add latency statistics
        if self.latency_window:
            metrics["latency"] = MetricsCalculator.calculate_latency_stats(self.latency_window)

        # Add confidence statistics
        if self.confidence_window:
            metrics["confidence"] = MetricsCalculator.calculate_confidence_stats(self.confidence_window)

        # Add accuracy if ground truth available
        ground_truth_records = [
            r for r in self.inference_records
            if r.get("ground_truth") is not None
        ]

        if ground_truth_records:
            predictions = [r["prediction"] for r in ground_truth_records]
            ground_truths = [r["ground_truth"] for r in ground_truth_records]

            try:
                accuracy = MetricsCalculator.accuracy(predictions, ground_truths)
                metrics["accuracy"] = accuracy
            except Exception as e:
                print(f"Error calculating accuracy: {str(e)}")

        return metrics

    def _save_inference_logs(self) -> str:
        """
        Save current inference records to disk.

        Returns:
            str: Path to the saved log file
        """
        if not self.inference_records:
            return ""

        # Create filename with timestamp and session ID
        timestamp = int(time.time())
        filename = f"inference_log_{timestamp}_{self.session_id}.json"
        log_path = self.storage_dir / filename

        # Save records
        save_json(self.inference_records, log_path)

        return str(log_path)

    def export_metrics(self, format: str = "json") -> str:
        """
        Export current metrics to a file.

        Args:
            format: Output format ('json' or 'csv')

        Returns:
            str: Path to the exported file
        """
        metrics = self.get_current_metrics()
        timestamp = int(time.time())

        if format.lower() == "json":
            filename = f"metrics_{timestamp}_{self.session_id}.json"
            file_path = self.storage_dir / filename
            save_json(metrics, file_path)

        elif format.lower() == "csv":
            filename = f"metrics_{timestamp}_{self.session_id}.csv"
            file_path = self.storage_dir / filename

            # Flatten metrics for CSV
            flat_metrics = self._flatten_metrics(metrics)

            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=flat_metrics.keys())
                writer.writeheader()
                writer.writerow(flat_metrics)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        return str(file_path)

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Flatten nested metrics dictionary for CSV export.

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
        Close the monitor and save any remaining logs.
        """
        if self.inference_records:
            self._save_inference_logs()

        # Export final metrics
        self.export_metrics("json")
