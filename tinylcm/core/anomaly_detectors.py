"""Anomaly detection strategies for InferenceMonitor."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from tinylcm.interfaces.monitoring import AnomalyDetector
from tinylcm.utils.logging import setup_logger

logger = setup_logger(__name__)

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
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

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
            (is_anomaly, reasons): Anomaly status and reasons
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
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

    def check_for_anomalies(
        self,
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check if record is statistically anomalous.

        Args:
            record: The inference record to check
            context: Additional context with statistical data

        Returns:
            (is_anomaly, reasons): Anomaly status and reasons
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
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

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
            (is_anomaly, reasons): Combined anomaly status and reasons
        """
        is_anomaly = False
        all_reasons = []

        for detector in self.detectors:
            detector_result, detector_reasons = detector.check_for_anomalies(record, context)

            if detector_result:
                is_anomaly = True
                all_reasons.extend(detector_reasons)

        return is_anomaly, all_reasons


# Registry for anomaly detectors
from tinylcm.interfaces.registry import Registry

anomaly_detector_registry = Registry(AnomalyDetector)

# Register standard detectors
anomaly_detector_registry.register("threshold", ThresholdAnomalyDetector)
anomaly_detector_registry.register("statistical", StatisticalAnomalyDetector)
anomaly_detector_registry.register("composite", CompositeAnomalyDetector)