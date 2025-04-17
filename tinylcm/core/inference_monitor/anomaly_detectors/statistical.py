"""Statistical-based anomaly detection."""

from typing import Any, Dict, List, Tuple

import numpy as np

from tinylcm.core.inference_monitor.anomaly_detectors.base import AnomalyDetector
from tinylcm.utils.logging import setup_logger

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