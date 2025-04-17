"""Composite anomaly detection combining multiple detectors."""

from typing import Any, Dict, List, Tuple

from tinylcm.core.inference_monitor.anomaly_detectors.base import AnomalyDetector
from tinylcm.utils.logging import setup_logger

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