"""Registry for anomaly detectors."""

from tinylcm.core.inference_monitor.anomaly_detectors.base import AnomalyDetector
from tinylcm.core.inference_monitor.anomaly_detectors.threshold import ThresholdAnomalyDetector
from tinylcm.core.inference_monitor.anomaly_detectors.statistical import StatisticalAnomalyDetector
from tinylcm.core.inference_monitor.anomaly_detectors.composite import CompositeAnomalyDetector
from tinylcm.interfaces.registry import Registry

# Registry for anomaly detectors
anomaly_detector_registry = Registry(AnomalyDetector)

# Register standard detectors
anomaly_detector_registry.register("threshold", ThresholdAnomalyDetector)
anomaly_detector_registry.register("statistical", StatisticalAnomalyDetector)
anomaly_detector_registry.register("composite", CompositeAnomalyDetector)