"""Tests für Anomalieerkennung-Komponenten."""

import pytest

from tinylcm.core.anomaly_detectors import (
    ThresholdAnomalyDetector,
    StatisticalAnomalyDetector,
    CompositeAnomalyDetector,
    anomaly_detector_registry
)


class TestThresholdAnomalyDetector:
    """Test ThresholdAnomalyDetector."""
    
    def test_low_confidence_detection(self):
        """Test, dass eine niedrige Konfidenz erkannt wird."""
        detector = ThresholdAnomalyDetector(confidence_threshold=0.5)
        
        # Aufzeichnung mit niedriger Konfidenz
        record = {"confidence": 0.3, "prediction": "cat"}
        
        is_anomaly, reasons = detector.check_for_anomalies(record, {})
        
        assert is_anomaly is True
        assert any("confidence" in reason.lower() for reason in reasons)
    
    def test_high_latency_detection(self):
        """Test, dass eine hohe Latenz erkannt wird."""
        detector = ThresholdAnomalyDetector(latency_threshold_ms=50.0)
        
        # Aufzeichnung mit hoher Latenz
        record = {"latency_ms": 75.0, "prediction": "cat"}
        
        is_anomaly, reasons = detector.check_for_anomalies(record, {})
        
        assert is_anomaly is True
        assert any("latency" in reason.lower() for reason in reasons)
    
    def test_incorrect_prediction_detection(self):
        """Test, dass eine falsche Vorhersage erkannt wird."""
        detector = ThresholdAnomalyDetector()
        
        # Aufzeichnung mit falscher Vorhersage
        record = {"prediction": "cat", "ground_truth": "dog"}
        
        is_anomaly, reasons = detector.check_for_anomalies(record, {})
        
        assert is_anomaly is True
        assert any("incorrect" in reason.lower() for reason in reasons)
    
    def test_normal_record(self):
        """Test, dass normale Aufzeichnungen keine Anomalien auslösen."""
        detector = ThresholdAnomalyDetector(confidence_threshold=0.5, latency_threshold_ms=50.0)
        
        # Normale Aufzeichnung
        record = {
            "confidence": 0.8,
            "latency_ms": 20.0,
            "prediction": "cat",
            "ground_truth": "cat"
        }
        
        is_anomaly, reasons = detector.check_for_anomalies(record, {})
        
        assert is_anomaly is False
        assert len(reasons) == 0


class TestStatisticalAnomalyDetector:
    """Test StatisticalAnomalyDetector."""
    
    def test_statistical_confidence_anomaly(self):
        """Test, dass statistische Konfidenzanomalien erkannt werden."""
        detector = StatisticalAnomalyDetector(confidence_z_threshold=-2.0)
        
        # Aufzeichnung mit niedriger Konfidenz
        record = {"confidence": 0.3, "prediction": "cat"}
        
        # Statistischer Kontext
        context = {
            "confidence_stats": {
                "mean": 0.8,
                "std": 0.2,
                "min": 0.5,
                "max": 0.95
            }
        }
        
        is_anomaly, reasons = detector.check_for_anomalies(record, context)
        
        assert is_anomaly is True
        assert any("statistically" in reason.lower() and "confidence" in reason.lower() for reason in reasons)
    
    def test_statistical_latency_anomaly(self):
        """Test, dass statistische Latenzanomalien erkannt werden."""
        detector = StatisticalAnomalyDetector(latency_z_threshold=2.0)
        
        # Aufzeichnung mit hoher Latenz
        record = {"latency_ms": 100.0, "prediction": "cat"}
        
        # Statistischer Kontext
        context = {
            "latency_stats": {
                "mean": 20.0,
                "std": 10.0,
                "min": 5.0,
                "max": 50.0
            }
        }
        
        is_anomaly, reasons = detector.check_for_anomalies(record, context)
        
        assert is_anomaly is True
        assert any("statistically" in reason.lower() and "latency" in reason.lower() for reason in reasons)


class TestCompositeAnomalyDetector:
    """Test CompositeAnomalyDetector."""
    
    def test_composite_detection(self):
        """Test, dass der zusammengesetzte Detektor Ergebnisse kombiniert."""
        # Erstelle zwei Testdetektoren
        threshold_detector = ThresholdAnomalyDetector(confidence_threshold=0.5)
        statistical_detector = StatisticalAnomalyDetector()
        
        # Erstelle zusammengesetzten Detektor
        composite = CompositeAnomalyDetector([threshold_detector, statistical_detector])
        
        # Aufzeichnung mit mehreren Anomalien
        record = {
            "confidence": 0.3,
            "latency_ms": 100.0,
            "prediction": "cat",
            "ground_truth": "dog"
        }
        
        # Statistischer Kontext
        context = {
            "confidence_stats": {
                "mean": 0.8,
                "std": 0.2,
                "min": 0.5,
                "max": 0.95
            },
            "latency_stats": {
                "mean": 20.0,
                "std": 10.0,
                "min": 5.0,
                "max": 50.0
            }
        }
        
        is_anomaly, reasons = composite.check_for_anomalies(record, context)
        
        assert is_anomaly is True
        # Sollte sowohl Grenzwert- als auch statistische Anomalien enthalten
        assert len(reasons) > 1
        threshold_anomaly = any("confidence" in reason.lower() and "statistically" not in reason.lower() for reason in reasons)
        statistical_anomaly = any("statistically" in reason.lower() for reason in reasons)
        assert threshold_anomaly and statistical_anomaly


class TestAnomalyDetectorRegistry:
    """Test anomaly_detector_registry."""
    
    def test_registry_contains_default_detectors(self):
        """Test, dass das Registry die Standarddetektoren enthält."""
        registered = anomaly_detector_registry.list_registered()
        
        assert "threshold" in registered
        assert "statistical" in registered
        assert "composite" in registered
    
    def test_can_create_detector_from_registry(self):
        """Test, dass Detektoren aus dem Registry erstellt werden können."""
        detector = anomaly_detector_registry.create("threshold", confidence_threshold=0.7)
        
        assert isinstance(detector, ThresholdAnomalyDetector)
        assert detector.confidence_threshold == 0.7