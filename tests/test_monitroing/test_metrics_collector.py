"""Tests für Metrics-Collector-Komponenten."""

import pytest

from tinylcm.core.monitoring.metrics_collector import InferenceMetricsCollector


class TestInferenceMetricsCollector:
    """Test InferenceMetricsCollector."""
    
    def test_add_record(self):
        """Test, dass Aufzeichnungen korrekt hinzugefügt werden."""
        collector = InferenceMetricsCollector()
        
        # Füge eine Aufzeichnung hinzu
        record = {
            "prediction": "cat",
            "confidence": 0.8,
            "latency_ms": 15.0
        }
        
        collector.add_record(record)
        
        # Prüfe internen Zustand
        assert collector.total_inferences == 1
        assert len(collector.confidence_window) == 1
        assert collector.confidence_window[0] == 0.8
        assert len(collector.latency_window) == 1
        assert collector.latency_window[0] == 15.0
        assert collector.prediction_counts["cat"] == 1
    
    def test_get_metrics_empty(self):
        """Test, dass get_metrics auch für leere Collector funktioniert."""
        collector = InferenceMetricsCollector()
        
        metrics = collector.get_metrics()
        
        assert "total_inferences" in metrics
        assert metrics["total_inferences"] == 0
        assert "prediction_distribution" in metrics
        assert len(metrics["prediction_distribution"]) == 0
    
    def test_get_metrics_with_data(self):
        """Test, dass get_metrics korrekte Metriken zurückgibt."""
        collector = InferenceMetricsCollector()
        
        # Füge einige Aufzeichnungen hinzu
        records = [
            {"prediction": "cat", "confidence": 0.8, "latency_ms": 10.0, "ground_truth": "cat"},
            {"prediction": "dog", "confidence": 0.7, "latency_ms": 15.0, "ground_truth": "dog"},
            {"prediction": "cat", "confidence": 0.9, "latency_ms": 12.0, "ground_truth": "cat"},
            {"prediction": "bird", "confidence": 0.6, "latency_ms": 20.0, "ground_truth": "cat"}
        ]
        
        for record in records:
            collector.add_record(record)
        
        metrics = collector.get_metrics()
        
        # Prüfe grundlegende Metriken
        assert metrics["total_inferences"] == 4
        assert "prediction_distribution" in metrics
        assert metrics["prediction_distribution"]["cat"] == 2
        assert metrics["prediction_distribution"]["dog"] == 1
        assert metrics["prediction_distribution"]["bird"] == 1
        
        # Prüfe Latenz- und Konfidenzmetriken
        assert "latency" in metrics
        assert "min_ms" in metrics["latency"]
        assert "max_ms" in metrics["latency"]
        assert "mean_ms" in metrics["latency"]
        
        assert "confidence" in metrics
        assert "min" in metrics["confidence"]
        assert "max" in metrics["confidence"]
        assert "mean" in metrics["confidence"]
        
        # Prüfe Genauigkeit
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 0.75  # 3/4 korrekt
    
    def test_get_statistical_context(self):
        """Test, dass get_statistical_context den korrekten Kontext liefert."""
        collector = InferenceMetricsCollector()
        
        # Füge einige Aufzeichnungen hinzu
        records = [
            {"prediction": "cat", "confidence": 0.8, "latency_ms": 10.0},
            {"prediction": "dog", "confidence": 0.7, "latency_ms": 15.0},
            {"prediction": "cat", "confidence": 0.9, "latency_ms": 12.0}
        ]
        
        for record in records:
            collector.add_record(record)
        
        context = collector.get_statistical_context()
        
        # Prüfe Kontext-Struktur
        assert "prediction_distribution" in context
        assert "confidence_stats" in context
        assert "latency_stats" in context
        
        # Prüfe statistische Werte
        assert "mean" in context["confidence_stats"]
        assert abs(context["confidence_stats"]["mean"] - 0.8) < 0.01  # ca. 0.8
        
        assert "std" in context["latency_stats"]
        assert context["latency_stats"]["min"] == 10.0
        assert context["latency_stats"]["max"] == 15.0
    
    def test_reset(self):
        """Test, dass reset alle Metriken zurücksetzt."""
        collector = InferenceMetricsCollector()
        
        # Füge eine Aufzeichnung hinzu
        record = {
            "prediction": "cat",
            "confidence": 0.8,
            "latency_ms": 15.0,
            "ground_truth": "cat"
        }
        
        collector.add_record(record)
        
        # Vergewissere dich, dass Daten vorhanden sind
        assert collector.total_inferences == 1
        
        # Reset
        collector.reset()
        
        # Prüfe internen Zustand nach Reset
        assert collector.total_inferences == 0
        assert len(collector.confidence_window) == 0
        assert len(collector.latency_window) == 0
        assert sum(collector.prediction_counts.values()) == 0
        assert collector.ground_truth_correct == 0
        assert collector.ground_truth_total == 0