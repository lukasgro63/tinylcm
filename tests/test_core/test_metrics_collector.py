"""Tests for metrics collector component."""

import pytest
import numpy as np

from tinylcm.core.metrics_collector import InferenceMetricsCollector


class TestInferenceMetricsCollector:
    """Test the inference metrics collector."""

    def setup_method(self):
        """Set up collector for testing."""
        self.collector = InferenceMetricsCollector(window_size=100)

    def test_add_record_updates_counters(self):
        """Test that adding records updates counters correctly."""
        # Add a record
        record = {
            "input_id": "test1",
            "prediction": "cat",
            "confidence": 0.8,
            "latency_ms": 20.0
        }
        
        self.collector.add_record(record)
        
        # Check counters
        assert self.collector.total_inferences == 1
        assert len(self.collector.confidence_window) == 1
        assert len(self.collector.latency_window) == 1
        assert self.collector.prediction_counts["cat"] == 1

    def test_get_metrics_returns_complete_metrics(self):
        """Test that get_metrics returns complete metrics."""
        # Add some test records
        for i in range(5):
            self.collector.add_record({
                "input_id": f"test{i}",
                "prediction": "cat" if i % 2 == 0 else "dog",
                "confidence": 0.7 + i * 0.05,
                "latency_ms": 15.0 + i * 3.0,
                "ground_truth": "cat" if i < 3 else "dog"
            })
        
        # Get metrics
        metrics = self.collector.get_metrics()
        
        # Check basic metrics
        assert metrics["total_inferences"] == 5
        assert "prediction_distribution" in metrics
        assert metrics["prediction_distribution"]["cat"] == 3
        assert metrics["prediction_distribution"]["dog"] == 2
        
        # Check latency metrics
        assert "latency" in metrics
        assert "mean" in metrics["latency"]
        assert "median" in metrics["latency"]
        assert "min" in metrics["latency"]
        assert "max" in metrics["latency"]
        
        # Check confidence metrics
        assert "confidence" in metrics
        assert "mean" in metrics["confidence"]
        assert "median" in metrics["confidence"]
        assert "min" in metrics["confidence"]
        assert "max" in metrics["confidence"]
        
        # Check accuracy
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_get_statistical_context(self):
        """Test generation of statistical context."""
        # Add records with varying latency and confidence
        for i in range(10):
            self.collector.add_record({
                "input_id": f"test{i}",
                "prediction": "cat" if i % 3 == 0 else ("dog" if i % 3 == 1 else "bird"),
                "confidence": 0.5 + i * 0.05,
                "latency_ms": 10.0 + i * 2.0
            })
        
        # Get statistical context
        context = self.collector.get_statistical_context()
        
        # Check prediction distribution
        assert "prediction_distribution" in context
        assert sum(context["prediction_distribution"].values()) == 10
        
        # Check confidence stats
        assert "confidence_stats" in context
        assert "mean" in context["confidence_stats"]
        assert "std" in context["confidence_stats"]
        assert "min" in context["confidence_stats"]
        assert context["confidence_stats"]["min"] == pytest.approx(0.5)
        assert context["confidence_stats"]["max"] == pytest.approx(0.95)
        
        # Check latency stats
        assert "latency_stats" in context
        assert "mean" in context["latency_stats"]
        assert "std" in context["latency_stats"]
        assert context["latency_stats"]["min"] == pytest.approx(10.0)
        assert context["latency_stats"]["max"] == pytest.approx(28.0)

    def test_reset_clears_all_data(self):
        """Test that reset clears all collected data."""
        # Add some records
        for i in range(5):
            self.collector.add_record({
                "input_id": f"test{i}",
                "prediction": "cat",
                "confidence": 0.8,
                "latency_ms": 20.0
            })
        
        # Verify data was added
        assert self.collector.total_inferences == 5
        assert len(self.collector.confidence_window) == 5
        
        # Reset collector
        self.collector.reset()
        
        # Verify all data was cleared
        assert self.collector.total_inferences == 0
        assert len(self.collector.confidence_window) == 0
        assert len(self.collector.latency_window) == 0
        assert sum(self.collector.prediction_counts.values()) == 0
        assert self.collector.ground_truth_correct == 0
        assert self.collector.ground_truth_total == 0

    def test_window_size_limits(self):
        """Test that window size limits are respected."""
        # Set a small window size
        collector = InferenceMetricsCollector(window_size=5)
        
        # Add more records than the window size
        for i in range(10):
            collector.add_record({
                "input_id": f"test{i}",
                "prediction": "cat",
                "confidence": 0.8,
                "latency_ms": 20.0 + i
            })
        
        # Check that windows have the correct size
        assert len(collector.confidence_window) == 5
        assert len(collector.latency_window) == 5
        
        # Check that the windows contain the most recent values
        assert collector.latency_window[-1] == 29.0  # Last added value