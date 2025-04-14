"""Tests for InferenceMonitor component."""

import json
import os
import shutil
import tempfile
import time

import pytest

from tinylcm.core.monitoring.inference_monitor import InferenceMonitor
from tinylcm.core.monitoring.anomaly_detectors import ThresholdAnomalyDetector
from tinylcm.core.monitoring.metrics_collector import InferenceMetricsCollector


class TestInferenceMonitor:
    """Test InferenceMonitor functionality."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = InferenceMonitor(storage_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up resources and temporary directory."""
        self.monitor.close()
        shutil.rmtree(self.temp_dir)

    def test_track_inference(self):
        """Test tracking a single inference."""
        # Track an inference
        record = self.monitor.track_inference(
            input_id="test_input_1",
            prediction="cat",
            confidence=0.85,
            latency_ms=15.3,
            ground_truth="cat"
        )

        # Check record
        assert record["input_id"] == "test_input_1"
        assert record["prediction"] == "cat"
        assert record["confidence"] == 0.85
        assert record["latency_ms"] == 15.3
        assert record["ground_truth"] == "cat"
        assert "timestamp" in record
        assert "session_id" in record

    def test_get_current_metrics(self):
        """Test getting current monitoring metrics."""
        # Add some inferences
        for i in range(5):
            self.monitor.track_inference(
                input_id=f"input_{i}",
                prediction="cat" if i % 2 == 0 else "dog",
                confidence=0.8 + i * 0.02,
                latency_ms=10.0 + i * 2
            )

        # Get metrics
        metrics = self.monitor.get_current_metrics()

        # Check metrics
        assert metrics["total_inferences"] == 5
        assert "prediction_distribution" in metrics
        assert metrics["prediction_distribution"]["cat"] == 3
        assert metrics["prediction_distribution"]["dog"] == 2

        # Check latency stats
        assert "latency" in metrics
        assert "mean" in metrics["latency"]
        assert "median" in metrics["latency"]
        assert "min" in metrics["latency"]
        assert "max" in metrics["latency"]
        assert "p95" in metrics["latency"]

        # Check confidence stats
        assert "confidence" in metrics
        assert "mean" in metrics["confidence"]
        assert "median" in metrics["confidence"]
        assert "min" in metrics["confidence"]
        assert "max" in metrics["confidence"]

    def test_register_anomaly_callback(self):
        """Test registering and triggering anomaly callbacks."""
        # Create a mock callback
        callback_calls = []

        def anomaly_callback(record):
            callback_calls.append(record)

        # Register the callback
        self.monitor.register_anomaly_callback(anomaly_callback)

        # Track some normal inferences
        for i in range(3):
            self.monitor.track_inference(
                input_id=f"normal_{i}",
                prediction="cat",
                confidence=0.9,
                latency_ms=10.0
            )

        # No anomalies yet
        assert len(callback_calls) == 0

        # Track an inference with very low confidence (should trigger anomaly)
        self.monitor.track_inference(
            input_id="low_conf",
            prediction="cat",
            confidence=0.2,  # Low confidence
            latency_ms=10.0
        )

        # Callback should have been called
        assert len(callback_calls) == 1
        assert callback_calls[0]["input_id"] == "low_conf"
        assert callback_calls[0]["anomaly"] is True
        assert "anomaly_reasons" in callback_calls[0]
        assert any("confidence" in reason.lower() for reason in callback_calls[0]["anomaly_reasons"])

    def test_export_metrics(self):
        """Test exporting metrics to file."""
        # Add some inferences
        for i in range(10):
            self.monitor.track_inference(
                input_id=f"input_{i}",
                prediction="class_" + str(i % 3),
                confidence=0.7 + (i % 3) * 0.1,
                latency_ms=10.0 + i
            )

        # Export to JSON
        json_path = self.monitor.export_metrics(format="json")

        # Check that file exists
        assert os.path.exists(json_path)

        # Check content
        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        assert metrics["total_inferences"] == 10
        assert "prediction_distribution" in metrics
        assert "latency" in metrics
        assert "confidence" in metrics

        # Export to CSV
        csv_path = self.monitor.export_metrics(format="csv")

        # Check that file exists
        assert os.path.exists(csv_path)

        # Check content (basic check)
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_content = f.read()

        assert "timestamp" in csv_content
        assert "total_inferences" in csv_content
        assert "10" in csv_content  # Should contain the count

    def test_close(self):
        """Test closing the monitor and saving remaining logs."""
        # Add some inferences without hitting the log_interval
        for i in range(3):  # Default log_interval is 100
            self.monitor.track_inference(
                input_id=f"input_{i}",
                prediction="cat",
                confidence=0.9,
                latency_ms=10.0
            )

        # Close the monitor
        self.monitor.close()

        # Check that metrics were exported
        metrics_files = [f for f in os.listdir(self.temp_dir) if f.startswith("metrics_")]
        assert len(metrics_files) > 0