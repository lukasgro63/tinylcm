# tests/test_monitoring/test_drift_detector.py
"""Tests for DriftDetector component."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tinylcm.core.drift_detector import (
    DriftDetector,
    DistributionDriftDetector,
    FeatureStatisticsDriftDetector,
    ConfidenceDriftDetector,
    PredictionFrequencyDriftDetector,
    CompositeDriftDetector,
    drift_detector_registry
)


class TestDriftDetector:
    """Test DriftDetector base class functionality."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = DriftDetector(
            storage_dir=self.temp_dir,
            window_size=100
        )

    def teardown_method(self):
        """Clean up resources and temporary directory."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
        shutil.rmtree(self.temp_dir)

    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        assert os.path.exists(self.temp_dir)
        assert os.path.exists(os.path.join(self.temp_dir, "reference"))
        assert os.path.exists(os.path.join(self.temp_dir, "snapshots"))

    def test_register_drift_callback(self):
        """Test registering callback for drift detection."""
        # Create a mock callback
        callback_calls = []

        def drift_callback(data):
            callback_calls.append(data)

        # Register the callback
        self.detector.register_drift_callback(drift_callback)

        # Test callback registration
        assert drift_callback in self.detector.drift_callbacks

    def test_create_reference_distribution(self):
        """Test creating a reference distribution."""
        # Create mock data
        mock_data = {
            "predictions": ["cat", "dog", "cat", "bird", "dog", "cat"],
            "confidences": [0.9, 0.8, 0.95, 0.7, 0.85, 0.9],
        }

        # Create reference distribution
        self.detector.create_reference_distribution(mock_data)

        # Check that reference file was created
        reference_files = [f for f in os.listdir(os.path.join(self.temp_dir, "reference")) 
                          if f.startswith("reference_")]
        assert len(reference_files) > 0

    def test_update_with_single_record(self):
        """Test updating the detector with a single record."""
        mock_record = {
            "prediction": "cat",
            "confidence": 0.9,
            "input_id": "test_1",
            "timestamp": time.time()
        }

        result = self.detector.update(mock_record)

        # Should return False since we don't have enough data for drift detection yet
        assert result is False

        # Check internal state
        assert len(self.detector.current_window) == 1

    def test_update_with_multiple_records(self):
        """Test updating the detector with multiple records."""
        # Create some mock records
        mock_records = []
        for i in range(10):
            record = {
                "prediction": "cat" if i % 3 == 0 else ("dog" if i % 3 == 1 else "bird"),
                "confidence": 0.7 + (i % 3) * 0.1,
                "input_id": f"test_{i}",
                "timestamp": time.time() + i
            }
            mock_records.append(record)

        # Update detector with each record
        for record in mock_records:
            self.detector.update(record)

        # Check state
        assert len(self.detector.current_window) == 10

    def test_export_current_state(self):
        """Test exporting the current state to a file."""
        # Add some mock data
        for i in range(5):
            record = {
                "prediction": "cat" if i % 2 == 0 else "dog",
                "confidence": 0.8 + i * 0.02,
                "input_id": f"test_{i}",
                "timestamp": time.time() + i
            }
            self.detector.update(record)

        # Export state
        state_path = self.detector.export_current_state()

        # Check that file exists
        assert os.path.exists(state_path)

        # Check content (basic check)
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        assert "timestamp" in state
        assert "window_size" in state
        assert "current_window" in state
        assert len(state["current_window"]) == 5


class TestDistributionDriftDetector:
    """Test DistributionDriftDetector functionality."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = DistributionDriftDetector(
            storage_dir=self.temp_dir,
            window_size=100,
            threshold=0.2
        )

    def teardown_method(self):
        """Clean up resources and temporary directory."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
        shutil.rmtree(self.temp_dir)

    def test_detect_drift_with_similar_distributions(self):
        """Test drift detection with similar distributions (no drift)."""
        # Create reference distribution (60% cat, 30% dog, 10% bird)
        reference_data = {"predictions": []}
        for _ in range(60):
            reference_data["predictions"].append("cat")
        for _ in range(30):
            reference_data["predictions"].append("dog")
        for _ in range(10):
            reference_data["predictions"].append("bird")

        self.detector.create_reference_distribution(reference_data)

        # Create current distribution (59% cat, 31% dog, 10% bird) - very similar
        for _ in range(59):
            self.detector.update({"prediction": "cat", "confidence": 0.9})
        for _ in range(31):
            self.detector.update({"prediction": "dog", "confidence": 0.9})
        for _ in range(10):
            self.detector.update({"prediction": "bird", "confidence": 0.9})

        # Check drift
        drift_result = self.detector.check_for_drift()
        
        # Should not detect drift since distributions are very similar
        assert drift_result["drift_detected"] is False
        assert "similarity_score" in drift_result
        assert drift_result["similarity_score"] > 0.95  # High similarity

    def test_detect_drift_with_different_distributions(self):
        """Test drift detection with different distributions (drift expected)."""
        # Create reference distribution (60% cat, 30% dog, 10% bird)
        reference_data = {"predictions": []}
        for _ in range(60):
            reference_data["predictions"].append("cat")
        for _ in range(30):
            reference_data["predictions"].append("dog")
        for _ in range(10):
            reference_data["predictions"].append("bird")

        self.detector.create_reference_distribution(reference_data)

        # Create current distribution (20% cat, 70% dog, 10% bird) - very different
        for _ in range(20):
            self.detector.update({"prediction": "cat", "confidence": 0.9})
        for _ in range(70):
            self.detector.update({"prediction": "dog", "confidence": 0.9})
        for _ in range(10):
            self.detector.update({"prediction": "bird", "confidence": 0.9})

        # Check drift
        drift_result = self.detector.check_for_drift()
        
        # Should detect drift since distributions are different
        assert drift_result["drift_detected"] is True
        assert "similarity_score" in drift_result
        assert drift_result["similarity_score"] < 0.8  # Lower similarity

    def test_drift_notification_callback(self):
        """Test that drift notifications trigger callbacks."""
        # Create a mock callback
        callback_calls = []

        def drift_callback(data):
            callback_calls.append(data)

        # Register the callback
        self.detector.register_drift_callback(drift_callback)

        # Create reference distribution
        reference_data = {"predictions": ["cat"] * 100}
        self.detector.create_reference_distribution(reference_data)

        # Create very different current distribution to ensure drift - auto_check=False um doppelte Aufrufe zu vermeiden
        for _ in range(100):
            self.detector.update({"prediction": "dog", "confidence": 0.9}, auto_check=False)

        # This should trigger the callback
        self.detector.check_for_drift()

        # Check that callback was called
        assert len(callback_calls) == 1

class TestFeatureStatisticsDriftDetector:
    """Test FeatureStatisticsDriftDetector functionality."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = FeatureStatisticsDriftDetector(
            storage_dir=self.temp_dir,
            window_size=100,
            threshold=0.1
        )

    def teardown_method(self):
        """Clean up resources and temporary directory."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
        shutil.rmtree(self.temp_dir)

    def test_detect_feature_drift(self):
        """Test detection of drift in feature statistics."""
        # Create reference data with features
        reference_data = {
            "features": [
                {"name": "feature1", "values": [0.1, 0.2, 0.3, 0.2, 0.1] * 20},
                {"name": "feature2", "values": [0.5, 0.6, 0.5, 0.7, 0.6] * 20}
            ]
        }
        
        self.detector.create_reference_distribution(reference_data)
        
        # Create current data with similar feature1 but different feature2
        for i in range(100):
            # feature1 is similar to reference
            feature1 = 0.1 + (i % 5) * 0.05
            
            # feature2 has significantly higher values than reference
            feature2 = 0.8 + (i % 5) * 0.04
            
            record = {
                "features": {
                    "feature1": feature1,
                    "feature2": feature2
                },
                "prediction": "cat",
                "confidence": 0.9
            }
            
            self.detector.update(record)
        
        # Check drift
        drift_result = self.detector.check_for_drift()
        
        # Should detect drift in feature2 but not feature1
        assert drift_result["drift_detected"] is True
        assert "feature_drifts" in drift_result
        assert "feature2" in drift_result["feature_drifts"]
        assert drift_result["feature_drifts"]["feature2"]["drift_detected"] is True
        assert drift_result["feature_drifts"].get("feature1", {}).get("drift_detected", False) is False


class TestConfidenceDriftDetector:
    """Test ConfidenceDriftDetector functionality."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = ConfidenceDriftDetector(
            storage_dir=self.temp_dir,
            window_size=100,
            threshold=0.1
        )

    def teardown_method(self):
        """Clean up resources and temporary directory."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
        shutil.rmtree(self.temp_dir)

    def test_detect_confidence_drift(self):
        """Test detection of drift in confidence values."""
        # Create reference data with high confidence
        reference_data = {
            "confidences": [0.8, 0.85, 0.9, 0.87, 0.82] * 20
        }
        
        self.detector.create_reference_distribution(reference_data)
        
        # Create current data with low confidence
        for _ in range(100):
            confidence = 0.5 + np.random.rand() * 0.2  # Between 0.5 and 0.7
            
            record = {
                "prediction": "cat",
                "confidence": confidence
            }
            
            self.detector.update(record)
        
        # Check drift
        drift_result = self.detector.check_for_drift()
        
        # Should detect drift in confidence
        assert drift_result["drift_detected"] is True
        assert "confidence_stats" in drift_result
        assert "reference_mean" in drift_result["confidence_stats"]
        assert "current_mean" in drift_result["confidence_stats"]
        assert drift_result["confidence_stats"]["reference_mean"] > 0.8
        assert drift_result["confidence_stats"]["current_mean"] < 0.7


class TestPredictionFrequencyDriftDetector:
    """Test PredictionFrequencyDriftDetector functionality."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = PredictionFrequencyDriftDetector(
            storage_dir=self.temp_dir,
            window_size=100,
            threshold=0.2
        )

    def teardown_method(self):
        """Clean up resources and temporary directory."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
        shutil.rmtree(self.temp_dir)

    def test_detect_prediction_frequency_drift(self):
        """Test detection of drift in prediction frequencies."""
        # Create reference data with balanced classes
        reference_data = {
            "predictions": ["cat"] * 33 + ["dog"] * 33 + ["bird"] * 34
        }
        
        self.detector.create_reference_distribution(reference_data)
        
        # Create current data with imbalanced classes
        for _ in range(80):
            self.detector.update({"prediction": "cat", "confidence": 0.9})
        for _ in range(15):
            self.detector.update({"prediction": "dog", "confidence": 0.9})
        for _ in range(5):
            self.detector.update({"prediction": "bird", "confidence": 0.9})
        
        # Check drift
        drift_result = self.detector.check_for_drift()
        
        # Should detect drift in prediction frequencies
        assert drift_result["drift_detected"] is True
        assert "class_frequencies" in drift_result
        assert "reference" in drift_result["class_frequencies"]
        assert "current" in drift_result["class_frequencies"]
        assert drift_result["class_frequencies"]["reference"]["cat"] == pytest.approx(0.33, abs=0.01)
        assert drift_result["class_frequencies"]["current"]["cat"] == pytest.approx(0.8, abs=0.01)


class TestCompositeDriftDetector:
    """Test CompositeDriftDetector functionality."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sub-detectors
        self.confidence_detector = ConfidenceDriftDetector(
            storage_dir=os.path.join(self.temp_dir, "confidence"),
            window_size=100,
            threshold=0.1
        )
        
        self.prediction_detector = PredictionFrequencyDriftDetector(
            storage_dir=os.path.join(self.temp_dir, "prediction"),
            window_size=100,
            threshold=0.2
        )
        
        # Create composite detector
        self.detector = CompositeDriftDetector(
            storage_dir=self.temp_dir,
            detectors=[self.confidence_detector, self.prediction_detector]
        )

    def teardown_method(self):
        """Clean up resources and temporary directory."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
        shutil.rmtree(self.temp_dir)

    def test_composite_drift_detection(self):
        """Test that composite detector combines results from sub-detectors."""
        # Create reference data
        reference_data = {
            "confidences": [0.8, 0.85, 0.9, 0.87, 0.82] * 20,
            "predictions": ["cat"] * 33 + ["dog"] * 33 + ["bird"] * 34
        }
        
        self.detector.create_reference_distribution(reference_data)
        
        # First test: No drift in any detector
        for _ in range(100):
            confidence = 0.8 + np.random.rand() * 0.1  # Similar confidence to reference
            prediction = np.random.choice(["cat", "dog", "bird"], p=[0.33, 0.33, 0.34])  # Similar distribution
            
            record = {
                "prediction": prediction,
                "confidence": confidence
            }
            
            self.detector.update(record)
        
        # Check drift
        drift_result = self.detector.check_for_drift()
        
        # Should not detect drift
        assert drift_result["drift_detected"] is False
        assert "detector_results" in drift_result
        assert len(drift_result["detector_results"]) == 2
        
        # Reset window
        self.detector.reset()
        
        # Second test: Drift in confidence but not in prediction
        for _ in range(100):
            confidence = 0.5 + np.random.rand() * 0.2  # Lower confidence than reference
            prediction = np.random.choice(["cat", "dog", "bird"], p=[0.33, 0.33, 0.34])  # Similar distribution
            
            record = {
                "prediction": prediction,
                "confidence": confidence
            }
            
            self.detector.update(record)
        
        # Check drift
        drift_result = self.detector.check_for_drift()
        
        # Should detect drift overall
        assert drift_result["drift_detected"] is True
        assert "detector_results" in drift_result
        assert any(d["drift_detected"] for d in drift_result["detector_results"])
        
        # Reset window
        self.detector.reset()
        
        # Third test: Drift in both confidence and prediction
        for _ in range(80):
            confidence = 0.5 + np.random.rand() * 0.2  # Lower confidence than reference
            
            record = {
                "prediction": "cat",  # Imbalanced prediction
                "confidence": confidence
            }
            
            self.detector.update(record)
            
        for _ in range(20):
            confidence = 0.5 + np.random.rand() * 0.2  # Lower confidence than reference
            prediction = np.random.choice(["dog", "bird"])
            
            record = {
                "prediction": prediction,
                "confidence": confidence
            }
            
            self.detector.update(record)
        
        # Check drift
        drift_result = self.detector.check_for_drift()
        
        # Should detect drift in both detectors
        assert drift_result["drift_detected"] is True
        assert "detector_results" in drift_result
        assert all(d["drift_detected"] for d in drift_result["detector_results"])


class TestDriftDetectorRegistry:
    """Test the drift detector registry."""
    
    def test_registry_contains_expected_detectors(self):
        """Test that registry contains the expected detectors."""
        registered = drift_detector_registry.list_registered()
        
        assert "distribution" in registered
        assert "feature" in registered
        assert "confidence" in registered
        assert "prediction" in registered
        assert "composite" in registered
    
    def test_create_detector_from_registry(self):
        """Test creating a detector from the registry."""
        detector = drift_detector_registry.create("confidence", threshold=0.05)
        
        assert isinstance(detector, ConfidenceDriftDetector)
        assert detector.threshold == 0.05