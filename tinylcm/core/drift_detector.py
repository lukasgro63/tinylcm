"""Drift detection for machine learning models.

This module provides components for detecting drift in model performance,
input data distributions, and prediction patterns.
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from tinylcm.interfaces.monitoring import DriftDetector as IDriftDetector
from tinylcm.interfaces.registry import Registry
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.file_utils import ensure_dir, load_json, save_json
from tinylcm.utils.metrics import MetricsCalculator
from tinylcm.utils.logging import setup_logger

# Set up logger
logger = setup_logger(__name__)

class DriftDetector(IDriftDetector):
    """
    Base class for drift detection.
    
    Tracks reference and current distributions to detect drift
    in model behavior or input data.
    """
    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        window_size: int = 100,
        threshold: float = 0.1,
        config: Optional[Config] = None
    ):
        """
        Initialize the drift detector.
        
        Args:
            storage_dir: Directory for storing reference data and state
            window_size: Size of the sliding window for current data
            threshold: Threshold for detecting drift
            config: Configuration object
        """
        self.config = config or get_config()
        component_config = self.config.get_component_config("drift_detector")
        
        # Set up logger
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Storage settings
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir"))
        self.reference_dir = ensure_dir(self.storage_dir / "reference")
        self.snapshots_dir = ensure_dir(self.storage_dir / "snapshots")
        
        # Configuration
        self.window_size = window_size
        self.threshold = threshold
        self.session_id = str(uuid.uuid4())
        
        # State
        self.current_window = deque(maxlen=window_size)
        self.reference_distribution = None
        self.reference_metadata = {}
        self.last_check_time = 0
        
        # Callbacks
        self.drift_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        self.logger.info(f"Initialized drift detector with session ID: {self.session_id}")
        self.logger.info(f"Using window size: {window_size}, threshold: {threshold}")
    
    def register_drift_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function for drift detection events.
        
        Args:
            callback: Function to call when drift is detected
        """
        self.drift_callbacks.append(callback)
        self.logger.debug(f"Registered drift callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def create_reference_distribution(self, data: Dict[str, Any]) -> None:
        """
        Create a reference distribution from data.
        
        Args:
            data: Dictionary with reference data
        """
        self.reference_distribution = self._process_reference_data(data)
        self.reference_metadata = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "sample_count": self._get_sample_count(data),
            "created_at": time.time(),
            "window_size": self.window_size,
            "threshold": self.threshold
        }
        
        # Save reference distribution
        self._save_reference_distribution()
        
        self.logger.info(f"Created reference distribution from {self._get_sample_count(data)} samples")
    
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reference data to extract distribution information.
        
        To be overridden by subclasses.
        
        Args:
            data: Dictionary with reference data
            
        Returns:
            Dictionary containing reference distribution data
        """
        # Default implementation just returns the data
        return data
    
    def _get_sample_count(self, data: Dict[str, Any]) -> int:
        """
        Get the number of samples in the data.
        
        Args:
            data: Dictionary with data
            
        Returns:
            int: Number of samples
        """
        # Try to infer from common fields
        for key in ["predictions", "confidences", "features"]:
            if key in data and isinstance(data[key], list):
                return len(data[key])
        
        # Default
        return 0
    
    def _save_reference_distribution(self) -> str:
        """
        Save the reference distribution to disk.
        
        Returns:
            str: Path to the saved reference file
        """
        if self.reference_distribution is None:
            self.logger.warning("Attempted to save None reference distribution")
            return ""
        
        timestamp = int(time.time())
        file_path = self.reference_dir / f"reference_{timestamp}_{self.session_id}.json"
        
        data = {
            "distribution": self.reference_distribution,
            "metadata": self.reference_metadata
        }
        
        save_json(data, file_path)
        self.logger.debug(f"Saved reference distribution to {file_path}")
        
        return str(file_path)
    
    def _load_reference_distribution(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Load a reference distribution from disk.
        
        Args:
            file_path: Path to the reference file (if None, latest is used)
            
        Returns:
            bool: True if loaded successfully
        """
        if file_path is None:
            # Find latest reference file
            reference_files = list(self.reference_dir.glob("reference_*.json"))
            if not reference_files:
                self.logger.warning("No reference files found")
                return False
            
            reference_files.sort(reverse=True)  # Latest first
            file_path = reference_files[0]
        
        try:
            data = load_json(file_path)
            self.reference_distribution = data.get("distribution")
            self.reference_metadata = data.get("metadata", {})
            
            self.logger.info(f"Loaded reference distribution from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load reference distribution: {e}")
            return False
    
    def update(self, record: Dict[str, Any], auto_check: bool = True) -> bool:
        """
        Update the detector with a new record.
        
        Args:
            record: New data record
            auto_check: If True, automatically check for drift based on interval
            
        Returns:
            bool: True if drift was detected after this update
        """
        self.current_window.append(record)
        
        # Return False if we don't have a reference distribution yet
        if self.reference_distribution is None:
            return False
        
        # Return False if we don't have enough data yet
        if len(self.current_window) < self.window_size:
            return False
        
        # Check for drift if auto_check is enabled and it's been a while since last check
        if auto_check:
            current_time = time.time()
            check_interval = self.config.get("drift_detector", "check_interval", 60.0)  # Default 60 seconds
            
            if current_time - self.last_check_time > check_interval:
                drift_result = self.check_for_drift()
                self.last_check_time = current_time
                return bool(drift_result.get("drift_detected", False))
        
        return False
    
    def check_for_drift(self) -> Dict[str, Any]:
        """
        Check if drift has occurred between reference and current distributions.
        
        Returns:
            Dict[str, Any]: Drift check results
        """
        # Check that we have a reference distribution
        if self.reference_distribution is None:
            self.logger.warning("Cannot check for drift: No reference distribution")
            return {"drift_detected": False, "error": "No reference distribution"}
        
        # Check that we have enough data
        if len(self.current_window) < self.window_size / 2:  # Allow checking with at least half window
            self.logger.warning(f"Not enough data to check for drift: {len(self.current_window)}/{self.window_size}")
            return {
                "drift_detected": False, 
                "error": f"Not enough data: {len(self.current_window)}/{self.window_size}"
            }
        
        # Calculate drift metrics
        drift_result = self._calculate_drift()
        
        # Ensure drift_detected is a Python bool, not numpy bool
        drift_detected = bool(drift_result.get("drift_detected", False))
        drift_result["drift_detected"] = drift_detected
        
        # If drift detected, call callbacks
        if drift_detected:
            self.logger.warning(f"Drift detected: {drift_result.get('drift_type', 'unknown')}")
            
            # Add snapshot to result
            drift_result["snapshot"] = self._save_current_state()
            
            # Create a copy of the result for callbacks to prevent modifying the return value
            callback_data = drift_result.copy()
            
            # Call callbacks with the copy
            for callback in self.drift_callbacks:
                try:
                    callback(callback_data)
                except Exception as e:
                    self.logger.error(f"Error in drift callback: {e}")
        
        return drift_result
    
    def _calculate_drift(self) -> Dict[str, Any]:
        """
        Calculate drift metrics between reference and current distributions.
        
        To be overridden by subclasses.
        
        Returns:
            Dict[str, Any]: Drift metrics
        """
        # Default implementation always returns no drift
        return {
            "drift_detected": False,
            "timestamp": time.time(),
            "session_id": self.session_id,
            "window_size": len(self.current_window),
            "threshold": self.threshold
        }
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.current_window.clear()
        self.last_check_time = 0
        self.logger.info("Reset drift detector state")
    
    def export_current_state(self) -> str:
        """
        Export the current state to a file.
        
        Returns:
            str: Path to the exported file
        """
        return self._save_current_state()
    
    def _save_current_state(self) -> str:
        """
        Save the current state to disk.
        
        Returns:
            str: Path to the saved state file
        """
        timestamp = int(time.time())
        file_path = self.snapshots_dir / f"snapshot_{timestamp}_{self.session_id}.json"
        
        state = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "current_window": list(self.current_window)
        }
        
        save_json(state, file_path)
        self.logger.debug(f"Saved current state to {file_path}")
        
        return str(file_path)
    
    def close(self) -> None:
        """Close the detector and clean up resources."""
        self.logger.info(f"Closing drift detector session: {self.session_id}")


class DistributionDriftDetector(DriftDetector):
    """
    Drift detector based on comparing probability distributions.
    
    Detects drift by comparing the distribution of predictions
    between reference and current data using statistical measures.
    """
    
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reference data to extract prediction distribution.
        
        Args:
            data: Dictionary with reference data
            
        Returns:
            Dictionary containing reference distribution data
        """
        distribution = {}
        
        # Extract predictions if available
        if "predictions" in data and isinstance(data["predictions"], list):
            predictions = data["predictions"]
            counter = Counter(predictions)
            total = len(predictions)
            
            if total > 0:
                distribution["class_distribution"] = {
                    cls: count / total for cls, count in counter.items()
                }
        
        return distribution
    
    def _calculate_drift(self) -> Dict[str, Any]:
        """
        Calculate drift by comparing prediction distributions.
        
        Returns:
            Dict[str, Any]: Drift results
        """
        result = super()._calculate_drift()
        
        if self.reference_distribution is None or "class_distribution" not in self.reference_distribution:
            return {**result, "error": "Invalid reference distribution"}
        
        # Extract current distribution
        predictions = [r.get("prediction") for r in self.current_window if "prediction" in r]
        counter = Counter(predictions)
        total = len(predictions)
        
        if total == 0:
            return {**result, "error": "No predictions in current window"}
        
        current_distribution = {cls: count / total for cls, count in counter.items()}
        
        # Compare distributions
        reference_dist = self.reference_distribution["class_distribution"]
        
        # KORREKTUR: Direkt TVD (Total Variation Distance) berechnen statt die Similarity-Funktion zu nutzen
        # TVD ist eine einfache und robuste Metrik für kategorische Verteilungen
        tvd = 0.0
        all_categories = set(reference_dist.keys()) | set(current_distribution.keys())
        
        for category in all_categories:
            ref_prob = reference_dist.get(category, 0.0)
            cur_prob = current_distribution.get(category, 0.0)
            tvd += abs(ref_prob - cur_prob)
        
        # Normalisieren (TVD hat einen Wertebereich von 0 bis 1)
        tvd = tvd / 2.0
        
        # Drift-Erkennung basierend auf TVD
        drift_detected = bool(tvd > self.threshold)
        
        # Zur Diagnose loggen
        self.logger.debug(f"TVD between distributions: {tvd}, threshold: {self.threshold}")
        self.logger.debug(f"Reference distribution: {reference_dist}")
        self.logger.debug(f"Current distribution: {current_distribution}")
        
        # Create result
        result.update({
            "drift_detected": drift_detected,
            "drift_type": "prediction_distribution",
            "distance": float(tvd),
            "similarity_score": float(1 - tvd),
            "threshold": self.threshold,
            "reference_distribution": reference_dist,
            "current_distribution": current_distribution
        })
        
        return result


class FeatureStatisticsDriftDetector(DriftDetector):
    """
    Drift detector based on feature statistics.
    
    Detects drift by comparing statistical properties of features
    between reference and current data.
    """
    
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reference data to extract feature statistics.
        
        Args:
            data: Dictionary with reference data
            
        Returns:
            Dictionary containing reference feature statistics
        """
        stats = {}
        
        # Process features if available
        if "features" in data and isinstance(data["features"], list):
            # Format: List of {"name": feature_name, "values": [values...]}
            for feature in data["features"]:
                if "name" in feature and "values" in feature and isinstance(feature["values"], list):
                    name = feature["name"]
                    values = feature["values"]
                    
                    if values:
                        values_array = np.array(values)
                        stats[name] = {
                            "mean": float(np.mean(values_array)),
                            "std": float(np.std(values_array)),
                            "min": float(np.min(values_array)),
                            "max": float(np.max(values_array)),
                            "median": float(np.median(values_array))
                        }
        
        return {"feature_stats": stats}
    
    def _calculate_drift(self) -> Dict[str, Any]:
        """
        Calculate drift by comparing feature statistics.
        
        Returns:
            Dict[str, Any]: Drift results
        """
        result = super()._calculate_drift()
        
        if self.reference_distribution is None or "feature_stats" not in self.reference_distribution:
            return {**result, "error": "Invalid reference distribution"}
        
        # Extract features from current window
        feature_values = defaultdict(list)
        
        for record in self.current_window:
            if "features" in record and isinstance(record["features"], dict):
                for name, value in record["features"].items():
                    if isinstance(value, (int, float)):
                        feature_values[name].append(value)
        
        # Calculate current statistics
        current_stats = {}
        for name, values in feature_values.items():
            if values:
                values_array = np.array(values)
                current_stats[name] = {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "median": float(np.median(values_array))
                }
        
        # Compare feature statistics
        reference_stats = self.reference_distribution["feature_stats"]
        feature_drifts = {}
        any_drift = False
        
        for name in set(reference_stats.keys()) & set(current_stats.keys()):
            ref_stat = reference_stats[name]
            cur_stat = current_stats[name]
            
            # Calculate Z-score for mean
            if ref_stat["std"] > 0:
                z_score = abs(cur_stat["mean"] - ref_stat["mean"]) / ref_stat["std"]
                
                # Check for drift
                feature_drift = bool(z_score > 2.0)  # More than 2 standard deviations, ensure Python bool
                
                feature_drifts[name] = {
                    "drift_detected": feature_drift,
                    "z_score": float(z_score),  # Ensure Python float
                    "reference": ref_stat,
                    "current": cur_stat
                }
                
                if feature_drift:
                    any_drift = True
        
        # Create result
        result.update({
            "drift_detected": bool(any_drift),  # Ensure Python bool
            "drift_type": "feature_statistics",
            "feature_drifts": feature_drifts,
            "threshold": self.threshold
        })
        
        return result


class ConfidenceDriftDetector(DriftDetector):
    """
    Drift detector based on confidence scores.
    
    Detects drift by comparing confidence score distributions
    between reference and current data.
    """
    
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reference data to extract confidence statistics.
        
        Args:
            data: Dictionary with reference data
            
        Returns:
            Dictionary containing reference confidence statistics
        """
        stats = {}
        
        # Extract confidences if available
        if "confidences" in data and isinstance(data["confidences"], list):
            confidences = data["confidences"]
            
            if confidences:
                conf_array = np.array(confidences)
                stats["confidence_stats"] = {
                    "mean": float(np.mean(conf_array)),
                    "std": float(np.std(conf_array)),
                    "min": float(np.min(conf_array)),
                    "max": float(np.max(conf_array)),
                    "median": float(np.median(conf_array))
                }
        
        return stats
    
    def _calculate_drift(self) -> Dict[str, Any]:
        """
        Calculate drift by comparing confidence distributions.
        
        Returns:
            Dict[str, Any]: Drift results
        """
        result = super()._calculate_drift()
        
        if self.reference_distribution is None or "confidence_stats" not in self.reference_distribution:
            return {**result, "error": "Invalid reference distribution"}
        
        # Extract confidences from current window
        confidences = [r.get("confidence") for r in self.current_window if "confidence" in r]
        
        if not confidences:
            return {**result, "error": "No confidence values in current window"}
        
        # Calculate current statistics
        conf_array = np.array(confidences)
        current_stats = {
            "mean": float(np.mean(conf_array)),
            "std": float(np.std(conf_array)),
            "min": float(np.min(conf_array)),
            "max": float(np.max(conf_array)),
            "median": float(np.median(conf_array))
        }
        
        # Compare confidence statistics
        reference_stats = self.reference_distribution["confidence_stats"]
        
        # Calculate relative difference in means
        ref_mean = reference_stats["mean"]
        cur_mean = current_stats["mean"]
        
        if ref_mean > 0:
            relative_diff = abs(cur_mean - ref_mean) / ref_mean
            drift_detected = bool(relative_diff > self.threshold)  # Ensure Python bool
        else:
            # Fallback if reference mean is 0
            drift_detected = bool(abs(cur_mean - ref_mean) > self.threshold)  # Ensure Python bool
        
        # Create result
        result.update({
            "drift_detected": drift_detected,
            "drift_type": "confidence_scores",
            "confidence_stats": {
                "reference_mean": float(reference_stats["mean"]),  # Ensure Python float
                "current_mean": float(current_stats["mean"]),  # Ensure Python float
                "relative_difference": float(relative_diff) if ref_mean > 0 else None,
                "reference": reference_stats,
                "current": current_stats
            },
            "threshold": self.threshold
        })
        
        return result


class PredictionFrequencyDriftDetector(DriftDetector):
    """
    Drift detector based on prediction frequencies.
    
    Detects drift by comparing how often each class is predicted
    between reference and current data.
    """
    
    def _process_reference_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reference data to extract prediction frequencies.
        
        Args:
            data: Dictionary with reference data
            
        Returns:
            Dictionary containing reference prediction frequencies
        """
        frequencies = {}
        
        # Extract predictions if available
        if "predictions" in data and isinstance(data["predictions"], list):
            predictions = data["predictions"]
            counter = Counter(predictions)
            total = len(predictions)
            
            if total > 0:
                frequencies["class_frequencies"] = {
                    cls: count / total for cls, count in counter.items()
                }
        
        return frequencies
    
    def _calculate_drift(self) -> Dict[str, Any]:
        """
        Calculate drift by comparing prediction frequencies.
        
        Returns:
            Dict[str, Any]: Drift results
        """
        result = super()._calculate_drift()
        
        if self.reference_distribution is None or "class_frequencies" not in self.reference_distribution:
            return {**result, "error": "Invalid reference distribution"}
        
        # Extract current prediction frequencies
        predictions = [r.get("prediction") for r in self.current_window if "prediction" in r]
        counter = Counter(predictions)
        total = len(predictions)
        
        if total == 0:
            return {**result, "error": "No predictions in current window"}
        
        current_frequencies = {cls: count / total for cls, count in counter.items()}
        
        # Compare frequencies
        reference_freq = self.reference_distribution["class_frequencies"]
        
        # Calculate Earth Mover's Distance (EMD) / Wasserstein distance
        # Simplified version for categorical distributions
        all_classes = set(reference_freq.keys()) | set(current_frequencies.keys())
        
        # Total variation distance
        tvd = 0.0
        for cls in all_classes:
            ref_freq = reference_freq.get(cls, 0.0)
            cur_freq = current_frequencies.get(cls, 0.0)
            tvd += abs(ref_freq - cur_freq)
        
        tvd = tvd / 2.0  # Normalize
        
        # Determine if drift detected
        drift_detected = bool(tvd > self.threshold)  # Ensure Python bool
        
        # Create result
        result.update({
            "drift_detected": drift_detected,
            "drift_type": "prediction_frequency",
            "distance": float(tvd),  # Ensure Python float
            "threshold": self.threshold,
            "class_frequencies": {
                "reference": reference_freq,
                "current": current_frequencies
            }
        })
        
        return result


class CompositeDriftDetector(DriftDetector):
    """
    Composite drift detector that combines multiple detectors.
    
    Allows using multiple drift detection strategies together.
    """
    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        detectors: Optional[List[DriftDetector]] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the composite drift detector.
        
        Args:
            storage_dir: Directory for storing reference data and state
            detectors: List of drift detectors to use
            config: Configuration object
        """
        super().__init__(storage_dir=storage_dir, config=config)
        
        # Set up detectors
        self.detectors = detectors or []
        
        self.logger.info(f"Initialized composite drift detector with {len(self.detectors)} sub-detectors")
    
    def create_reference_distribution(self, data: Dict[str, Any]) -> None:
        """
        Create reference distributions for all detectors.
        
        Args:
            data: Dictionary with reference data
        """
        # Call parent implementation
        super().create_reference_distribution(data)
        
        # Create reference for each detector
        for detector in self.detectors:
            detector.create_reference_distribution(data)
    
    def update(self, record: Dict[str, Any]) -> bool:
        """
        Update all detectors with a new record.
        
        Args:
            record: New data record
            
        Returns:
            bool: True if drift was detected by any detector
        """
        # Call parent implementation
        parent_result = super().update(record)
        
        # Update each detector
        any_drift = parent_result
        
        for detector in self.detectors:
            detector_result = detector.update(record)
            any_drift = any_drift or detector_result
        
        return bool(any_drift)  # Ensure Python bool
    
    def check_for_drift(self) -> Dict[str, Any]:
        """
        Check all detectors for drift.
        
        Returns:
            Dict[str, Any]: Combined drift check results
        """
        # Get results from all detectors
        detector_results = []
        
        for detector in self.detectors:
            detector_result = detector.check_for_drift()
            detector_results.append(detector_result)
        
        # Determine if any detector detected drift - ensure Python bool
        any_drift = bool(any(bool(result.get("drift_detected", False)) for result in detector_results))
        
        # Create composite result
        result = {
            "drift_detected": any_drift,
            "drift_type": "composite",
            "timestamp": time.time(),
            "session_id": self.session_id,
            "detector_results": detector_results,  # Ensure this is always included
            "threshold": self.threshold,
            "window_size": len(self.current_window)
        }
        
        # If drift detected, call callbacks
        if any_drift:
            self.logger.warning("Composite drift detected")
            
            # Add snapshot to result
            result["snapshot"] = self._save_current_state()
            
            # Create a copy of the result for callbacks
            callback_data = result.copy()
            
            # Call callbacks with the copy
            for callback in self.drift_callbacks:
                try:
                    callback(callback_data)
                except Exception as e:
                    self.logger.error(f"Error in drift callback: {e}")
        
        return result
    
    def reset(self) -> None:
        """Reset all detectors."""
        # Call parent implementation
        super().reset()
        
        # Reset each detector
        for detector in self.detectors:
            detector.reset()
    
    def close(self) -> None:
        """Close all detectors."""
        # Call parent implementation
        super().close()
        
        # Close each detector
        for detector in self.detectors:
            detector.close()


# Create registry for drift detectors
drift_detector_registry = Registry(IDriftDetector)

# Register standard detectors
drift_detector_registry.register("distribution", DistributionDriftDetector)
drift_detector_registry.register("feature", FeatureStatisticsDriftDetector)
drift_detector_registry.register("confidence", ConfidenceDriftDetector)
drift_detector_registry.register("prediction", PredictionFrequencyDriftDetector)
drift_detector_registry.register("composite", CompositeDriftDetector)