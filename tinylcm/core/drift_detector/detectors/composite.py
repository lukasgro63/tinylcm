"""Composite drift detector implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

from tinylcm.core.drift_detector.base import DriftDetector
from tinylcm.utils.config import Config


class CompositeDriftDetector(DriftDetector):
    """
    Composite drift detector that combines multiple detectors.
    
    Allows using multiple drift detection strategies together.
    """
    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        detectors: Optional[List[DriftDetector]] = None,
        window_size: int = 100,  # Add window_size parameter
        threshold: float = 0.1,  # Add threshold parameter
        config: Optional[Config] = None
    ):
        """
        Initialize the composite drift detector.
        
        Args:
            storage_dir: Directory for storing reference data and state
            detectors: List of drift detectors to use
            window_size: Size of the sliding window for current data
            threshold: Threshold for detecting drift
            config: Configuration object
        """
        # Pass window_size and threshold to the parent class
        super().__init__(
            storage_dir=storage_dir, 
            window_size=window_size,
            threshold=threshold,
            config=config
        )
        
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
    
    def update(self, record: Dict[str, Any], auto_check: bool = True) -> bool:
        """
        Update all detectors with a new record.
        
        Args:
            record: New data record
            auto_check: Whether to automatically check for drift
            
        Returns:
            bool: True if drift was detected by any detector
        """
        # Add record to own current window
        self.current_window.append(record)
        
        # Update each detector with the record
        for detector in self.detectors:
            detector.current_window.append(record)
        
        # Return False if we don't have a reference distribution yet
        if self.reference_distribution is None:
            return False
        
        # Return False if we don't have enough data yet
        if len(self.current_window) < self.window_size:
            return False
        
        # Check for drift if auto_check is enabled and it's been a while since last check
        if auto_check:
            current_time = time.time()
            check_interval = self.config.get("drift_detector", "check_interval", 60.0)
            
            if current_time - self.last_check_time > check_interval:
                drift_result = self.check_for_drift()
                self.last_check_time = current_time
                return bool(drift_result.get("drift_detected", False))
        
        return False
    
    def check_for_drift(self) -> Dict[str, Any]:
        # Stelle sicher, dass alle Detektoren dieselben Daten haben
        for detector in self.detectors:
            detector.current_window.clear()
            for record in self.current_window:
                detector.current_window.append(record)
                
        # Beziehe Ergebnisse von allen Detektoren
        detector_results = []
        
        for detector in self.detectors:
            detector_result = detector.check_for_drift()
            detector_results.append(detector_result)
        
        # Bestimme, ob irgendein Detektor Drift erkannt hat
        any_drift = bool(any(bool(result.get("drift_detected", False)) for result in detector_results))
        # Create composite result
        result = {
            "drift_detected": any_drift,
            "drift_type": "composite",
            "timestamp": time.time(),
            "session_id": self.session_id,
            "detector_results": detector_results,
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