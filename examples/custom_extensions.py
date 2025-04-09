#!/usr/bin/env python3
"""
Advanced example demonstrating TinyLCM extensibility.

This example shows how to:
1. Create custom anomaly detectors
2. Implement custom data storage strategies
3. Extend the core components with new functionality
4. Register custom callbacks for events
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Protocol, cast

import tinylcm
from tinylcm import ModelManager, InferenceMonitor, DataLogger
from tinylcm.core.inference_monitor import AnomalyDetector
from tinylcm.core.data_logger import DataStorageStrategy
from tinylcm.utils.metrics import MovingAverage


class PatternAnomalyDetector(AnomalyDetector):
    """Custom anomaly detector that looks for patterns in predictions."""
    
    def __init__(self, pattern_size: int = 3, threshold: float = 0.8):
        """
        Initialize the pattern anomaly detector.
        
        Args:
            pattern_size: Number of consecutive identical predictions to trigger
            threshold: Similarity threshold for patterns
        """
        self.pattern_size = pattern_size
        self.threshold = threshold
        self.recent_predictions: List[str] = []
    
    def check_for_anomalies(
        self, 
        record: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check for repetitive prediction patterns.
        
        Args:
            record: The inference record to check
            context: Additional context information
            
        Returns:
            Tuple[bool, List[str]]: (is_anomaly, reasons)
        """
        reasons = []
        
        if "prediction" in record:
            # Add to recent predictions
            self.recent_predictions.append(record["prediction"])
            
            # Keep only the most recent ones
            if len(self.recent_predictions) > self.pattern_size * 2:
                self.recent_predictions = self.recent_predictions[-self.pattern_size*2:]
            
            # Check for patterns if we have enough history
            if len(self.recent_predictions) >= self.pattern_size:
                # Get the last N predictions
                recent = self.recent_predictions[-self.pattern_size:]
                
                # Check if they're all the same
                if len(set(recent)) == 1:
                    reasons.append(
                        f"Repetitive prediction pattern detected: "
                        f"'{recent[0]}' repeated {self.pattern_size} times"
                    )
        
        return bool(reasons), reasons


class EncryptedTextStorage(DataStorageStrategy):
    """Custom storage strategy that 'encrypts' text data (for demonstration)."""
    
    def __init__(self, encryption_key: str = "tinylcm"):
        """
        Initialize with encryption key.
        
        Args:
            encryption_key: Key for 'encryption' (simple XOR for demonstration)
        """
        self.encryption_key = encryption_key
    
    def store(
        self, 
        data: str, 
        data_type: str, 
        entry_id: str, 
        storage_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store 'encrypted' text data.
        
        Args:
            data: Text data to store
            data_type: Type of data (should be "text")
            entry_id: Unique entry identifier
            storage_dir: Base storage directory
            metadata: Additional metadata
            
        Returns:
            str: Relative path to the stored text file
        """
        if not isinstance(data, str):
            raise TypeError(f"Expected string data for text storage, got {type(data)}")
        
        # "Encrypt" the data (simple XOR for demonstration)
        encrypted_data = self._encrypt(data)
        
        # Create relative path
        relative_path = f"encrypted/{entry_id}.enc"
        full_path = Path(storage_dir) / relative_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write encrypted text to file
        with open(full_path, "wb") as f:
            f.write(encrypted_data)
        
        return relative_path
    
    def load(
        self, 
        file_path: Union[str, Path]
    ) -> str:
        """
        Load and decrypt text data from file.
        
        Args:
            file_path: Path to the encrypted file
            
        Returns:
            str: Decrypted text data
        """
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Encrypted file not found: {file_path}")
        
        with open(path_obj, "rb") as f:
            encrypted_data = f.read()
        
        # Decrypt the data
        return self._decrypt(encrypted_data)
    
    def _encrypt(self, text: str) -> bytes:
        """Simple XOR 'encryption' for demonstration."""
        key_bytes = self.encryption_key.encode('utf-8')
        text_bytes = text.encode('utf-8')
        
        # XOR each byte with the corresponding byte from the key
        encrypted = bytearray()
        for i, byte in enumerate(text_bytes):
            key_byte = key_bytes[i % len(key_bytes)]
            encrypted.append(byte ^ key_byte)
        
        return bytes(encrypted)
    
    def _decrypt(self, encrypted_data: bytes) -> str:
        """Simple XOR 'decryption' for demonstration."""
        # Decryption is the same as encryption with XOR
        decrypted_bytes = self._encrypt(encrypted_data.decode('latin1')).decode('latin1')
        return decrypted_bytes.encode('latin1').decode('utf-8')


@dataclass
class ModelMetrics:
    """Tracking class for model performance metrics over time."""
    
    model_id: str
    accuracy_window: MovingAverage = field(default_factory=lambda: MovingAverage(window_size=100))
    latency_window: MovingAverage = field(default_factory=lambda: MovingAverage(window_size=100))
    confidence_window: MovingAverage = field(default_factory=lambda: MovingAverage(window_size=100))
    inference_count: int = 0
    last_update: float = field(default_factory=time.time)
    
    def update(
        self, 
        accuracy: Optional[float] = None, 
        latency: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> None:
        """Update metrics with new values."""
        if accuracy is not None:
            self.accuracy_window.add(accuracy)
        
        if latency is not None:
            self.latency_window.add(latency)
        
        if confidence is not None:
            self.confidence_window.add(confidence)
        
        self.inference_count += 1
        self.last_update = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            "model_id": self.model_id,
            "inference_count": self.inference_count,
            "accuracy": self.accuracy_window.average(),
            "latency": self.latency_window.average(),
            "confidence": self.confidence_window.average(),
            "last_update": datetime.fromtimestamp(self.last_update).isoformat()
        }


class ModelMonitoringSystem:
    """
    Custom extension that combines TinyLCM components to create a comprehensive monitoring system.
    
    This demonstrates how TinyLCM's modular components can be composed into higher-level systems.
    """
    
    def __init__(self, storage_dir: Union[str, Path]):
        """
        Initialize the monitoring system.
        
        Args:
            storage_dir: Base directory for all data
        """
        self.storage_dir = Path(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize TinyLCM components
        self.model_manager = ModelManager(storage_dir=self.storage_dir / "models")
        self.inference_monitor = InferenceMonitor(storage_dir=self.storage_dir / "inference")
        self.data_logger = DataLogger(storage_dir=self.storage_dir / "data")
        
        # Add our custom anomaly detector
        self.pattern_detector = PatternAnomalyDetector(pattern_size=3)
        self.inference_monitor.anomaly_detector.detectors.append(self.pattern_detector)
        
        # Register callbacks
        self.inference_monitor.register_anomaly_callback(self._on_anomaly)
        
        # Initialize model metrics tracking
        self.model_metrics: Dict[str, ModelMetrics] = {}
        
        # Set up performance thresholds
        self.min_acceptable_accuracy = 0.7
        self.max_acceptable_latency = 50.0  # ms
        
        print(f"Model Monitoring System initialized at {self.storage_dir}")
    
    def register_model(
        self, 
        model_path: Union[str, Path],
        model_format: str,
        description: str = "",
        version: Optional[str] = None,
        set_active: bool = False
    ) -> str:
        """
        Register a model with the system.
        
        Args:
            model_path: Path to the model file
            model_format: Format of the model
            description: Model description
            version: Model version
            set_active: Whether to set as active
            
        Returns:
            str: Model ID
        """
        model_id = self.model_manager.save_model(
            model_path=model_path,
            model_format=model_format,
            version=version,
            description=description,
            set_active=set_active
        )
        
        # Initialize metrics tracking for this model
        self.model_metrics[model_id] = ModelMetrics(model_id=model_id)
        
        print(f"Registered model {model_id}: {description}")
        return model_id
    
    def track_inference(
        self,
        input_data: Any,
        input_type: str,
        prediction: str,
        confidence: float,
        latency_ms: float,
        ground_truth: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track a model inference with all components.
        
        Args:
            input_data: Input data
            input_type: Type of input data
            prediction: Model prediction
            confidence: Prediction confidence
            latency_ms: Inference latency
            ground_truth: Correct label (if known)
            model_id: ID of the model used (defaults to active)
            
        Returns:
            Dict[str, Any]: Combined tracking information
        """
        # Get the model ID if not provided
        if model_id is None:
            try:
                active_model = self.model_manager.get_model_metadata()
                model_id = active_model["model_id"]
            except Exception:
                model_id = "unknown"
        
        # Generate a unique ID for this inference
        input_id = f"inf_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Log the input data
        if input_type == "text" and isinstance(input_data, str):
            # Use our custom encrypted storage for text
            from tinylcm.core.data_logger import DataStorageFactory
            original_create_storage = DataStorageFactory.create_storage
            
            # Monkey patch the factory to use our custom storage
            def patched_create_storage(data_type, **kwargs):
                if data_type == "text":
                    return EncryptedTextStorage()
                return original_create_storage(data_type, **kwargs)
            
            DataStorageFactory.create_storage = patched_create_storage
            
            try:
                entry_id = self.data_logger.log_data(
                    input_data=input_data,
                    input_type=input_type,
                    prediction=prediction,
                    confidence=confidence,
                    label=ground_truth,
                    metadata={"model_id": model_id, "latency_ms": latency_ms}
                )
            finally:
                # Restore original method
                DataStorageFactory.create_storage = original_create_storage
        else:
            # Use standard storage for other types
            entry_id = self.data_logger.log_data(
                input_data=input_data,
                input_type=input_type,
                prediction=prediction,
                confidence=confidence,
                label=ground_truth,
                metadata={"model_id": model_id, "latency_ms": latency_ms}
            )
        
        # Track with inference monitor
        monitor_record = self.inference_monitor.track_inference(
            input_id=input_id,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            ground_truth=ground_truth
        )
        
        # Update model metrics
        if model_id in self.model_metrics:
            accuracy = 1.0 if ground_truth is None or prediction == ground_truth else 0.0
            self.model_metrics[model_id].update(
                accuracy=accuracy,
                latency=latency_ms,
                confidence=confidence
            )
            
            # Check if we need to update model metadata with latest metrics
            metrics = self.model_metrics[model_id]
            if metrics.inference_count % 10 == 0:  # Update every 10 inferences
                self.model_manager.update_metrics(model_id, {
                    "accuracy": metrics.accuracy_window.average(),
                    "avg_latency_ms": metrics.latency_window.average(),
                    "avg_confidence": metrics.confidence_window.average()
                })
        
        return {
            "input_id": input_id,
            "entry_id": entry_id,
            "monitor_record": monitor_record
        }
    
    def _on_anomaly(self, record: Dict[str, Any]) -> None:
        """Handle anomaly detection events."""
        print("\n==== ANOMALY DETECTED ====")
        print(f"Input ID: {record['input_id']}")
        print(f"Prediction: {record['prediction']}, Confidence: {record['confidence']}")
        print(f"Reasons: {', '.join(record.get('anomaly_reasons', []))}")
        
        # Check if we need to take action
        if "ground_truth" in record and record["ground_truth"] is not None:
            correct = record["prediction"] == record["ground_truth"]
            if not correct:
                print("ACTION: Incorrect prediction - adding to retraining dataset")
                # In a real system, would save this for retraining
        
        print("===========================\n")
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report on model performance."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        # Add metrics for each model
        for model_id, metrics in self.model_metrics.items():
            report["models"][model_id] = metrics.get_summary()
        
        # Add overall system metrics
        system_metrics = self.inference_monitor.get_current_metrics()
        report["system"] = {
            "total_inferences": system_metrics.get("total_inferences", 0),
            "prediction_distribution": system_metrics.get("prediction_distribution", {}),
            "latency": system_metrics.get("latency", {}),
            "confidence": system_metrics.get("confidence", {}),
        }
        
        return report
    
    def close(self) -> None:
        """Clean up resources."""
        self.inference_monitor.close()
        self.data_logger.close()
        
        # Save final report
        report = self.get_model_performance_report()
        report_path = self.storage_dir / "final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Final performance report saved to {report_path}")


def main() -> None:
    """Run the extensions example."""
    print("TinyLCM Extensions Example")
    print(f"Version: {tinylcm.__version__}")
    print("=" * 50)
    
    # Create working directory
    work_dir = Path("tinylcm_extension_example")
    
    # Create our custom monitoring system
    monitor = ModelMonitoringSystem(storage_dir=work_dir)
    
    # Create a mock model
    mock_model_path = work_dir / "advanced_model.json"
    mock_model = {
        "name": "AdvancedClassifier",
        "version": "2.0.0",
        "layers": [20, 10, 5, 4],
        "classes": ["apple", "banana", "orange", "grape"],
        "weights": [random.random() for _ in range(50)]
    }
    
    os.makedirs(os.path.dirname(mock_model_path), exist_ok=True)
    with open(mock_model_path, "w") as f:
        json.dump(mock_model, f, indent=2)
    
    # Register the model
    model_id = monitor.register_model(
        model_path=mock_model_path,
        model_format="json",
        description="Advanced fruit classifier",
        version="v2",
        set_active=True
    )
    
    # Simulate inferences with different input types
    print("\nSimulating inferences with different input types...")
    
    # Text inputs
    for i in range(10):
        input_text = f"This is a sample text about a {random.choice(mock_model['classes'])}."
        prediction = random.choice(mock_model['classes'])
        confidence = random.uniform(0.7, 0.98)
        latency = random.uniform(5, 20)
        
        # Generate pattern anomaly for demonstration
        if 6 <= i <= 8:
            prediction = "banana"  # Force same prediction to trigger pattern detector
        
        monitor.track_inference(
            input_data=input_text,
            input_type="text",
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency,
            ground_truth=random.choice(mock_model['classes']) if random.random() > 0.8 else prediction
        )
        
        print(f"  Processed text input {i+1}/10: {prediction} ({confidence:.2f})")
        time.sleep(0.1)
    
    # JSON inputs
    for i in range(5):
        input_json = {
            "feature1": random.random(),
            "feature2": random.random(),
            "feature3": random.random(),
            "metadata": {
                "source": "sensor_array",
                "timestamp": time.time()
            }
        }
        
        prediction = random.choice(mock_model['classes'])
        confidence = random.uniform(0.6, 0.95)
        latency = random.uniform(10, 30)
        
        monitor.track_inference(
            input_data=input_json,
            input_type="json",
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency
        )
        
        print(f"  Processed JSON input {i+1}/5: {prediction} ({confidence:.2f})")
        time.sleep(0.1)
    
    # Generate performance report
    report = monitor.get_model_performance_report()
    print("\nModel Performance Report:")
    print(f"  Total models: {len(report['models'])}")
    
    for model_id, metrics in report["models"].items():
        print(f"  Model {model_id}:")
        print(f"    Inferences: {metrics['inference_count']}")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Avg Latency: {metrics['latency']:.2f}ms")
        print(f"    Avg Confidence: {metrics['confidence']:.4f}")
    
    # Clean up
    monitor.close()
    
    print("\nExtensions example completed successfully.")


if __name__ == "__main__":
    main()