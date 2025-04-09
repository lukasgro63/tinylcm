#!/usr/bin/env python3
"""
Basic usage example for TinyLCM components.

This example demonstrates the core functionality of TinyLCM:
1. Model management (saving, loading, versioning)
2. Inference monitoring
3. Data logging

It simulates a simple image classification scenario with a mock model.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import tinylcm
from tinylcm import ModelManager, InferenceMonitor, DataLogger


def create_mock_model(save_path: str) -> None:
    """Create a mock model file for demonstration."""
    mock_model = {
        "name": "MockClassifier",
        "version": "1.0.0",
        "layers": [10, 5, 3],
        "classes": ["cat", "dog", "bird"],
        "weights": [random.random() for _ in range(20)]
    }
    
    with open(save_path, "w") as f:
        json.dump(mock_model, f, indent=2)
    
    print(f"Created mock model at {save_path}")


def mock_inference(model: Dict[str, Any], input_id: str) -> Tuple[str, float, float]:
    """
    Simulate model inference with random results.
    
    Args:
        model: Mock model dictionary
        input_id: Identifier for the input
        
    Returns:
        Tuple[str, float, float]: (prediction, confidence, latency_ms)
    """
    # Simulate processing time
    start_time = time.time()
    time.sleep(random.uniform(0.01, 0.05))
    
    # Generate random prediction
    classes = model["classes"]
    prediction = random.choice(classes)
    
    # Generate confidence (biased toward high values for the example)
    confidence = random.uniform(0.7, 1.0)
    
    # Calculate actual latency
    latency_ms = (time.time() - start_time) * 1000
    
    return prediction, confidence, latency_ms


def mock_image_data() -> bytes:
    """Generate mock image data for demonstration."""
    # This is just a placeholder - in a real scenario, you'd have actual image bytes
    return b"MOCK_IMAGE_DATA_" + str(random.randint(1000, 9999)).encode()


def anomaly_callback(record: Dict[str, Any]) -> None:
    """Callback function for anomaly detection."""
    print("\n*** ANOMALY DETECTED ***")
    print(f"Input ID: {record['input_id']}")
    print(f"Prediction: {record['prediction']}, Confidence: {record['confidence']}")
    print(f"Reasons: {', '.join(record.get('anomaly_reasons', []))}")
    print("*************************\n")


def prepare_workspace(work_dir: Path) -> None:
    """
    Prepare a clean workspace for the example.
    
    Args:
        work_dir: Path to the workspace directory
    """
    # Import shutil to handle directory operations
    import shutil
    
    # Create main directory if it doesn't exist
    os.makedirs(work_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ["models", "inference", "data"]
    for subdir in subdirs:
        subdir_path = work_dir / subdir
        
        # Clean up if it exists already
        if subdir_path.exists():
            # Handle symlinks first
            active_link = subdir_path / "active_model"
            if active_link.exists():
                if active_link.is_symlink():
                    os.unlink(active_link)
                elif active_link.is_file():
                    os.remove(active_link)
            
            # Clean out files but don't remove directory structure
            for item in subdir_path.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        os.remove(item)
                except Exception as e:
                    print(f"Warning: Could not remove {item}: {e}")
        else:
            # Create if it doesn't exist
            os.makedirs(subdir_path, exist_ok=True)
    
    print(f"Prepared workspace at {work_dir}")


def main() -> None:
    """Run the basic example."""
    print("TinyLCM Basic Example")
    print(f"Version: {tinylcm.__version__}")
    print("=" * 50)
    
    # Create a working directory for the example
    work_dir = Path("tinylcm_example")
    prepare_workspace(work_dir)
    
    # Create and initialize components
    model_manager = ModelManager(storage_dir=work_dir / "models")
    inference_monitor = InferenceMonitor(storage_dir=work_dir / "inference")
    data_logger = DataLogger(storage_dir=work_dir / "data")
    
    # Register anomaly callback
    inference_monitor.register_anomaly_callback(anomaly_callback)
    
    # Create a mock model
    mock_model_path = work_dir / "mock_model.json"
    create_mock_model(mock_model_path)
    
    # Save the model with the model manager
    model_id = model_manager.save_model(
        model_path=mock_model_path,
        model_format="json",
        version="v1",
        description="Mock image classifier",
        tags=["example", "mock"],
        metrics={"accuracy": 0.85, "f1": 0.84},
        set_active=True
    )
    
    print(f"\nSaved model with ID: {model_id}")
    model_metadata = model_manager.get_model_metadata(model_id)
    print(f"Model is now active: {model_metadata['is_active']}")
    
    # Load the model for inference
    try:
        model_path = model_manager.load_model()
        with open(model_path, "r") as f:
            model = json.load(f)
        
        print(f"\nLoaded model: {model['name']} v{model['version']}")
        print(f"Model classes: {', '.join(model['classes'])}")
    except Exception as e:
        print(f"\nError loading active model: {e}")
        print("Falling back to loading model by ID...")
        
        # Alternative approach: use the model ID directly
        model_path = model_manager.load_model(model_id)
        with open(model_path, "r") as f:
            model = json.load(f)
        
        print(f"Loaded model by ID: {model['name']} v{model['version']}")
        print(f"Model classes: {', '.join(model['classes'])}")
    
    # Simulate inference loop
    print("\nRunning inference simulation...")
    
    try:
        for i in range(20):
            # Generate a unique input ID
            input_id = f"img_{i:04d}"
            
            # Generate mock image data
            image_data = mock_image_data()
            
            # Log the input data
            entry_id = data_logger.log_image(
                image_data=image_data,
                metadata={"source": "camera_1", "sequence": i}
            )
            
            # Simulate model inference
            prediction, confidence, latency_ms = mock_inference(model, input_id)
            
            # Occasionally inject an anomaly for demonstration
            if i == 15:
                confidence = 0.2  # Low confidence anomaly
            if i == 18:
                latency_ms = 200  # High latency anomaly
            
            # Track the inference
            inference_monitor.track_inference(
                input_id=input_id,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                # Ground truth would come from a human or another source in practice
                ground_truth=prediction if random.random() > 0.2 else random.choice(model["classes"])
            )
            
            # Update the data log with the prediction
            data_logger.log_prediction(
                input_id=entry_id,
                prediction=prediction,
                confidence=confidence
            )
            
            # Print progress
            print(f"  [{i+1}/20] Processed {input_id}: {prediction} (confidence: {confidence:.4f}, latency: {latency_ms:.2f}ms)")
            
            # Slow down the example for readability
            time.sleep(0.1)
        
        # Get and print metrics
        metrics = inference_monitor.get_current_metrics()
        prediction_distribution = metrics.get("prediction_distribution", {})
        latency = metrics.get("latency", {})
        confidence = metrics.get("confidence", {})
        
        print("\nInference Metrics:")
        print(f"  Total inferences: {metrics['total_inferences']}")
        print("  Prediction distribution:")
        for cls, count in prediction_distribution.items():
            print(f"    {cls}: {count}")
        
        print("  Latency (ms):")
        print(f"    Min: {latency.get('min_ms', 0):.2f}, Max: {latency.get('max_ms', 0):.2f}, Avg: {latency.get('mean_ms', 0):.2f}")
        
        print("  Confidence:")
        print(f"    Min: {confidence.get('min', 0):.4f}, Max: {confidence.get('max', 0):.4f}, Avg: {confidence.get('mean', 0):.4f}")
        
        if "accuracy" in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        # Export metrics
        metrics_path = inference_monitor.export_metrics(format="json")
        print(f"\nExported metrics to: {metrics_path}")
        
        # Export data log
        csv_path = data_logger.export_to_csv()
        print(f"Exported data log to: {csv_path}")
        
        # List models
        models = model_manager.list_models()
        print(f"\nAvailable models: {len(models)}")
        for model_meta in models:
            print(f"  {model_meta['model_id']}: {model_meta.get('description', 'No description')} ({model_meta.get('version', 'unknown')})")
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
    finally:
        # Always clean up resources
        print("\nCleaning up resources...")
        inference_monitor.close()
        data_logger.close()
    
    print("\nExample completed successfully.")

if __name__ == "__main__":
    main()