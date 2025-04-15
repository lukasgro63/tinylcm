#!/usr/bin/env python3
"""
Synchronization example for TinyLCM.

This example demonstrates the usage of the SyncInterface component
to prepare data packages for synchronization between an edge device
and a central server.
"""

import json
import os
import time
from pathlib import Path

import tinylcm
from tinylcm import (
    ModelManager,
    InferenceMonitor,
    DataLogger,
    SyncInterface
)


def create_sample_data(base_dir: Path) -> None:
    """
    Create sample data files for demonstration.
    
    Args:
        base_dir: Base directory for sample data
    """
    # Create directories
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a sample model file
    model_file = models_dir / "sample_model.json"
    model_data = {
        "name": "SampleModel",
        "version": "1.0",
        "layers": [10, 5, 2],
        "weights": [0.1, 0.2, 0.3, 0.4, 0.5],
        "classes": ["class_a", "class_b"]
    }
    
    with open(model_file, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)
    
    # Create a sample metrics file
    metrics_file = data_dir / "metrics.json"
    metrics_data = {
        "accuracy": 0.92,
        "precision": 0.88,
        "recall": 0.90,
        "f1": 0.89,
        "latency_ms": {
            "mean": 15.3,
            "median": 12.8,
            "min": 8.2,
            "max": 45.7
        }
    }
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    
    # Create a sample log file
    log_file = data_dir / "inferences.jsonl"
    log_entries = []
    
    for i in range(10):
        entry = {
            "timestamp": time.time() + i,
            "input_id": f"input_{i}",
            "prediction": "class_a" if i % 2 == 0 else "class_b",
            "confidence": 0.7 + (i % 3) * 0.1,
            "latency_ms": 10 + i * 2
        }
        log_entries.append(entry)
    
    with open(log_file, "w", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Created sample data in {base_dir}")
    print(f"  - Model file: {model_file}")
    print(f"  - Metrics file: {metrics_file}")
    print(f"  - Log file: {log_file}")


def basic_sync_package_example(work_dir: Path) -> None:
    """
    Demonstrate basic package creation and management.
    
    Args:
        work_dir: Working directory
    """
    print("\n=== Basic Sync Package Example ===")
    
    # Create sample data
    create_sample_data(work_dir)
    
    # Initialize sync interface
    sync_dir = work_dir / "sync"
    sync_interface = SyncInterface(sync_dir=sync_dir)
    
    # Create a sync package
    print("\nCreating sync package...")
    package_id = sync_interface.create_package(
        device_id="edge_device_1",
        package_type="mixed",
        description="Sample package with model and data",
        compression="gzip"
    )
    
    print(f"Created package with ID: {package_id}")
    
    # Add files to the package
    print("\nAdding files to package...")
    
    # Add model file
    model_file = work_dir / "models" / "sample_model.json"
    sync_interface.add_file_to_package(
        package_id=package_id,
        file_path=model_file,
        file_type="model",
        metadata={"model_type": "classifier", "framework": "custom"}
    )
    print(f"Added model file: {model_file}")
    
    # Add metrics file
    metrics_file = work_dir / "data" / "metrics.json"
    sync_interface.add_file_to_package(
        package_id=package_id,
        file_path=metrics_file,
        file_type="metrics"
    )
    print(f"Added metrics file: {metrics_file}")
    
    # Add log file
    log_file = work_dir / "data" / "inferences.jsonl"
    sync_interface.add_file_to_package(
        package_id=package_id,
        file_path=log_file,
        file_type="logs"
    )
    print(f"Added log file: {log_file}")
    
    # Finalize the package
    print("\nFinalizing package...")
    package_path = sync_interface.finalize_package(package_id)
    print(f"Package finalized: {package_path}")
    
    # List packages
    print("\nListing available packages:")
    packages = sync_interface.list_packages()
    for pkg in packages:
        print(f"  - {pkg['package_id']}: {pkg['package_type']} package with {len(pkg['files'])} files")
    
    # Mark as synced (simulating successful server sync)
    print("\nMarking package as synced with server...")
    sync_interface.mark_as_synced(
        package_id=package_id,
        sync_time=time.time(),
        server_id="central_server_1",
        status="success"
    )
    print(f"Package {package_id} marked as successfully synced")
    
    # List including synced packages
    print("\nListing all packages (including synced):")
    all_packages = sync_interface.list_packages(include_synced=True)
    for pkg in all_packages:
        status = pkg.get('sync_status', 'pending')
        print(f"  - {pkg['package_id']}: {pkg['package_type']} package, status: {status}")


def component_integration_example(work_dir: Path) -> None:
    """
    Demonstrate integration with other TinyLCM components.
    
    Args:
        work_dir: Working directory
    """
    print("\n=== Component Integration Example ===")
    
    # Initialize components
    print("\nInitializing TinyLCM components...")
    model_dir = work_dir / "model_manager"
    monitor_dir = work_dir / "monitor"
    data_dir = work_dir / "data_logger"
    sync_dir = work_dir / "sync"
    
    model_manager = ModelManager(storage_dir=model_dir)
    inference_monitor = InferenceMonitor(storage_dir=monitor_dir)
    data_logger = DataLogger(storage_dir=data_dir)
    sync_interface = SyncInterface(sync_dir=sync_dir)
    
    # Create and save a model
    print("\nSaving a model...")
    model_file = work_dir / "models" / "sample_model.json"
    model_id = model_manager.save_model(
        model_path=model_file,
        model_format="json",
        version="v1",
        description="Sample model for sync demo",
        set_active=True
    )
    print(f"Model saved with ID: {model_id}")
    
    # Log some inferences
    print("\nLogging inferences...")
    for i in range(5):
        # Log input data
        input_data = f"Input data {i}"
        input_id = data_logger.log_data(
            input_data=input_data,
            input_type="text"
        )
        
        # Track inference
        inference_monitor.track_inference(
            input_id=f"test_{i}",
            prediction=f"class_{i % 2}",
            confidence=0.8 + (i % 3) * 0.05,
            latency_ms=12.5 + i
        )
        print(f"Logged inference {i+1}/5")
    
    # Create package from components
    print("\nCreating sync package from components...")
    package_id = sync_interface.create_package_from_components(
        device_id="edge_device_1",
        model_manager=model_manager,
        inference_monitor=inference_monitor,
        data_logger=data_logger,
        compression="gzip"
    )
    
    # Finalize package
    package_path = sync_interface.finalize_package(package_id)
    print(f"Component package finalized: {package_path}")
    
    # List package details
    packages = sync_interface.list_packages()
    for pkg in packages:
        print(f"\nPackage details:")
        print(f"  - ID: {pkg['package_id']}")
        print(f"  - Type: {pkg['package_type']}")
        print(f"  - Device: {pkg['device_id']}")
        print(f"  - Created: {time.ctime(pkg['creation_time'])}")
        print(f"  - Files:")
        
        for file_info in pkg['files']:
            print(f"    - {file_info['file_type']}: {file_info['package_path']} ({file_info['size_bytes']} bytes)")


def main() -> None:
    """Run the sync examples."""
    print("TinyLCM Sync Example")
    print(f"Version: {tinylcm.__version__}")
    print("=" * 50)
    
    # Create a working directory
    work_dir = Path("tinylcm_sync_example")
    os.makedirs(work_dir, exist_ok=True)
    
    # Run the examples
    basic_sync_package_example(work_dir)
    component_integration_example(work_dir)
    
    print("\nExample completed.")


if __name__ == "__main__":
    main()