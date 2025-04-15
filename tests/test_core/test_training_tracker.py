"""Tests for TrainingTracker component."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from tinylcm.core.training_tracker import TrainingTracker


class TestTrainingTracker:
    """Test TrainingTracker functionality."""

    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = TrainingTracker(storage_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up resources and temporary directory."""
        self.tracker.close()
        shutil.rmtree(self.temp_dir)

    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        assert os.path.exists(self.temp_dir)
        assert os.path.exists(os.path.join(self.temp_dir, "runs"))
        assert os.path.exists(os.path.join(self.temp_dir, "artifacts"))

    def test_start_run(self):
        """Test starting a training run."""
        # Start a run
        run_id = self.tracker.start_run(
            run_name="test_run",
            description="Test training run",
            tags={"test": "true", "env": "unittest"}
        )

        # Check that run_id is a string
        assert isinstance(run_id, str)
        
        # Check that run metadata was created
        run_dir = os.path.join(self.temp_dir, "runs", run_id)
        assert os.path.exists(run_dir)
        
        # Check metadata file
        metadata_path = os.path.join(run_dir, "metadata.json")
        assert os.path.exists(metadata_path)
        
        # Check metadata content
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        assert metadata["run_id"] == run_id
        assert metadata["run_name"] == "test_run"
        assert metadata["description"] == "Test training run"
        assert metadata["tags"]["test"] == "true"
        assert metadata["status"] == "RUNNING"
        assert "start_time" in metadata

    def test_end_run(self):
        """Test ending a training run."""
        # Start a run
        run_id = self.tracker.start_run(run_name="test_run")
        
        # End the run
        result = self.tracker.end_run(status="COMPLETED")
        
        # Check result
        assert result is True
        
        # Check metadata was updated
        run_dir = os.path.join(self.temp_dir, "runs", run_id)
        metadata_path = os.path.join(run_dir, "metadata.json")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        assert metadata["status"] == "COMPLETED"
        assert "end_time" in metadata
        assert metadata["end_time"] > metadata["start_time"]

    def test_end_run_with_specific_run_id(self):
        """Test ending a run with specific run_id."""
        # Start two runs
        run_id1 = self.tracker.start_run(run_name="run1")
        self.tracker.start_run(run_name="run2")  # This makes run2 the active run
        
        # End the first run
        result = self.tracker.end_run(run_id=run_id1, status="COMPLETED")
        
        # Check result
        assert result is True
        
        # Check first run metadata
        run_dir1 = os.path.join(self.temp_dir, "runs", run_id1)
        metadata_path1 = os.path.join(run_dir1, "metadata.json")
        
        with open(metadata_path1, "r") as f:
            metadata1 = json.load(f)
        
        assert metadata1["status"] == "COMPLETED"
        assert "end_time" in metadata1

    def test_log_param(self):
        """Test logging parameters."""
        # Start a run
        run_id = self.tracker.start_run(run_name="test_run")
        
        # Log some parameters
        self.tracker.log_param("learning_rate", 0.01)
        self.tracker.log_param("batch_size", 32)
        self.tracker.log_param("epochs", 100)
        
        # Check params file
        params_path = os.path.join(self.temp_dir, "runs", run_id, "params.json")
        assert os.path.exists(params_path)
        
        # Check params content
        with open(params_path, "r") as f:
            params = json.load(f)
        
        assert params["learning_rate"] == 0.01
        assert params["batch_size"] == 32
        assert params["epochs"] == 100

    def test_log_params(self):
        """Test logging multiple parameters at once."""
        # Start a run
        run_id = self.tracker.start_run(run_name="test_run")
        
        # Log multiple parameters
        params = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam"
        }
        self.tracker.log_params(params)
        
        # Check params file
        params_path = os.path.join(self.temp_dir, "runs", run_id, "params.json")
        
        with open(params_path, "r") as f:
            saved_params = json.load(f)
        
        assert saved_params["learning_rate"] == 0.01
        assert saved_params["batch_size"] == 32
        assert saved_params["epochs"] == 100
        assert saved_params["optimizer"] == "adam"

    def test_log_metric(self):
        """Test logging metrics."""
        # Start a run
        run_id = self.tracker.start_run(run_name="test_run")
        
        # Log some metrics
        self.tracker.log_metric("accuracy", 0.85)
        self.tracker.log_metric("loss", 0.23)
        
        # Log a metric with a step
        self.tracker.log_metric("val_accuracy", 0.82, step=1)
        
        # Check metrics file
        metrics_path = os.path.join(self.temp_dir, "runs", run_id, "metrics.json")
        assert os.path.exists(metrics_path)
        
        # Check metrics content
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        assert "accuracy" in metrics
        assert metrics["accuracy"][0]["value"] == 0.85
        assert "timestamp" in metrics["accuracy"][0]
        
        assert "loss" in metrics
        assert metrics["loss"][0]["value"] == 0.23
        
        assert "val_accuracy" in metrics
        assert metrics["val_accuracy"][0]["value"] == 0.82
        assert metrics["val_accuracy"][0]["step"] == 1

    def test_log_metrics(self):
        """Test logging multiple metrics at once."""
        # Start a run
        run_id = self.tracker.start_run(run_name="test_run")
        
        # Log multiple metrics
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1": 0.80
        }
        self.tracker.log_metrics(metrics, step=5)
        
        # Check metrics file
        metrics_path = os.path.join(self.temp_dir, "runs", run_id, "metrics.json")
        
        with open(metrics_path, "r") as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics["accuracy"][0]["value"] == 0.85
        assert saved_metrics["precision"][0]["value"] == 0.82
        assert saved_metrics["f1"][0]["value"] == 0.80
        assert saved_metrics["accuracy"][0]["step"] == 5

    def test_log_artifact(self):
        """Test logging artifacts."""
        # Start a run
        run_id = self.tracker.start_run(run_name="test_run")
        
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test_artifact.txt")
        with open(test_file, "w") as f:
            f.write("This is a test artifact")
        
        # Log the artifact
        artifact_path = self.tracker.log_artifact(test_file)
        
        # Check artifact path
        assert os.path.exists(artifact_path)
        
        # Check artifact content
        with open(artifact_path, "r") as f:
            content = f.read()
        
        assert content == "This is a test artifact"
        
        # Check artifacts list
        artifacts_path = os.path.join(self.temp_dir, "runs", run_id, "artifacts.json")
        assert os.path.exists(artifacts_path)
        
        with open(artifacts_path, "r") as f:
            artifacts = json.load(f)
        
        assert len(artifacts) == 1
        assert artifacts[0]["name"] == "test_artifact.txt"
        assert "path" in artifacts[0]
        assert "timestamp" in artifacts[0]

    def test_log_figure(self):
        """Test logging a figure."""
        # Start a run
        run_id = self.tracker.start_run(run_name="test_run")
        
        # Create a test figure (as bytes for the test)
        figure_data = b"MOCK_FIGURE_DATA"
        
        # Log the figure
        figure_path = self.tracker.log_figure(
            figure_data, 
            name="test_figure.png",
            description="Test figure"
        )
        
        # Check figure path
        assert os.path.exists(figure_path)
        
        # Check figure content
        with open(figure_path, "rb") as f:
            content = f.read()
        
        assert content == figure_data
        
        # Check artifacts list
        artifacts_path = os.path.join(self.temp_dir, "runs", run_id, "artifacts.json")
        
        with open(artifacts_path, "r") as f:
            artifacts = json.load(f)
        
        assert len(artifacts) == 1
        assert artifacts[0]["name"] == "test_figure.png"
        assert artifacts[0]["type"] == "figure"
        assert artifacts[0]["description"] == "Test figure"

    def test_log_model(self):
        """Test logging a model."""
        # Start a run
        run_id = self.tracker.start_run(run_name="test_run")
        
        # Create a test model file
        model_file = os.path.join(self.temp_dir, "test_model.json")
        model_data = {"weights": [1.0, 2.0, 3.0], "biases": [0.1, 0.2]}
        
        with open(model_file, "w") as f:
            json.dump(model_data, f)
        
        # Log the model
        model_dir = self.tracker.log_model(
            model_path=model_file,
            model_format="json",
            flavor="custom",
            custom_properties={"version": "1.0", "accuracy": 0.95}
        )
        
        # Check model directory
        assert os.path.exists(model_dir)
        assert os.path.exists(os.path.join(model_dir, "test_model.json"))
        
        # Check model metadata
        model_meta_path = os.path.join(model_dir, "model_info.json")
        assert os.path.exists(model_meta_path)
        
        with open(model_meta_path, "r") as f:
            model_meta = json.load(f)
        
        assert model_meta["format"] == "json"
        assert model_meta["flavor"] == "custom"
        assert model_meta["custom_properties"]["version"] == "1.0"
        assert model_meta["custom_properties"]["accuracy"] == 0.95
        
        # Check artifacts list
        artifacts_path = os.path.join(self.temp_dir, "runs", run_id, "artifacts.json")
        
        with open(artifacts_path, "r") as f:
            artifacts = json.load(f)
        
        assert len(artifacts) == 1
        assert artifacts[0]["type"] == "model"
        assert "path" in artifacts[0]

    def test_get_run_info(self):
        """Test getting run information."""
        # Start a run with parameters and metrics
        run_id = self.tracker.start_run(run_name="test_run")
        self.tracker.log_param("learning_rate", 0.01)
        self.tracker.log_metric("accuracy", 0.85)
        
        # Get run info
        run_info = self.tracker.get_run_info(run_id)
        
        # Check run info
        assert run_info["run_id"] == run_id
        assert run_info["run_name"] == "test_run"
        assert run_info["status"] == "RUNNING"
        assert "params" in run_info
        assert run_info["params"]["learning_rate"] == 0.01
        assert "metrics" in run_info
        assert run_info["metrics"]["accuracy"][0]["value"] == 0.85

    def test_list_runs(self):
        """Test listing runs."""
        # Create multiple runs
        self.tracker.start_run(run_name="run1", tags={"experiment": "A"})
        run1_id = self.tracker.active_run_id
        self.tracker.end_run()
        
        self.tracker.start_run(run_name="run2", tags={"experiment": "A"})
        run2_id = self.tracker.active_run_id
        self.tracker.end_run()
        
        self.tracker.start_run(run_name="run3", tags={"experiment": "B"})
        run3_id = self.tracker.active_run_id
        self.tracker.end_run()
        
        # List all runs
        all_runs = self.tracker.list_runs()
        
        # Check all runs are listed
        assert len(all_runs) == 3
        run_ids = [run["run_id"] for run in all_runs]
        assert run1_id in run_ids
        assert run2_id in run_ids
        assert run3_id in run_ids
        
        # List runs with filter
        exp_a_runs = self.tracker.list_runs(filter_func=lambda run: run["tags"].get("experiment") == "A")
        
        # Check filtered runs
        assert len(exp_a_runs) == 2
        exp_a_ids = [run["run_id"] for run in exp_a_runs]
        assert run1_id in exp_a_ids
        assert run2_id in exp_a_ids
        assert run3_id not in exp_a_ids

    def test_delete_run(self):
        """Test deleting a run."""
        # Create a run
        self.tracker.start_run(run_name="test_run")
        run_id = self.tracker.active_run_id
        self.tracker.end_run()
        
        # Check run exists
        run_dir = os.path.join(self.temp_dir, "runs", run_id)
        assert os.path.exists(run_dir)
        
        # Delete the run
        result = self.tracker.delete_run(run_id)
        
        # Check result
        assert result is True
        
        # Check run directory no longer exists
        assert not os.path.exists(run_dir)
        
        # Deleting again should return False
        result = self.tracker.delete_run(run_id)
        assert result is False

    def test_nested_runs(self):
        """Test nested runs."""
        # Start parent run
        parent_id = self.tracker.start_run(run_name="parent_run")
        
        # Start child run
        child_id = self.tracker.start_run(run_name="child_run", nested=True)
        
        # Check child run has parent reference
        child_info = self.tracker.get_run_info(child_id)
        assert child_info["parent_run_id"] == parent_id
        
        # End child run
        self.tracker.end_run()
        
        # Check active run is now parent again
        assert self.tracker.active_run_id == parent_id
        
        # End parent run
        self.tracker.end_run()
        
        # Check both runs are completed
        parent_info = self.tracker.get_run_info(parent_id)
        child_info = self.tracker.get_run_info(child_id)
        
        assert parent_info["status"] == "COMPLETED"
        assert child_info["status"] == "COMPLETED"

    def test_export_to_mlflow_format(self):
        """Test exporting runs to MLflow format."""
        # Start a run with params, metrics and artifacts
        run_id = self.tracker.start_run(run_name="test_run")
        self.tracker.log_param("learning_rate", 0.01)
        self.tracker.log_metric("accuracy", 0.85)
        
        # Create a test artifact
        test_file = os.path.join(self.temp_dir, "test_artifact.txt")
        with open(test_file, "w") as f:
            f.write("This is a test artifact")
        
        self.tracker.log_artifact(test_file)
        self.tracker.end_run()
        
        # Export run
        export_dir = os.path.join(self.temp_dir, "mlflow_export")
        result = self.tracker.export_to_mlflow_format(run_id, export_dir)
        
        # Check result
        assert result is True
        
        # Check MLflow directory structure
        mlflow_run_dir = os.path.join(export_dir, run_id)
        assert os.path.exists(mlflow_run_dir)
        assert os.path.exists(os.path.join(mlflow_run_dir, "meta.yaml"))
        assert os.path.exists(os.path.join(mlflow_run_dir, "params"))
        assert os.path.exists(os.path.join(mlflow_run_dir, "metrics"))
        assert os.path.exists(os.path.join(mlflow_run_dir, "artifacts"))
        
        # Check params
        with open(os.path.join(mlflow_run_dir, "params", "learning_rate"), "r") as f:
            lr = f.read().strip()
        assert lr == "0.01"
        
        # Check metrics - MLflow saves metrics as separate files
        accuracy_files = list(Path(os.path.join(mlflow_run_dir, "metrics")).glob("accuracy*"))
        assert len(accuracy_files) == 1
        
        # Check artifact was copied
        assert os.path.exists(os.path.join(mlflow_run_dir, "artifacts", "test_artifact.txt"))

    def test_restore_run(self):
        """Test restoring a deleted run."""
        # Create and delete a run
        run_id = self.tracker.start_run(run_name="test_run")
        self.tracker.log_param("param1", "value1")
        self.tracker.end_run()
        
        # Create a backup before deletion
        backup_result = self.tracker.backup_run(run_id)
        assert backup_result is True
        
        # Delete the run
        self.tracker.delete_run(run_id)
        
        # Verify run is deleted
        run_exists = os.path.exists(os.path.join(self.temp_dir, "runs", run_id))
        assert run_exists is False
        
        # Restore the run
        restore_result = self.tracker.restore_run(run_id)
        
        # Check result
        assert restore_result is True
        
        # Verify run was restored
        restored_run_dir = os.path.join(self.temp_dir, "runs", run_id)
        assert os.path.exists(restored_run_dir)
        
        # Check contents were restored
        run_info = self.tracker.get_run_info(run_id)
        assert run_info["run_name"] == "test_run"
        assert run_info["params"]["param1"] == "value1"

    def test_auto_resuming_run(self):
        """Test automatically resuming a previous run."""
        # Start a run
        original_id = self.tracker.start_run(run_name="test_run")
        self.tracker.log_param("learning_rate", 0.01)
        self.tracker.log_metric("accuracy", 0.80, step=1)
        self.tracker.end_run()
        
        # Resume the run
        resumed_id = self.tracker.start_run(run_name="test_run", resume=True)
        
        # Check it's the same run
        assert resumed_id == original_id
        
        # Add more metrics (continue from where we left off)
        self.tracker.log_metric("accuracy", 0.85, step=2)
        self.tracker.end_run()
        
        # Check run info
        run_info = self.tracker.get_run_info(original_id)
        
        # Check metrics have been appended
        metrics = run_info["metrics"]["accuracy"]
        assert len(metrics) == 2
        assert metrics[0]["value"] == 0.80
        assert metrics[0]["step"] == 1
        assert metrics[1]["value"] == 0.85
        assert metrics[1]["step"] == 2

    def test_error_handling_log_when_no_active_run(self):
        """Test error handling when logging with no active run."""
        # Try to log a param with no active run
        with pytest.raises(ValueError):
            self.tracker.log_param("param1", "value1")
        
        # Try to log a metric with no active run
        with pytest.raises(ValueError):
            self.tracker.log_metric("metric1", 1.0)
        
        # Try to log an artifact with no active run
        with pytest.raises(ValueError):
            self.tracker.log_artifact("non_existent_file.txt")

    def test_error_handling_invalid_inputs(self):
        """Test error handling with invalid inputs."""
        # Start a run for testing
        self.tracker.start_run(run_name="test_run")
        
        # Invalid param name
        with pytest.raises(ValueError):
            self.tracker.log_param("", "value")
        
        # Invalid metric name
        with pytest.raises(ValueError):
            self.tracker.log_metric("", 1.0)
        
        # Non-existent artifact
        with pytest.raises(FileNotFoundError):
            self.tracker.log_artifact("non_existent_file.txt")
        
        # End run to clean up
        self.tracker.end_run()

    def test_close(self):
        """Test closing the tracker and auto-ending active runs."""
        # Start a run but don't end it
        self.tracker.start_run(run_name="test_run")
        run_id = self.tracker.active_run_id
        
        # Close the tracker
        self.tracker.close()
        
        # Check that the run was automatically ended
        metadata_path = os.path.join(self.temp_dir, "runs", run_id, "metadata.json")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        assert metadata["status"] != "RUNNING"
        assert "end_time" in metadata