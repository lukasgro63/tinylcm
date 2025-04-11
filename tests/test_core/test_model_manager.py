"""Tests for ModelManager component."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from tinylcm.core.model_manager import ModelManager
from tinylcm.utils.errors import ModelNotFoundError


class TestModelManager:
    """Test ModelManager functionality."""

    def setup_method(self):
        """Set up temporary directory and mock models for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        self.model_manager = ModelManager(storage_dir=self.models_dir)

        # Create a mock model file
        self.mock_model_dir = os.path.join(self.temp_dir, "mock_models")
        os.makedirs(self.mock_model_dir, exist_ok=True)

        self.model_file_path = os.path.join(self.mock_model_dir, "test_model.json")
        self.model_content = {"weights": [1.0, 2.0, 3.0], "layers": [10, 5, 1]}

        with open(self.model_file_path, "w") as f:
            json.dump(self.model_content, f)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        assert os.path.exists(self.models_dir)
        assert os.path.exists(os.path.join(self.models_dir, "models"))
        assert os.path.exists(os.path.join(self.models_dir, "metadata"))

    def test_save_model(self):
        """Test saving a model."""
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            version="v1",
            description="Test model",
            tags=["test", "json"],
            metrics={"accuracy": 0.95}
        )

        # Check that returned model ID is a string
        assert isinstance(model_id, str)

        # Check that model was copied to the right place
        model_dir = os.path.join(self.models_dir, "models", model_id)
        assert os.path.exists(model_dir)

        model_copy_path = os.path.join(model_dir, "test_model.json")
        assert os.path.exists(model_copy_path)

        # Check that metadata was created
        metadata_path = os.path.join(self.models_dir, "metadata", f"{model_id}.json")
        assert os.path.exists(metadata_path)

        # Check metadata content
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["model_id"] == model_id
        assert metadata["version"] == "v1"
        assert metadata["model_format"] == "json"
        assert metadata["description"] == "Test model"
        assert "test" in metadata["tags"]
        assert "json" in metadata["tags"]
        assert metadata["metrics"]["accuracy"] == 0.95
        assert "md5_hash" in metadata

    def test_save_model_with_set_active(self):
        """Test saving a model and setting it as active."""
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            set_active=True
        )

        # Check that symlink was created
        active_link = os.path.join(self.models_dir, "active_model")
        assert os.path.exists(active_link)
        assert os.path.islink(active_link)

        # Check that metadata marks it as active
        metadata_path = os.path.join(self.models_dir, "metadata", f"{model_id}.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["is_active"] is True

    def test_load_model(self):
        """Test loading a model."""
        # Save a model first
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json"
        )

        # Load the model
        model_path = self.model_manager.load_model(model_id)

        # Check that returned path is a string
        assert isinstance(model_path, str)

        # Check that file exists
        assert os.path.exists(model_path)

        # Check file content
        with open(model_path, "r") as f:
            loaded_content = json.load(f)

        assert loaded_content == self.model_content

    def test_load_active_model(self):
        """Test loading the active model."""
        # Save a model and set as active
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            set_active=True
        )

        # Load the active model
        model_path = self.model_manager.load_model()  # No model_id -> active model

        # Check that file exists and has correct content
        assert os.path.exists(model_path)
        with open(model_path, "r") as f:
            loaded_content = json.load(f)

        assert loaded_content == self.model_content

    def test_load_model_nonexistent(self):
        """Test loading a nonexistent model."""
        with pytest.raises(ModelNotFoundError):
            self.model_manager.load_model("nonexistent_model")

    def test_get_model_metadata(self):
        """Test getting model metadata."""
        # Save a model
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            description="Test model"
        )

        # Get metadata
        metadata = self.model_manager.get_model_metadata(model_id)

        # Check metadata content
        assert metadata["model_id"] == model_id
        assert metadata["description"] == "Test model"

    def test_get_active_model_metadata(self):
        """Test getting active model metadata."""
        # Save a model and set as active
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            set_active=True
        )

        # Get active model metadata
        metadata = self.model_manager.get_model_metadata()  # No model_id -> active model

        # Check metadata content
        assert metadata["model_id"] == model_id
        assert metadata["is_active"] is True

    def test_list_models(self):
        """Test listing models."""
        # Save multiple models
        model1_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            version="v1",
            tags=["baseline"]
        )

        model2_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            version="v2",
            tags=["improved"]
        )

        # List all models
        models = self.model_manager.list_models()

        # Check that both models are listed
        assert len(models) == 2
        model_ids = [m["model_id"] for m in models]
        assert model1_id in model_ids
        assert model2_id in model_ids

    def test_list_models_with_filter(self):
        """Test listing models with filtering."""
        # Save models with different tags
        self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            tags=["tag1"]
        )

        self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="tflite",
            tags=["tag2"]
        )

        # Filter by tag
        tag1_models = self.model_manager.list_models(tag="tag1")
        assert len(tag1_models) == 1
        assert "tag1" in tag1_models[0]["tags"]

        # Filter by format
        tflite_models = self.model_manager.list_models(model_format="tflite")
        assert len(tflite_models) == 1
        assert tflite_models[0]["model_format"] == "tflite"

    def test_set_active_model(self):
        """Test setting the active model."""
        # Save two models
        model1_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            version="v1"
        )

        model2_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            version="v2"
        )

        # Set model2 as active
        self.model_manager.set_active_model(model2_id)

        # Check that model2 is active
        active_link = os.path.join(self.models_dir, "active_model")
        assert os.path.exists(active_link)

        # Check metadata
        metadata1 = self.model_manager.get_model_metadata(model1_id)
        metadata2 = self.model_manager.get_model_metadata(model2_id)

        assert metadata1["is_active"] is False
        assert metadata2["is_active"] is True

    def test_delete_model(self):
        """Test deleting a model."""
        # Save a model
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json"
        )

        # Delete the model
        result = self.model_manager.delete_model(model_id)

        # Check result
        assert result is True

        # Check that model files and metadata are deleted
        model_dir = os.path.join(self.models_dir, "models", model_id)
        metadata_path = os.path.join(self.models_dir, "metadata", f"{model_id}.json")

        assert not os.path.exists(model_dir)
        assert not os.path.exists(metadata_path)

    def test_delete_active_model(self):
        """Test deleting the active model."""
        # Save a model and set as active
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            set_active=True
        )

        # Try to delete without force - should raise
        with pytest.raises(ValueError):
            self.model_manager.delete_model(model_id)

        # Delete with force
        result = self.model_manager.delete_model(model_id, force=True)
        assert result is True

        # Check that symlink is removed
        active_link = os.path.join(self.models_dir, "active_model")
        assert not os.path.exists(active_link)

    def test_add_tag(self):
        """Test adding a tag to a model."""
        # Save a model
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            tags=["initial"]
        )

        # Add a tag
        self.model_manager.add_tag(model_id, "new_tag")

        # Check metadata
        metadata = self.model_manager.get_model_metadata(model_id)
        assert "initial" in metadata["tags"]
        assert "new_tag" in metadata["tags"]

    def test_remove_tag(self):
        """Test removing a tag from a model."""
        # Save a model with tags
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            tags=["tag1", "tag2"]
        )

        # Remove a tag
        self.model_manager.remove_tag(model_id, "tag1")

        # Check metadata
        metadata = self.model_manager.get_model_metadata(model_id)
        assert "tag1" not in metadata["tags"]
        assert "tag2" in metadata["tags"]

    def test_update_metrics(self):
        """Test updating metrics for a model."""
        # Save a model with initial metrics
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json",
            metrics={"accuracy": 0.8}
        )

        # Update metrics
        self.model_manager.update_metrics(model_id, {
            "accuracy": 0.85,
            "precision": 0.9
        })

        # Check metadata
        metadata = self.model_manager.get_model_metadata(model_id)
        assert metadata["metrics"]["accuracy"] == 0.85
        assert metadata["metrics"]["precision"] == 0.9

    def test_verify_model_integrity(self):
        """Test model integrity verification."""
        # Save a model
        model_id = self.model_manager.save_model(
            model_path=self.model_file_path,
            model_format="json"
        )

        # Verify integrity
        is_valid = self.model_manager.verify_model_integrity(model_id)
        assert is_valid is True

        # Tamper with the model file
        model_path = self.model_manager.load_model(model_id)
        with open(model_path, "w") as f:
            json.dump({"tampered": True}, f)

        # Verify again
        is_valid = self.model_manager.verify_model_integrity(model_id)
        assert is_valid is False