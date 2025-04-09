"""Tests for DataLogger component."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from tinylcm.core.data_logger import DataLogger


class TestDataLogger:
    """Test DataLogger functionality."""
    
    def setup_method(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.temp_dir, "data_logs")
        self.logger = DataLogger(
            storage_dir=self.storage_dir,
            buffer_size=3  # Small buffer for testing
        )
    
    def teardown_method(self):
        """Clean up resources and temporary directory."""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        assert os.path.exists(self.storage_dir)
        assert os.path.exists(os.path.join(self.storage_dir, "images"))
        assert os.path.exists(os.path.join(self.storage_dir, "metadata"))
    
    def test_log_data_text(self):
        """Test logging text data."""
        # Log some text data
        entry_id = self.logger.log_data(
            input_data="This is some test text",
            input_type="text",
            prediction="positive",
            confidence=0.92,
            label="positive",
            metadata={"source": "test"}
        )
        
        # Check that entry_id is returned and is a string
        assert isinstance(entry_id, str)
        
        # Check that a file was created (in buffer, not yet on disk)
        assert len(self.logger.metadata_buffer) == 1
        
        # Get the entry
        entry = self.logger.get_entry(entry_id)
        
        # Check entry content
        assert entry["entry_id"] == entry_id
        assert entry["input_type"] == "text"
        assert entry["prediction"] == "positive"
        assert entry["confidence"] == 0.92
        assert entry["label"] == "positive"
        assert entry["metadata"]["source"] == "test"
        assert "timestamp" in entry
        assert "filename" in entry
        assert "session_id" in entry
    
    def test_log_data_json(self):
        """Test logging JSON data."""
        # Log some JSON data
        json_data = {"feature1": 1.0, "feature2": 2.0, "nested": {"key": "value"}}
        entry_id = self.logger.log_data(
            input_data=json_data,
            input_type="json",
            prediction="class_a"
        )
        
        # Get the entry
        entry = self.logger.get_entry(entry_id)
        
        # Check entry content
        assert entry["input_type"] == "json"
        assert entry["prediction"] == "class_a"
        assert "filename" in entry
    
    def test_log_image(self):
        """Test logging image data."""
        # Create image data
        image_data = b"MOCK_IMAGE_DATA"
        
        # Log image
        entry_id = self.logger.log_image(
            image_data=image_data,
            prediction="cat",
            confidence=0.88,
            metadata={"camera_id": "test_cam"}
        )
        
        # Get the entry
        entry = self.logger.get_entry(entry_id)
        
        # Check entry content
        assert entry["input_type"] == "image"
        assert entry["prediction"] == "cat"
        assert entry["confidence"] == 0.88
        assert entry["metadata"]["camera_id"] == "test_cam"
        
        # Force write to disk
        self.logger._write_metadata_buffer()
        
        # Check that image file exists
        entry = self.logger.get_entry(entry_id)  # Refresh from disk
        image_path = os.path.join(self.storage_dir, entry["filename"])
        assert os.path.exists(image_path)
        
        # Check image content
        with open(image_path, "rb") as f:
            saved_image = f.read()
        assert saved_image == image_data
    
    def test_buffer_flush_on_threshold(self):
        """Test that buffer flushes when it reaches the threshold."""
        # Buffer size is 3 (set in setup_method)
        
        # Log 3 items (should trigger a flush)
        for i in range(3):
            self.logger.log_data(
                input_data=f"Data {i}",
                input_type="text",
                prediction=f"class_{i}"
            )
        
        # Check that buffer was flushed
        assert len(self.logger.metadata_buffer) == 0
        
        # Check that files were created on disk
        metadata_files = os.listdir(os.path.join(self.storage_dir, "metadata"))
        assert len(metadata_files) >= 3
    
    def test_log_prediction(self):
        """Test adding a prediction to existing data."""
        # Log data without prediction
        entry_id = self.logger.log_data(
            input_data="Data without prediction",
            input_type="text"
        )
        
        # Add prediction later
        result = self.logger.log_prediction(
            input_id=entry_id,
            prediction="late_prediction",
            confidence=0.75
        )
        
        # Check result
        assert result is True
        
        # Check that prediction was added
        entry = self.logger.get_entry(entry_id)
        assert entry["prediction"] == "late_prediction"
        assert entry["confidence"] == 0.75
    
    def test_query_entries(self):
        """Test querying entries with filters."""
        # Log entries with different timestamps
        base_time = time.time()
        
        # Older entry
        entry1 = self.logger.log_data(
            input_data="Older data",
            input_type="text",
            metadata={"time_override": base_time - 100}
        )
        
        # Middle entry
        entry2 = self.logger.log_data(
            input_data="Middle data",
            input_type="json",
            metadata={"time_override": base_time - 50}
        )
        
        # Newer entry
        entry3 = self.logger.log_data(
            input_data="Newer data",
            input_type="text",
            metadata={"time_override": base_time}
        )
        
        # Force write to disk
        self.logger._write_metadata_buffer()
        
        # Query with time range
        results = self.logger.query_entries(
            start_time=base_time - 75,
            end_time=base_time + 1
        )
        
        # Should include entry2 and entry3
        assert len(results) == 2
        result_ids = [r["entry_id"] for r in results]
        assert entry2 in result_ids
        assert entry3 in result_ids
        assert entry1 not in result_ids
        
        # Query with input type
        text_results = self.logger.query_entries(input_type="text")
        assert len(text_results) == 2
        text_ids = [r["entry_id"] for r in text_results]
        assert entry1 in text_ids
        assert entry3 in text_ids
    
    def test_export_to_csv(self):
        """Test exporting data to CSV."""
        # Log some entries
        for i in range(5):
            self.logger.log_data(
                input_data=f"Data {i}",
                input_type="text",
                prediction=f"class_{i % 2}",
                confidence=0.8 + i * 0.02
            )
        
        # Force write to disk
        self.logger._write_metadata_buffer()
        
        # Export to CSV
        csv_path = self.logger.export_to_csv()
        
        # Check that file exists
        assert os.path.exists(csv_path)
        
        # Check content (basic check)
        with open(csv_path, "r") as f:
            csv_content = f.read()
        
        # Should have a header row and 5 data rows
        assert len(csv_content.strip().split("\n")) == 6
        
        # Check that headers are present
        assert "timestamp" in csv_content
        assert "entry_id" in csv_content
        assert "prediction" in csv_content
        assert "confidence" in csv_content
    
    def test_get_data_file(self):
        """Test getting the file path for an entry's data."""
        # Log some data
        entry_id = self.logger.log_data(
            input_data="Test data",
            input_type="text"
        )
        
        # Force write to disk
        self.logger._write_metadata_buffer()
        
        # Get data file path
        file_path = self.logger.get_data_file(entry_id)
        
        # Check that path is returned and file exists
        assert file_path is not None
        assert os.path.exists(file_path)
    
    def test_count_entries(self):
        """Test counting entries with filters."""
        # Log entries with different types
        for i in range(10):
            input_type = "text" if i % 2 == 0 else "json"
            self.logger.log_data(
                input_data=f"Data {i}",
                input_type=input_type
            )
        
        # Force write to disk
        self.logger._write_metadata_buffer()
        
        # Count all entries
        total_count = self.logger.count_entries()
        assert total_count == 10
        
        # Count text entries
        text_count = self.logger.count_entries(input_type="text")
        assert text_count == 5
        
        # Count JSON entries
        json_count = self.logger.count_entries(input_type="json")
        assert json_count == 5
    
    def test_close(self):
        """Test closing the logger and saving remaining data."""
        # Log some data without filling the buffer
        self.logger.log_data(
            input_data="Final data",
            input_type="text"
        )
        
        # Close logger
        self.logger.close()
        
        # Check that data was written to disk
        metadata_files = os.listdir(os.path.join(self.storage_dir, "metadata"))
        assert len(metadata_files) > 0