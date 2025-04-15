"""Tests for SyncClient component."""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from tinylcm.client.sync_client import SyncClient
from tinylcm.core.sync_interface import SyncInterface
from tinylcm.utils.errors import SyncError, ConnectionError


class TestSyncClient:
    """Test SyncClient functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sync_dir = os.path.join(self.temp_dir, "sync")
        
        # Create sync interface for preparing packages
        self.sync_interface = SyncInterface(sync_dir=self.sync_dir)
        
        # Create client with mock server URL
        self.server_url = "http://example.com/api"
        self.api_key = "test_api_key"
        self.device_id = "test_device_123"
        
        self.client = SyncClient(
            server_url=self.server_url,
            api_key=self.api_key,
            device_id=self.device_id,
            sync_interface=self.sync_interface
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('requests.post')
    def test_register_device(self, mock_post):
        """Test device registration with server."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "device_id": self.device_id,
            "status": "registered",
            "server_time": time.time()
        }
        mock_post.return_value = mock_response
        
        # Call register method
        result = self.client.register_device()
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{self.server_url}/devices/register"
        assert kwargs["json"]["device_id"] == self.device_id
        assert "device_info" in kwargs["json"]
        assert "headers" in kwargs
        assert kwargs["headers"]["Authorization"] == f"Bearer {self.api_key}"
        
        # Verify result
        assert result is True
    
    @patch('requests.post')
    def test_register_device_failure(self, mock_post):
        """Test device registration failure handling."""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": "Invalid API key"
        }
        mock_post.return_value = mock_response
        
        # Call register method should raise error
        with pytest.raises(ConnectionError):
            self.client.register_device()
    
    @patch('requests.post')
    def test_send_package(self, mock_post):
        """Test sending a sync package to server."""
        # Create a test package
        package_id = self.sync_interface.create_package(
            device_id=self.device_id,
            package_type="test_data"
        )
        
        # Create a dummy file to add to package
        dummy_file = os.path.join(self.temp_dir, "dummy.txt")
        with open(dummy_file, "w") as f:
            f.write("Test data")
        
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=dummy_file,
            file_type="test"
        )
        
        package_path = self.sync_interface.finalize_package(package_id)
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "package_id": package_id,
            "status": "received",
            "server_time": time.time()
        }
        mock_post.return_value = mock_response
        
        # Call send package method
        result = self.client.send_package(package_id)
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{self.server_url}/packages/upload"
        assert "files" in kwargs
        assert "headers" in kwargs
        assert kwargs["headers"]["Authorization"] == f"Bearer {self.api_key}"
        
        # Verify result
        assert result is True
        
        # Verify package is marked as synced
        packages = self.sync_interface.list_packages(include_synced=True)
        synced_package = next((p for p in packages if p["package_id"] == package_id), None)
        assert synced_package is not None
        assert synced_package["sync_status"] == "success"
    
    @patch('requests.post')
    def test_upload_failure_handling(self, mock_post):
        """Test handling of upload failures."""
        # Create a test package
        package_id = self.sync_interface.create_package(
            device_id=self.device_id,
            package_type="test_data"
        )
        
        # Create a dummy file to add to package
        dummy_file = os.path.join(self.temp_dir, "dummy.txt")
        with open(dummy_file, "w") as f:
            f.write("Test data")
        
        self.sync_interface.add_file_to_package(
            package_id=package_id,
            file_path=dummy_file,
            file_type="test"
        )
        
        package_path = self.sync_interface.finalize_package(package_id)
        
        # Mock network error
        mock_post.side_effect = requests.ConnectionError("Network error")
        
        # Call send package method should raise error
        with pytest.raises(ConnectionError):
            self.client.send_package(package_id)
        
        # Verify package is not marked as synced
        packages = self.sync_interface.list_packages(include_synced=True)
        synced_package = next((p for p in packages if p["package_id"] == package_id), None)
        assert synced_package is not None
        assert "sync_status" not in synced_package
    
    @patch('requests.get')
    def test_check_server_status(self, mock_get):
        """Test checking server status."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "online",
            "version": "1.0.0",
            "server_time": time.time()
        }
        mock_get.return_value = mock_response
        
        # Call status check method
        status = self.client.check_server_status()
        
        # Verify request was made correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == f"{self.server_url}/status"
        
        # Verify result
        assert status["status"] == "online"
        assert "version" in status
    
    def test_validate_server_url(self):
        """Test validation of server URL."""
        # Valid URLs
        assert SyncClient.validate_server_url("http://example.com/api") is True
        assert SyncClient.validate_server_url("https://example.com/api") is True
        
        # Invalid URLs
        assert SyncClient.validate_server_url("ftp://example.com") is False
        assert SyncClient.validate_server_url("not a url") is False
        assert SyncClient.validate_server_url("") is False
    
    @patch('requests.post')
    def test_sync_all_packages(self, mock_post):
        """Test syncing all pending packages."""
        # Create multiple test packages
        package_ids = []
        for i in range(3):
            package_id = self.sync_interface.create_package(
                device_id=self.device_id,
                package_type=f"test_data_{i}"
            )
            
            # Create a dummy file to add to package
            dummy_file = os.path.join(self.temp_dir, f"dummy_{i}.txt")
            with open(dummy_file, "w") as f:
                f.write(f"Test data {i}")
            
            self.sync_interface.add_file_to_package(
                package_id=package_id,
                file_path=dummy_file,
                file_type="test"
            )
            
            self.sync_interface.finalize_package(package_id)
            package_ids.append(package_id)
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "package_id": "any_id",
            "status": "received",
            "server_time": time.time()
        }
        mock_post.return_value = mock_response
        
        # Call sync all method
        results = self.client.sync_all_pending_packages()
        
        # Verify results
        assert len(results) == 3
        assert all(r["success"] for r in results)
        
        # Verify all packages marked as synced
        packages = self.sync_interface.list_packages(include_synced=True)
        for package_id in package_ids:
            pkg = next((p for p in packages if p["package_id"] == package_id), None)
            assert pkg is not None
            assert pkg["sync_status"] == "success"