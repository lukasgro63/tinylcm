"""Tests for ConnectionManager component."""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from tinylcm.client.connection_manager import ConnectionManager
from tinylcm.utils.errors import ConnectionError


class TestConnectionManager:
    """Test ConnectionManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.server_url = "http://example.com/api"
        self.max_retries = 3
        self.retry_delay = 0.1  # Short delay for testing
        
        self.manager = ConnectionManager(
            server_url=self.server_url,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )
        
        # Set up callback tracking
        self.connection_status_changes = []
        
        def status_callback(status, info=None):
            self.connection_status_changes.append((status, info))
        
        self.manager.register_status_callback(status_callback)
    
    def test_initial_status(self):
        """Test initial connection status."""
        assert self.manager.connection_status == "disconnected"
        assert self.manager.last_connection_time is None
        assert self.manager.failed_attempts == 0
    
    @patch('requests.get')
    def test_successful_connection(self, mock_get):
        """Test successful connection attempt."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "online"}
        mock_get.return_value = mock_response
        
        # Test connect method
        result = self.manager.connect()
        
        # Verify results
        assert result is True
        assert self.manager.connection_status == "connected"
        assert self.manager.last_connection_time is not None
        assert self.manager.failed_attempts == 0
        
        # Verify callback was called
        assert len(self.connection_status_changes) == 1
        assert self.connection_status_changes[0][0] == "connected"
    
    @patch('requests.get')
    def test_failed_connection(self, mock_get):
        """Test failed connection attempt."""
        # Mock failed response
        mock_get.side_effect = requests.ConnectionError("Network error")
        
        # Test connect method
        result = self.manager.connect()
        
        # Verify results
        assert result is False
        assert self.manager.connection_status == "disconnected"
        assert self.manager.failed_attempts == 1
        
        # Verify callback was called
        assert len(self.connection_status_changes) == 1
        assert self.connection_status_changes[0][0] == "error"
        assert "Network error" in self.connection_status_changes[0][1]
    
    @patch('requests.get')
    def test_retry_mechanism(self, mock_get):
        """Test connection retry mechanism."""
        # Mock intermittent failure
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"status": "online"}
        
        # First attempt fails, second succeeds
        mock_get.side_effect = [
            requests.ConnectionError("Network error"),
            mock_response_success
        ]
        
        # Test connect_with_retry method
        result = self.manager.connect_with_retry()
        
        # Verify results
        assert result is True
        assert self.manager.connection_status == "connected"
        assert self.manager.failed_attempts == 0  # Reset after success
        assert mock_get.call_count == 2
        
        # Verify callbacks were called
        assert len(self.connection_status_changes) == 3  # error, retrying, connected
        assert self.connection_status_changes[0][0] == "error"
        assert self.connection_status_changes[1][0] == "retrying"
        assert self.connection_status_changes[2][0] == "connected"
    
    @patch('requests.get')
    def test_retry_exhaustion(self, mock_get):
        """Test exhausting all retry attempts."""
        # Mock persistent failure
        mock_get.side_effect = requests.ConnectionError("Network error")
        
        # Test connect_with_retry method
        with pytest.raises(ConnectionError):
            self.manager.connect_with_retry()
        
        # Verify correct number of attempts
        assert mock_get.call_count == self.max_retries
        assert self.manager.failed_attempts == self.max_retries
        
        # Verify callbacks
        assert len(self.connection_status_changes) == 2 * self.max_retries  # error, retrying for each attempt
        assert self.connection_status_changes[-1][0] == "error"
    
    @patch('requests.get')
    def test_is_connected(self, mock_get):
        """Test checking if currently connected."""
        # Initial state
        assert self.manager.is_connected() is False
        
        # Mock successful connection
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "online"}
        mock_get.return_value = mock_response
        
        self.manager.connect()
        assert self.manager.is_connected() is True
        
        # Test after timeout
        self.manager.connection_timeout = 0.1  # Short timeout for testing
        time.sleep(0.2)
        
        # Should return False and trigger a re-check
        mock_get.reset_mock()
        assert self.manager.is_connected() is True  # Still connected due to successful re-check
        mock_get.assert_called_once()  # Verify re-check happened
    
    @patch('requests.get')
    def test_connection_backoff(self, mock_get):
        """Test exponential backoff for connection retries."""
        # Enable backoff
        self.manager = ConnectionManager(
            server_url=self.server_url,
            max_retries=3,
            retry_delay=0.1,
            use_exponential_backoff=True
        )
        
        # Mock persistent failure
        mock_get.side_effect = requests.ConnectionError("Network error")
        
        # Register for timing information
        retry_times = []
        def track_retry_time(status, info=None):
            if status == "retrying":
                retry_times.append(time.time())
        
        self.manager.register_status_callback(track_retry_time)
        
        # Test connect_with_retry method (will raise eventually)
        with pytest.raises(ConnectionError):
            self.manager.connect_with_retry()
        
        # Verify increasing delays between retries
        assert len(retry_times) == 2  # 3 attempts, 2 retries
        time_diff = retry_times[1] - retry_times[0]
        assert time_diff > 0.1  # Second retry should be delayed longer than first
    
    def test_reset_connection(self):
        """Test resetting connection state."""
        # Set some connection state
        self.manager.connection_status = "connected"
        self.manager.last_connection_time = time.time()
        self.manager.failed_attempts = 2
        
        # Reset connection
        self.manager.reset_connection()
        
        # Verify reset state
        assert self.manager.connection_status == "disconnected"
        assert self.manager.last_connection_time is None
        assert self.manager.failed_attempts == 0
        
        # Verify callback
        assert self.connection_status_changes[-1][0] == "disconnected"
    
    @patch('requests.post')
    def test_execute_request(self, mock_post):
        """Test executing a request with auto-connection."""
        # Mock successful connection and request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response
        
        # Test execute_request method
        response = self.manager.execute_request(
            "POST", 
            "/test-endpoint",
            data={"test": "data"},
            auto_connect=True
        )
        
        # Verify results
        assert response.status_code == 200
        assert response.json()["result"] == "success"
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{self.server_url}/test-endpoint"
        assert kwargs["json"] == {"test": "data"}