"""Connection management for TinyLCM client.

Provides reliable connection handling between edge devices and servers,
with support for retries, backoff, and status monitoring.
"""

import json
import logging
import time
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests

from tinylcm.utils.errors import ConnectionError
from tinylcm.utils.logging import setup_logger


class ConnectionManager:
    """
    Manages connections to a TinyLCM server.
    
    Handles connection establishment, monitoring, retries, and
    exponential backoff to ensure reliable communication even
    in challenging network environments.
    """
    
    def __init__(
        self,
        server_url: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        connection_timeout: float = 300.0,
        use_exponential_backoff: bool = False,
        backoff_factor: float = 2.0,
        request_timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the connection manager.
        
        Args:
            server_url: URL of the TinyLCM server
            max_retries: Maximum number of connection retry attempts
            retry_delay: Base delay between retries in seconds
            connection_timeout: How long a connection is considered valid before re-checking
            use_exponential_backoff: Whether to use exponential backoff for retries
            backoff_factor: Factor to increase delay by on each retry attempt
            request_timeout: Timeout for HTTP requests in seconds
            headers: Default headers to include in all requests
        """
        # Set up logger
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Connection parameters
        self.server_url = server_url.rstrip('/')  # Remove trailing slash for consistency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_timeout = connection_timeout
        self.use_exponential_backoff = use_exponential_backoff
        self.backoff_factor = backoff_factor
        self.request_timeout = request_timeout
        self.headers = headers or {}
        
        # Connection state
        self.connection_status = "disconnected"
        self.last_connection_time = None
        self.failed_attempts = 0
        
        # Callbacks for connection state changes
        self.status_callbacks: List[Callable[[str, Optional[str]], None]] = []
        
        self.logger.info(f"Initialized connection manager for server: {server_url}")
    
    def register_status_callback(
        self,
        callback: Callable[[str, Optional[str]], None]
    ) -> None:
        """
        Register a callback for connection status changes.
        
        Args:
            callback: Function to call when connection status changes.
                     Takes status string and optional info as parameters.
        """
        self.status_callbacks.append(callback)
        self.logger.debug(f"Registered status callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def _notify_status_change(self, status: str, info: Optional[str] = None) -> None:
        """
        Notify all registered callbacks about a status change.
        
        Args:
            status: New connection status
            info: Optional additional information
        """
        self.connection_status = status
        
        for callback in self.status_callbacks:
            try:
                callback(status, info)
            except Exception as e:
                self.logger.error(f"Error in status callback: {str(e)}")
    
    def connect(self) -> bool:
        """
        Attempt to connect to the server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        self.logger.debug(f"Attempting to connect to server: {self.server_url}")
        
        try:
            # Send a GET request to the server status endpoint
            response = requests.get(
                f"{self.server_url}/status",
                headers=self.headers,
                timeout=self.request_timeout
            )
            
            # Check if response indicates server is available
            if response.status_code == 200:
                self.logger.info(f"Successfully connected to server: {self.server_url}")
                self.last_connection_time = time.time()
                self.failed_attempts = 0
                self._notify_status_change("connected")
                return True
            else:
                error_msg = f"Server returned unexpected status code: {response.status_code}"
                self.logger.warning(error_msg)
                self.failed_attempts += 1
                self._notify_status_change("error", error_msg)
                return False
                
        except requests.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            self.logger.warning(error_msg)
            self.failed_attempts += 1
            self._notify_status_change("error", error_msg)
            return False
    
    def connect_with_retry(self) -> bool:
        """
        Attempt to connect with automatic retries.
        
        Uses exponential backoff if configured.
        
        Returns:
            bool: True if connection successful
            
        Raises:
            ConnectionError: If all retry attempts fail
        """
        # First attempt
        if self.connect():
            return True
        
        # If first attempt fails, retry with backoff
        retry_count = 1
        while retry_count < self.max_retries:
            # Calculate delay
            if self.use_exponential_backoff:
                delay = self.retry_delay * (self.backoff_factor ** (retry_count - 1))
            else:
                delay = self.retry_delay
            
            self.logger.info(f"Retrying connection (attempt {retry_count + 1}/{self.max_retries}) after {delay:.2f}s delay")
            self._notify_status_change("retrying", f"Attempt {retry_count + 1}/{self.max_retries}")
            
            # Wait before retry
            time.sleep(delay)
            
            # Try again
            if self.connect():
                return True
            
            retry_count += 1
        
        # If we get here, all retries failed
        error_msg = f"Failed to connect after {self.max_retries} attempts"
        self.logger.error(error_msg)
        raise ConnectionError(error_msg)
    
    def is_connected(self) -> bool:
        """
        Check if currently connected to the server.
        
        If last connection time exceeds timeout, re-checks connection.
        
        Returns:
            bool: True if connected, False otherwise
        """
        # If never connected, return False
        if self.last_connection_time is None:
            return False
        
        # Check if connection has timed out
        if time.time() - self.last_connection_time > self.connection_timeout:
            self.logger.debug("Connection timeout expired, re-checking connection")
            return self.connect()
        
        return self.connection_status == "connected"
    
    def reset_connection(self) -> None:
        """
        Reset connection state.
        
        Useful after network changes or to force reconnection.
        """
        self.logger.info("Resetting connection state")
        self.connection_status = "disconnected"
        self.last_connection_time = None
        self.failed_attempts = 0
        self._notify_status_change("disconnected")
    
    def execute_request(
        self,
        method: str,
        endpoint: str,
        auto_connect: bool = True,
        retry_on_failure: bool = True,
        **kwargs
    ) -> requests.Response:
        """
        Execute an HTTP request to the server.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: Server endpoint (will be appended to server_url)
            auto_connect: Whether to check connection before request
            retry_on_failure: Whether to retry on connection failures
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            requests.Response: Response from the server
            
        Raises:
            ConnectionError: If connection fails and retries are exhausted
            requests.RequestException: For other request errors
        """
        # If auto_connect, ensure we're connected
        if auto_connect and not self.is_connected():
            if retry_on_failure:
                self.connect_with_retry()
            else:
                self.connect()
                
        # Normalize endpoint
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        
        url = f"{self.server_url}/{endpoint}"
        
        # Add default headers if not overridden
        request_headers = self.headers.copy()
        if 'headers' in kwargs:
            request_headers.update(kwargs.pop('headers'))
        
        # Set default timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.request_timeout
        
        try:
            # Execute request with appropriate method
            method = method.upper()
            
            if method == 'GET':
                response = requests.get(url, headers=request_headers, **kwargs)
            elif method == 'POST':
                response = requests.post(url, headers=request_headers, **kwargs)
            elif method == 'PUT':
                response = requests.put(url, headers=request_headers, **kwargs)
            elif method == 'DELETE':
                response = requests.delete(url, headers=request_headers, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return response
            
        except requests.RequestException as e:
            self.logger.error(f"Request error for {method} {url}: {str(e)}")
            # Update connection status if we received a connection error
            if isinstance(e, (requests.ConnectionError, requests.Timeout)):
                self.failed_attempts += 1
                self._notify_status_change("error", str(e))
                
                # If retry on failure is enabled, try to reconnect
                if retry_on_failure:
                    self.logger.info("Attempting to reconnect before retrying request")
                    if self.connect_with_retry():
                        self.logger.info("Reconnected successfully, retrying request")
                        return self.execute_request(
                            method=method,
                            endpoint=endpoint,
                            auto_connect=False,  # Already connected
                            retry_on_failure=False,  # Avoid infinite recursion
                            **kwargs
                        )
            
            # Re-raise the original exception
            raise