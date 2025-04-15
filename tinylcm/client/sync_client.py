"""Client for synchronizing data with TinyLCM server.

Provides functionality for edge devices to synchronize models, metrics,
and logs with a central TinyLCM server.
"""

import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from tinylcm.client.connection_manager import ConnectionManager
from tinylcm.core.sync_interface import SyncInterface
from tinylcm.utils.errors import SyncError, ConnectionError
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.versioning import calculate_file_hash


class SyncClient:
    """
    Client for synchronizing data with a TinyLCM server.
    
    Handles device registration, package transmission, and
    server communication for TinyLCM-enabled edge devices.
    """
    
    def __init__(
        self,
        server_url: str,
        api_key: str,
        device_id: str,
        sync_interface: Optional[SyncInterface] = None,
        sync_dir: Optional[Union[str, Path]] = None,
        max_retries: int = 3,
        connection_timeout: float = 300.0,
        auto_register: bool = True
    ):
        """
        Initialize the synchronization client.
        
        Args:
            server_url: URL of the TinyLCM server
            api_key: API key for authentication with the server
            device_id: Unique identifier for this device
            sync_interface: SyncInterface instance (or one will be created)
            sync_dir: Directory for sync data (if sync_interface not provided)
            max_retries: Maximum connection retry attempts
            connection_timeout: How long a connection is considered valid
            auto_register: Whether to automatically register with server on first connection
            
        Raises:
            ValueError: If server_url is invalid
        """
        # Validate server URL
        if not self.validate_server_url(server_url):
            raise ValueError(f"Invalid server URL: {server_url}")
        
        # Set up logger
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize parameters
        self.server_url = server_url.rstrip('/')  # Remove trailing slash for consistency
        self.api_key = api_key
        self.device_id = device_id
        self.auto_register = auto_register
        
        # Initialize SyncInterface if not provided
        if sync_interface is None:
            if sync_dir is None:
                raise ValueError("Either sync_interface or sync_dir must be provided")
            self.sync_interface = SyncInterface(sync_dir=sync_dir)
        else:
            self.sync_interface = sync_interface
        
        # Default request headers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Device-ID": device_id
        }
        
        # Initialize connection manager
        self.connection_manager = ConnectionManager(
            server_url=server_url,
            max_retries=max_retries,
            connection_timeout=connection_timeout,
            headers=self.headers
        )
        
        self.logger.info(f"Initialized sync client for server: {server_url}")
        
        # Auto-register if configured
        if auto_register:
            try:
                self.register_device()
            except ConnectionError as e:
                self.logger.warning(f"Auto-registration failed: {str(e)}. Will retry on next connection.")
    
    @staticmethod
    def validate_server_url(url: str) -> bool:
        """
        Validate if a URL is a valid TinyLCM server URL.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Basic URL validation
        if not url:
            return False
            
        # Must be HTTP or HTTPS
        url_pattern = re.compile(r'^https?://.*')
        if not url_pattern.match(url):
            return False
            
        # Additional validation could be added here
        
        return True
    
    def _get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the current device.
        
        Returns:
            Dict[str, Any]: Device information
        """
        import socket
        
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
        except Exception:
            hostname = "unknown"
            ip_address = "unknown"
        
        return {
            "device_id": self.device_id,
            "hostname": hostname,
            "ip_address": ip_address,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "tinylcm_version": self._get_tinylcm_version()
        }
    
    def _get_tinylcm_version(self) -> str:
        """
        Get the version of TinyLCM being used.
        
        Returns:
            str: TinyLCM version
        """
        try:
            import tinylcm
            return tinylcm.__version__
        except (ImportError, AttributeError):
            return "unknown"
    
    def register_device(self) -> bool:
        """
        Register this device with the TinyLCM server.
        
        Returns:
            bool: True if registration successful
            
        Raises:
            ConnectionError: If registration fails due to connection issues
        """
        self.logger.info(f"Registering device {self.device_id} with server")
        
        # Prepare registration data
        registration_data = {
            "device_id": self.device_id,
            "device_info": self._get_device_info(),
            "registration_time": time.time()
        }
        
        try:
            # Send registration request
            response = self.connection_manager.execute_request(
                method="POST",
                endpoint="devices/register",
                json=registration_data
            )
            
            # Check response
            if response.status_code == 200:
                self.logger.info(f"Successfully registered device {self.device_id}")
                return True
            else:
                error_msg = f"Registration failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise ConnectionError(error_msg)
                
        except requests.RequestException as e:
            error_msg = f"Registration request failed: {str(e)}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def check_server_status(self) -> Dict[str, Any]:
        """
        Check the status of the TinyLCM server.
        
        Returns:
            Dict[str, Any]: Server status information
            
        Raises:
            ConnectionError: If status check fails
        """
        self.logger.debug("Checking server status")
        
        try:
            response = self.connection_manager.execute_request(
                method="GET",
                endpoint="status"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Server status check failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise ConnectionError(error_msg)
                
        except requests.RequestException as e:
            error_msg = f"Server status request failed: {str(e)}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def send_package(self, package_id: str) -> bool:
        """
        Send a sync package to the TinyLCM server.
        
        Args:
            package_id: ID of the package to send
            
        Returns:
            bool: True if successful
            
        Raises:
            SyncError: If package not found or invalid
            ConnectionError: If sending fails due to connection issues
        """
        self.logger.info(f"Preparing to send package: {package_id}")
        
        # Get package metadata
        try:
            packages = self.sync_interface.list_packages(
                filter_func=lambda p: p["package_id"] == package_id
            )
            
            if not packages:
                raise SyncError(f"Package not found: {package_id}")
                
            package_meta = packages[0]
            
            # Find package file (sorted by modification time if multiple match)
            package_dir = Path(self.sync_interface.packages_dir)
            package_files = list(package_dir.glob(f"{package_id}_*.tar.gz")) + \
                           list(package_dir.glob(f"{package_id}_*.zip")) + \
                           list(package_dir.glob(f"{package_id}_*.tar"))
                           
            if not package_files:
                raise SyncError(f"Package file not found for ID: {package_id}")
                
            # Sort by modification time, newest first
            package_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            package_file = package_files[0]
            
            self.logger.debug(f"Found package file: {package_file}")
            
            # Calculate hash for integrity verification
            file_hash = calculate_file_hash(package_file)
            
            # Prepare upload request
            files = {
                'package': (package_file.name, open(package_file, 'rb'), 'application/octet-stream')
            }
            
            # Prepare metadata for the form
            data = {
                'device_id': self.device_id,
                'package_id': package_id,
                'package_type': package_meta.get('package_type', 'unknown'),
                'hash': file_hash,
                'timestamp': time.time()
            }
            
            # Send the package
            try:
                response = self.connection_manager.execute_request(
                    method="POST",
                    endpoint="packages/upload",
                    files=files,
                    data={'metadata': json.dumps(data)}
                )
                
                # Check response
                if response.status_code == 200:
                    self.logger.info(f"Successfully sent package {package_id}")
                    
                    # Mark as synced
                    self.sync_interface.mark_as_synced(
                        package_id=package_id,
                        sync_time=time.time(),
                        server_id=response.json().get('server_id', 'unknown'),
                        status="success"
                    )
                    
                    return True
                else:
                    error_msg = f"Package upload failed: {response.status_code} - {response.text}"
                    self.logger.error(error_msg)
                    
                    # Mark as failed
                    self.sync_interface.mark_as_synced(
                        package_id=package_id,
                        sync_time=time.time(),
                        server_id="none",
                        status="error"
                    )
                    
                    raise ConnectionError(error_msg)
                    
            except requests.RequestException as e:
                error_msg = f"Package upload request failed: {str(e)}"
                self.logger.error(error_msg)
                raise ConnectionError(error_msg)
            finally:
                # Close file handle
                if 'files' in locals() and 'package' in files:
                    files['package'][1].close()
                
        except SyncError as e:
            self.logger.error(f"Error preparing package {package_id}: {str(e)}")
            raise
    
    def sync_all_pending_packages(self) -> List[Dict[str, Any]]:
        """
        Synchronize all pending packages with the server.
        
        Returns:
            List[Dict[str, Any]]: Results for each package sync attempt
        """
        self.logger.info("Synchronizing all pending packages")
        
        # Get all unsynced packages
        packages = self.sync_interface.list_packages(include_synced=False)
        
        if not packages:
            self.logger.info("No pending packages to synchronize")
            return []
            
        self.logger.info(f"Found {len(packages)} pending packages")
        
        # Process each package
        results = []
        for package in packages:
            package_id = package["package_id"]
            result = {
                "package_id": package_id,
                "success": False,
                "error": None
            }
            
            try:
                success = self.send_package(package_id)
                result["success"] = success
            except Exception as e:
                self.logger.error(f"Failed to sync package {package_id}: {str(e)}")
                result["error"] = str(e)
            
            results.append(result)
            
        # Summarize results
        success_count = sum(1 for r in results if r["success"])
        self.logger.info(f"Sync complete: {success_count}/{len(results)} packages successful")
        
        return results
    
    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current synchronization status.
        
        Returns:
            Dict[str, Any]: Synchronization status information
        """
        # Get all packages
        all_packages = self.sync_interface.list_packages(include_synced=True)
        
        # Count by status
        pending = sum(1 for p in all_packages if "sync_status" not in p)
        synced = sum(1 for p in all_packages if p.get("sync_status") == "success")
        failed = sum(1 for p in all_packages if p.get("sync_status") == "error")
        
        # Group by type
        types = {}
        for package in all_packages:
            pkg_type = package.get("package_type", "unknown")
            if pkg_type not in types:
                types[pkg_type] = 0
            types[pkg_type] += 1
        
        return {
            "total_packages": len(all_packages),
            "pending_packages": pending,
            "synced_packages": synced,
            "failed_packages": failed,
            "package_types": types,
            "connection_status": self.connection_manager.connection_status,
            "last_connection_time": self.connection_manager.last_connection_time
        }
    
    def close(self) -> None:
        """
        Clean up resources before shutdown.
        """
        self.logger.info("Closing sync client")
        # Additional cleanup if needed