#!/usr/bin/env python3
"""
Client synchronization example for TinyLCM.

This example demonstrates how to use the SyncClient to synchronize
data between an edge device and a central TinyLCM server.
"""

import argparse
import json
import os
import time
from pathlib import Path

import tinylcm
from tinylcm.client import SyncClient
from tinylcm.core.sync_interface import SyncInterface
from tinylcm.utils.errors import ConnectionError, SyncError


def setup_sync_client(args):
    """
    Set up the sync client based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        SyncClient: Configured sync client
    """
    print(f"Setting up sync client for server: {args.server_url}")
    
    # Create work directory if it doesn't exist
    work_dir = Path(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    
    # Create sync interface
    sync_dir = work_dir / "sync"
    sync_interface = SyncInterface(sync_dir=sync_dir)
    
    # Create sync client
    client = SyncClient(
        server_url=args.server_url,
        api_key=args.api_key,
        device_id=args.device_id,
        sync_interface=sync_interface,
        max_retries=args.max_retries,
        auto_register=True
    )
    
    return client


def check_server_status(client):
    """
    Check and display server status.
    
    Args:
        client: SyncClient instance
    """
    print("\nChecking server status...")
    
    try:
        status = client.check_server_status()
        print(f"Server status: {status['status']}")
        print(f"Server version: {status.get('version', 'unknown')}")
        print(f"Server time: {status.get('server_time', 'unknown')}")
    except ConnectionError as e:
        print(f"Error checking server status: {e}")


def create_and_send_package(client, sync_interface, package_type, content_file=None):
    """
    Create and send a test package.
    
    Args:
        client: SyncClient instance
        sync_interface: SyncInterface instance
        package_type: Type of package to create
        content_file: Optional file to include in package
        
    Returns:
        str: Package ID if successful, None otherwise
    """
    print(f"\nCreating {package_type} package...")
    
    # Create a package
    package_id = sync_interface.create_package(
        device_id=client.device_id,
        package_type=package_type,
        description=f"Test {package_type} package",
        compression="gzip"
    )
    
    # Create a test file if none provided
    if content_file is None:
        test_data = {
            "timestamp": time.time(),
            "package_type": package_type,
            "test_data": [1, 2, 3, 4, 5],
            "device_id": client.device_id
        }
        
        temp_dir = Path(sync_interface.sync_dir) / "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        content_file = temp_dir / f"test_data_{int(time.time())}.json"
        with open(content_file, "w") as f:
            json.dump(test_data, f, indent=2)
    
    # Add the file to the package
    sync_interface.add_file_to_package(
        package_id=package_id,
        file_path=content_file,
        file_type="test_data"
    )
    
    # Finalize the package
    sync_interface.finalize_package(package_id)
    print(f"Created package with ID: {package_id}")
    
    # Send the package
    print("Sending package to server...")
    try:
        result = client.send_package(package_id)
        if result:
            print(f"Successfully sent package {package_id}")
            return package_id
        else:
            print(f"Failed to send package {package_id}")
            return None
    except (ConnectionError, SyncError) as e:
        print(f"Error sending package: {e}")
        return None


def sync_all_packages(client):
    """
    Synchronize all pending packages.
    
    Args:
        client: SyncClient instance
    """
    print("\nSynchronizing all pending packages...")
    
    try:
        results = client.sync_all_pending_packages()
        
        if not results:
            print("No pending packages to synchronize")
            return
        
        success_count = sum(1 for r in results if r["success"])
        print(f"Synchronized {success_count}/{len(results)} packages")
        
        for result in results:
            status = "✓" if result["success"] else "✗"
            pkg_id = result["package_id"]
            error = f" - Error: {result['error']}" if result["error"] else ""
            print(f"  {status} {pkg_id}{error}")
            
    except ConnectionError as e:
        print(f"Error synchronizing packages: {e}")


def show_sync_status(client):
    """
    Display synchronization status.
    
    Args:
        client: SyncClient instance
    """
    print("\nSynchronization status:")
    
    status = client.get_sync_status()
    
    print(f"Total packages: {status['total_packages']}")
    print(f"Pending: {status['pending_packages']}")
    print(f"Synced: {status['synced_packages']}")
    print(f"Failed: {status['failed_packages']}")
    
    print("\nPackage types:")
    for pkg_type, count in status['package_types'].items():
        print(f"  {pkg_type}: {count}")
    
    print(f"\nConnection status: {status['connection_status']}")
    if status['last_connection_time']:
        last_conn = time.strftime('%Y-%m-%d %H:%M:%S', 
                                  time.localtime(status['last_connection_time']))
        print(f"Last connection: {last_conn}")


def connection_status_callback(status, info=None):
    """
    Callback for connection status changes.
    
    Args:
        status: Connection status
        info: Additional information
    """
    status_emoji = {
        "connected": "🟢",
        "disconnected": "⚪",
        "error": "🔴",
        "retrying": "🟡"
    }
    
    emoji = status_emoji.get(status, "⚪")
    info_text = f" - {info}" if info else ""
    
    print(f"\n{emoji} Connection status: {status}{info_text}")


def main():
    """Run the client example."""
    parser = argparse.ArgumentParser(description="TinyLCM Client Example")
    
    parser.add_argument("--server-url", default="http://localhost:8000/api",
                        help="URL of the TinyLCM server")
    parser.add_argument("--api-key", default="test_key",
                        help="API key for authentication")
    parser.add_argument("--device-id", default=f"device_{int(time.time())}",
                        help="Unique device identifier")
    parser.add_argument("--work-dir", default="tinylcm_client_example",
                        help="Working directory for the example")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum connection retry attempts")
    
    args = parser.parse_args()
    
    print("TinyLCM Client Example")
    print(f"Version: {tinylcm.__version__}")
    print("=" * 50)
    
    try:
        # Set up client
        client = setup_sync_client(args)
        
        # Register status callback
        client.connection_manager.register_status_callback(connection_status_callback)
        
        # Check server status
        check_server_status(client)
        
        # Create and send a test package
        create_and_send_package(client, client.sync_interface, "test_data")
        
        # Create another package but don't send it
        create_and_send_package(client, client.sync_interface, "logs")
        
        # Show sync status
        show_sync_status(client)
        
        # Sync all pending packages
        sync_all_packages(client)
        
        # Show final status
        show_sync_status(client)
        
        print("\nExample completed successfully.")
        
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()