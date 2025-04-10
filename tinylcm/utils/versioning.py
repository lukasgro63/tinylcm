"""Versioning utilities for model and data lifecycle management.

Provides tools for generating version identifiers, calculating content hashes,
and comparing different versions of models or data artifacts.
"""

import datetime
import hashlib
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Protocol, cast

from tinylcm.utils.file_utils import get_file_size


class VersionInfo(Protocol):
    """Protocol defining the structure of version information."""

    @property
    def version_id(self) -> str:
        """Get the version identifier."""
        ...

    @property
    def timestamp(self) -> float:
        """Get the timestamp when the version was created."""
        ...

def generate_timestamp_version(prefix: str = "v_") -> str:
    """
    Generate a version string based on current timestamp.

    Format: v_YYYYMMDD_HHMMSS

    Args:
        prefix: Prefix to use for the version string

    Returns:
        str: Timestamp-based version string
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{timestamp}"


def generate_incremental_version(
    directory: Union[str, Path],
    prefix: str = "v_",
    digits: int = 3
) -> str:
    """
    Generate an incremental version number by examining existing versions.

    Finds the highest existing version number in the directory and increments it.

    Args:
        directory: Directory to search for existing versions
        prefix: Prefix to use for version numbers
        digits: Number of digits to use for the version number (only used for v_ prefix)

    Returns:
        str: Next version string in sequence
    """
    dir_path = Path(directory)

    # Ensure directory exists
    if not dir_path.exists():
        os.makedirs(dir_path, exist_ok=True)
        # Special case for default prefix
        if prefix == "v_":
            return f"{prefix}{1:0{digits}d}"
        else:
            return f"{prefix}1"

    # Find all existing versions with this prefix
    pattern = re.compile(f"^{re.escape(prefix)}(\\d+)$")

    max_version = 0
    for item in os.listdir(dir_path):
        match = pattern.match(item)
        if match:
            try:
                version_num = int(match.group(1))
                max_version = max(max_version, version_num)
            except ValueError:
                continue

    # Increment the highest version found
    next_version = max_version + 1

    # Format depends on the prefix type
    if prefix == "v_":
        return f"{prefix}{next_version:0{digits}d}"
    else:
        # For custom prefixes like "model_", don't use leading zeros
        return f"{prefix}{next_version}"

def calculate_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "md5",
    buffer_size: int = 65536
) -> str:
    """
    Calculate hash for a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
        buffer_size: Size of buffer for reading file in chunks

    Returns:
        str: Hexadecimal hash string

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If algorithm is not supported
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Create appropriate hash object
    if algorithm == "md5":
        hash_obj = hashlib.md5()
    elif algorithm == "sha1":
        hash_obj = hashlib.sha1()
    elif algorithm == "sha256":
        hash_obj = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    # Calculate hash in chunks to handle large files
    with open(path_obj, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hash_obj.update(data)

    return hash_obj.hexdigest()


def calculate_content_hash(content: Union[str, bytes], algorithm: str = "md5") -> str:
    """
    Calculate hash for content (string or bytes).

    Args:
        content: Content to hash
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')

    Returns:
        str: Hexadecimal hash string

    Raises:
        ValueError: If algorithm is not supported
    """
    # Create appropriate hash object
    if algorithm == "md5":
        hash_obj = hashlib.md5()
    elif algorithm == "sha1":
        hash_obj = hashlib.sha1()
    elif algorithm == "sha256":
        hash_obj = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    # Convert string to bytes if needed
    if isinstance(content, str):
        content_bytes = content.encode('utf-8')
    else:
        content_bytes = content

    hash_obj.update(content_bytes)
    return hash_obj.hexdigest()


def create_version_info(
    source_file: Optional[Union[str, Path]] = None,
    content: Optional[Union[str, bytes]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    version_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a version info dictionary for a file or content.

    Either source_file or content must be provided.

    Args:
        source_file: Path to source file (optional)
        content: Content as string or bytes (optional)
        metadata: Additional metadata to include
        version_id: Custom version ID (if None, a UUID will be generated)

    Returns:
        Dict[str, Any]: Version info dictionary

    Raises:
        ValueError: If neither source_file nor content is provided
    """
    if source_file is None and content is None:
        raise ValueError("Either source_file or content must be provided")

    # Base version info
    version_info = {
        "version_id": version_id or str(uuid.uuid4()),
        "timestamp": time.time(),
        "metadata": metadata or {}
    }

    # Add file-specific info if provided
    if source_file is not None:
        path_obj = Path(source_file)
        version_info.update({
            "filename": path_obj.name,
            "file_size_bytes": get_file_size(path_obj),
            "file_hash": calculate_file_hash(path_obj)
        })

    # Add content-specific info if provided
    if content is not None:
        content_size = len(content.encode('utf-8') if isinstance(content, str) else content)
        version_info.update({
            "content_hash": calculate_content_hash(content),
            "content_size_bytes": content_size
        })

    return version_info


def compare_versions(version1: Dict[str, Any], version2: Dict[str, Any]) -> bool:
    """
    Compare two version info dictionaries to check if they represent the same content.

    Comparison is based on file_hash if present, otherwise on content_hash.

    Args:
        version1: First version info
        version2: Second version info

    Returns:
        bool: True if versions represent the same content
    """
    # Compare file hashes if available
    if "file_hash" in version1 and "file_hash" in version2:
        return version1["file_hash"] == version2["file_hash"]

    # Compare content hashes if available
    if "content_hash" in version1 and "content_hash" in version2:
        return version1["content_hash"] == version2["content_hash"]

    # If one has file_hash and the other has content_hash, they're different
    return False


def get_version_diff(
    old_version: Dict[str, Any],
    new_version: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate differences between two versions.

    Args:
        old_version: Original version info
        new_version: New version info

    Returns:
        Dict[str, Any]: Dictionary of differences
    """
    diff = {
        "is_same_content": compare_versions(old_version, new_version),
        "time_difference_seconds": new_version.get("timestamp", 0) - old_version.get("timestamp", 0)
    }

    # Check size differences if applicable
    if "file_size_bytes" in old_version and "file_size_bytes" in new_version:
        diff["size_difference_bytes"] = new_version["file_size_bytes"] - old_version["file_size_bytes"]

    if "content_size_bytes" in old_version and "content_size_bytes" in new_version:
        diff["content_size_difference_bytes"] = (
            new_version["content_size_bytes"] - old_version["content_size_bytes"]
        )

    # Compare metadata fields
    old_metadata = old_version.get("metadata", {})
    new_metadata = new_version.get("metadata", {})

    metadata_changes = {}
    all_keys = set(old_metadata.keys()) | set(new_metadata.keys())

    for key in all_keys:
        old_value = old_metadata.get(key)
        new_value = new_metadata.get(key)

        if old_value != new_value:
            metadata_changes[key] = {
                "from": old_value,
                "to": new_value
            }

    if metadata_changes:
        diff["metadata_changes"] = metadata_changes

    return diff
