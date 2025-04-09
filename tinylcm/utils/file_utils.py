"""Utility functions for file operations.

This module provides consistent interfaces for common file operations,
with type checking, error handling, and support for both string and Path objects.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, Protocol, overload

# Type aliases
PathLike = Union[str, Path]
T = TypeVar('T')  # Generic type for type hinting


def ensure_dir(dir_path: PathLike) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to the directory to ensure exists.
    
    Returns:
        Path: Path object for the directory.
    """
    path_obj = Path(dir_path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_size(file_path: PathLike) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file.
    
    Returns:
        int: Size of the file in bytes.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path_obj.stat().st_size


@overload
def load_json(file_path: PathLike) -> Dict[str, Any]: ...

@overload
def load_json(file_path: PathLike, default: T) -> Union[Dict[str, Any], T]: ...


def load_json(file_path: PathLike, default: Optional[T] = None) -> Union[Dict[str, Any], T]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file.
        default: Default value to return if file not found.
    
    Returns:
        Dict or default: Parsed JSON data as a dictionary, or default if file not found and default provided.
    
    Raises:
        FileNotFoundError: If the file does not exist and no default is provided.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with path_obj.open('r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: PathLike, pretty: bool = True) -> None:
    """
    Save data as JSON to a file.
    
    Args:
        data: Data to save as JSON.
        file_path: Path where to save the JSON file.
        pretty: If True, format the JSON with indentation for readability.
    """
    path_obj = Path(file_path)
    
    # Ensure parent directory exists
    ensure_dir(path_obj.parent)
    
    with path_obj.open('w') as f:
        if pretty:
            json.dump(data, f, indent=2, sort_keys=True)
        else:
            json.dump(data, f)


def list_files(
    directory: PathLike, 
    pattern: str = "*",
    recursive: bool = False,
    absolute: bool = False
) -> List[Path]:
    """
    List files in a directory that match a pattern.
    
    Args:
        directory: Directory to search.
        pattern: Glob pattern for matching files.
        recursive: If True, search recursively in subdirectories.
        absolute: If True, return absolute paths.
    
    Returns:
        List[Path]: List of Path objects for the matching files.
    """
    path_obj = Path(directory)
    
    if recursive:
        matched_files = list(path_obj.glob(f"**/{pattern}"))
    else:
        matched_files = list(path_obj.glob(pattern))
    
    # Filter out directories
    matched_files = [f for f in matched_files if f.is_file()]
    
    if absolute:
        matched_files = [f.absolute() for f in matched_files]
    
    return matched_files