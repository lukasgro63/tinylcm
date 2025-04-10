"""Utility functions for file operations.

This module provides consistent interfaces for common file operations,
with type checking, error handling, and support for both string and Path objects.
"""

import json
import logging # Use logging
import os
import shutil # Import shutil for rmtree
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, Protocol, overload

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type aliases
PathLike = Union[str, Path]
T = TypeVar('T')  # Generic type for type hinting


def ensure_dir(dir_path: PathLike) -> Path:
    """
    Ensure a directory exists, creating it and its parents if necessary.

    Args:
        dir_path: Path to the directory (string or Path object).

    Returns:
        Path: Path object for the directory.

    Raises:
        OSError: If the directory creation fails for reasons other than
                 it already existing (e.g., permissions).
    """
    path_obj = Path(dir_path)
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path_obj}")
    except OSError as e:
        logger.error(f"Failed to create or access directory {path_obj}: {e}")
        raise # Re-raise the error after logging
    return path_obj


def get_file_size(file_path: PathLike) -> int:
    """
    Get the size of a file in bytes.

    Args:
        file_path: Path to the file (string or Path object).

    Returns:
        int: Size of the file in bytes.

    Raises:
        FileNotFoundError: If the file does not exist or is not a file.
    """
    path_obj = Path(file_path)
    if not path_obj.is_file(): # Check specifically for a file
        raise FileNotFoundError(f"File not found or not a regular file: {file_path}")
    try:
        size = path_obj.stat().st_size
        logger.debug(f"Size of file {path_obj}: {size} bytes")
        return size
    except OSError as e:
         logger.error(f"Could not get size for file {path_obj}: {e}")
         # Re-raise as FileNotFoundError as the file might be inaccessible
         raise FileNotFoundError(f"Could not access file stats for: {file_path}") from e


@overload
def load_json(file_path: PathLike) -> Dict[str, Any]: ...

@overload
def load_json(file_path: PathLike, default: T) -> Union[Dict[str, Any], T]: ...


def load_json(file_path: PathLike, default: Optional[T] = None) -> Union[Dict[str, Any], T]:
    """
    Load JSON data from a file.

    Args:
        file_path: Path to the JSON file (string or Path object).
        default: Default value to return if file not found or is empty/invalid JSON.
                 If None (default), raises errors instead of returning default.

    Returns:
        Parsed JSON data as a dictionary, or the default value if provided and an
        error occurs (FileNotFoundError, JSONDecodeError, EmptyFile).

    Raises:
        FileNotFoundError: If the file does not exist and no default is provided.
        json.JSONDecodeError: If the file contains invalid JSON and no default is provided.
        ValueError: If the file is empty and no default is provided.
    """
    path_obj = Path(file_path)
    if not path_obj.is_file():
        if default is not None:
            logger.warning(f"JSON file not found at {path_obj}, returning default.")
            return default
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with path_obj.open('r', encoding='utf-8') as f:
            # Handle empty file case explicitly
            content = f.read()
            if not content:
                 if default is not None:
                      logger.warning(f"JSON file is empty at {path_obj}, returning default.")
                      return default
                 raise ValueError(f"JSON file is empty: {file_path}")
            # Attempt to parse
            data = json.loads(content)
            logger.debug(f"Successfully loaded JSON from: {path_obj}")
            return data
    except json.JSONDecodeError as e:
        if default is not None:
             logger.warning(f"Invalid JSON in file {path_obj} (Error: {e}), returning default.")
             return default
        logger.error(f"Failed to decode JSON from {path_obj}: {e}")
        raise # Re-raise the original error
    except OSError as e: # Catch potential read errors
         if default is not None:
              logger.warning(f"Could not read file {path_obj} (Error: {e}), returning default.")
              return default
         logger.error(f"Error reading file {path_obj}: {e}")
         raise FileNotFoundError(f"Could not read file: {file_path}") from e


def save_json(data: Dict[str, Any], file_path: PathLike, pretty: bool = True) -> None:
    """
    Save data as JSON to a file.

    Ensures the parent directory exists before writing.

    Args:
        data: Data dictionary to save as JSON.
        file_path: Path where to save the JSON file (string or Path object).
        pretty: If True (default), format JSON with indentation for readability.

    Raises:
        TypeError: If the data is not JSON serializable.
        OSError: If the file cannot be written (e.g., permissions).
    """
    path_obj = Path(file_path)

    # Ensure parent directory exists first
    ensure_dir(path_obj.parent) # ensure_dir handles potential errors

    try:
        with path_obj.open('w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        logger.debug(f"Successfully saved JSON to: {path_obj}")
    except TypeError as e:
         logger.error(f"Data for {path_obj} is not JSON serializable: {e}", exc_info=True)
         raise
    except OSError as e:
         logger.error(f"Failed to write JSON to {path_obj}: {e}")
         raise


def list_files(
    directory: PathLike,
    pattern: str = "*",
    recursive: bool = False,
    absolute: bool = False
) -> List[Path]:
    """
    List files in a directory matching a glob pattern.

    Args:
        directory: Directory to search (string or Path object).
        pattern: Glob pattern for matching filenames (e.g., "*.txt", "data_*").
        recursive: If True, search recursively into subdirectories (using "**").
        absolute: If True, return absolute paths; otherwise, returns paths
                  relative to the input directory Path object.

    Returns:
        List[Path]: List of Path objects for the matching files. Returns empty list
                    if directory doesn't exist or no files match.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
         logger.warning(f"Directory not found for listing files: {dir_path}")
         return []

    glob_pattern = f"**/{pattern}" if recursive else pattern
    logger.debug(f"Listing files in '{dir_path}' matching '{glob_pattern}' (Recursive={recursive})")

    try:
        matched_items = list(dir_path.glob(glob_pattern))
        # Filter out directories, keeping only files
        matched_files = [item for item in matched_items if item.is_file()]
    except Exception as e:
         logger.error(f"Error listing files in {dir_path} with pattern {glob_pattern}: {e}", exc_info=True)
         return []

    # Convert to absolute paths if requested
    if absolute:
        result_paths = [f.resolve() for f in matched_files] # Use resolve for absolute path
    else:
        # Return paths relative to the starting directory object 'dir_path' if possible,
        # otherwise keep the paths as returned by glob (which might mix absolute/relative depending on input)
        # For simplicity and consistency, let's return the paths as yielded by glob if not absolute.
        result_paths = matched_files

    logger.debug(f"Found {len(result_paths)} matching files.")
    return result_paths


def safe_remove(path: PathLike) -> bool:
    """
    Safely removes a file, symbolic link, or directory tree.

    Logs warnings on errors but does not raise exceptions unless critical.

    Args:
        path: Path to the item to remove (string or Path object).

    Returns:
        bool: True if removal was successful OR if the item did not exist initially.
              False if an error occurred during removal of an existing item.
    """
    path_obj = Path(path)
    try:
        if path_obj.is_symlink():
            logger.debug(f"Removing symbolic link: {path_obj}")
            path_obj.unlink()
        elif path_obj.is_file():
            logger.debug(f"Removing file: {path_obj}")
            path_obj.unlink()
        elif path_obj.is_dir():
            logger.debug(f"Removing directory tree: {path_obj}")
            shutil.rmtree(path_obj)
        else:
            # Path doesn't exist, consider it successfully "removed" in terms of state
            logger.debug(f"Item to remove does not exist: {path_obj}")
            return True
        # If unlink/rmtree succeeded without error
        logger.info(f"Successfully removed: {path_obj}")
        return True
    except OSError as e:
        # Log the specific error encountered during removal
        logger.warning(f"Could not remove {path_obj}: {e}")
        return False
    except Exception as e: # Catch unexpected errors during removal
         logger.error(f"Unexpected error removing {path_obj}: {e}", exc_info=True)
         return False
