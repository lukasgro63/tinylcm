"""Training tracking and experiment management for TinyLCM.

Provides functionality for tracking training runs, logging parameters,
metrics and artifacts in a lightweight, MLflow-compatible format.
"""

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tinylcm.constants import (
    DEFAULT_TRAINING_DIR,
    FILE_FORMAT_JSON,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from tinylcm.utils.config import Config, get_config
from tinylcm.utils.file_utils import ensure_dir, load_json, save_json
from tinylcm.utils.logging import setup_logger
from tinylcm.utils.versioning import calculate_file_hash


class TrainingTracker:
    """
    Tracker for machine learning training runs.
    
    Tracks parameters, metrics, and artifacts for model training runs
    in a format that's compatible with MLflow for later synchronization.
    """
    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the training tracker.
        
        Args:
            storage_dir: Directory for storing run data
            config: Configuration object
        """
        self.config = config or get_config()
        component_config = self.config.get_component_config("training_tracker")
        
        # Set up logger
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Storage settings
        self.storage_dir = Path(storage_dir or component_config.get("storage_dir", DEFAULT_TRAINING_DIR))
        self.runs_dir = ensure_dir(self.storage_dir / "runs")
        self.artifacts_dir = ensure_dir(self.storage_dir / "artifacts")
        self.backups_dir = ensure_dir(self.storage_dir / "backups")
        
        # Configuration
        self.log_artifacts = component_config.get("log_artifacts", True)
        
        # State
        self.active_run_id = None
        self.run_stack = []  # For nested runs
        
        self.logger.info(f"Initialized training tracker with storage at: {self.storage_dir}")
    
    def start_run(
        self,
        run_name: str,
        run_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
        resume: bool = False
    ) -> str:
        """
        Start a new training run.
        
        Args:
            run_name: Name of the run
            run_id: Specific run ID (if None, auto-generated)
            description: Optional description
            tags: Optional tags for the run
            nested: If True, this is a child run of the current active run
            resume: If True, try to resume a previous run with the same name
            
        Returns:
            str: Run ID
            
        Raises:
            ValueError: If trying to start a nested run with no active parent
        """
        # If resuming, try to find an existing run with the same name
        if resume:
            existing_runs = self.list_runs(
                filter_func=lambda run: run.get("run_name") == run_name and run.get("status") != STATUS_RUNNING
            )
            if existing_runs:
                # Resume the most recent run
                existing_runs.sort(key=lambda run: run.get("end_time", 0), reverse=True)
                run_id = existing_runs[0]["run_id"]
                self.logger.info(f"Resuming existing run: {run_id}")
        
        # Handle nested runs
        parent_run_id = None
        if nested:
            if not self.active_run_id:
                raise ValueError("Cannot start a nested run with no active parent run")
            parent_run_id = self.active_run_id
            # Save current run to the stack
            self.run_stack.append(self.active_run_id)
        elif self.active_run_id:
            self.logger.warning(f"Starting a new run without ending the previous run: {self.active_run_id}")
            self.end_run()  # Auto-end previous run
        
        # Generate or use provided run ID
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        # Create run directory
        run_dir = ensure_dir(self.runs_dir / run_id)
        
        # Create run metadata
        metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "description": description or "",
            "tags": tags or {},
            "start_time": time.time(),
            "status": STATUS_RUNNING,
            "parent_run_id": parent_run_id
        }
        
        # Save metadata
        metadata_path = run_dir / "metadata.json"
        save_json(metadata, metadata_path)
        
        # Set as active run
        self.active_run_id = run_id
        
        self.logger.info(f"Started run '{run_name}' with ID: {run_id}")
        return run_id
    
    def end_run(
        self,
        run_id: Optional[str] = None,
        status: str = STATUS_COMPLETED
    ) -> bool:
        """
        End a training run.
        
        Args:
            run_id: Run ID to end (if None, uses active run)
            status: Final status (COMPLETED, FAILED, etc.)
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If no run ID provided and no active run
        """
        # Determine which run to end
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run to end")
        
        # Get run directory
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run directory not found: {run_dir}")
            return False
        
        # Update metadata
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            self.logger.warning(f"Run metadata not found: {metadata_path}")
            return False
        
        try:
            metadata = load_json(metadata_path)
            metadata["status"] = status
            metadata["end_time"] = time.time()
            save_json(metadata, metadata_path)
            
            # If this was the active run, clear it
            if run_id == self.active_run_id:
                # If we have a run stack, pop and restore the parent
                if self.run_stack:
                    self.active_run_id = self.run_stack.pop()
                    self.logger.info(f"Restored parent run: {self.active_run_id}")
                else:
                    self.active_run_id = None
            
            self.logger.info(f"Ended run {run_id} with status: {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ending run {run_id}: {str(e)}")
            return False
    
    def log_param(
        self,
        key: str,
        value: Any,
        run_id: Optional[str] = None
    ) -> bool:
        """
        Log a parameter for a run.
        
        Args:
            key: Parameter name
            value: Parameter value
            run_id: Run ID (if None, uses active run)
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If no run ID provided and no active run
        """
        if not key:
            raise ValueError("Parameter key cannot be empty")
        
        # Determine which run to use
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging parameters")
        
        # Get run directory
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run directory not found: {run_dir}")
            return False
        
        # Load existing params or create new
        params_path = run_dir / "params.json"
        if params_path.exists():
            try:
                params = load_json(params_path)
            except Exception:
                params = {}
        else:
            params = {}
        
        # Add or update parameter
        params[key] = value
        
        # Save params
        try:
            save_json(params, params_path)
            self.logger.debug(f"Logged parameter '{key}' for run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging parameter '{key}' for run {run_id}: {str(e)}")
            return False
    
    def log_params(
        self,
        params_dict: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> bool:
        """
        Log multiple parameters for a run.
        
        Args:
            params_dict: Dictionary of parameter names and values
            run_id: Run ID (if None, uses active run)
            
        Returns:
            bool: True if successful
        """
        # Validate
        if not params_dict:
            return True  # Nothing to log
        
        # Determine which run to use
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging parameters")
        
        # Get run directory
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run directory not found: {run_dir}")
            return False
        
        # Load existing params or create new
        params_path = run_dir / "params.json"
        if params_path.exists():
            try:
                existing_params = load_json(params_path)
            except Exception:
                existing_params = {}
        else:
            existing_params = {}
        
        # Update parameters
        existing_params.update(params_dict)
        
        # Save params
        try:
            save_json(existing_params, params_path)
            self.logger.debug(f"Logged {len(params_dict)} parameters for run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging parameters for run {run_id}: {str(e)}")
            return False
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> bool:
        """
        Log a metric for a run.
        
        Args:
            key: Metric name
            value: Metric value (should be numeric)
            step: Optional step number (for tracking progress)
            run_id: Run ID (if None, uses active run)
            
        Returns:
            bool: True if successful
            
        Raises:
            ValueError: If no run ID provided and no active run
        """
        if not key:
            raise ValueError("Metric key cannot be empty")
        
        # Determine which run to use
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging metrics")
        
        # Get run directory
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run directory not found: {run_dir}")
            return False
        
        # Load existing metrics or create new
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = load_json(metrics_path)
            except Exception:
                metrics = {}
        else:
            metrics = {}
        
        # Create metric entry
        metric_entry = {
            "value": value,
            "timestamp": time.time()
        }
        if step is not None:
            metric_entry["step"] = step
        
        # Add to metrics record
        if key not in metrics:
            metrics[key] = []
        metrics[key].append(metric_entry)
        
        # Save metrics
        try:
            save_json(metrics, metrics_path)
            self.logger.debug(f"Logged metric '{key}' with value {value} for run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging metric '{key}' for run {run_id}: {str(e)}")
            return False
    
    def log_metrics(
        self,
        metrics_dict: Dict[str, float],
        step: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> bool:
        """
        Log multiple metrics for a run.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Optional step number (for tracking progress)
            run_id: Run ID (if None, uses active run)
            
        Returns:
            bool: True if successful
        """
        # Validate
        if not metrics_dict:
            return True  # Nothing to log
        
        # Log each metric individually
        success = True
        for key, value in metrics_dict.items():
            result = self.log_metric(key, value, step=step, run_id=run_id)
            success = success and result
        
        return success
    
    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
        description: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> str:
        """
        Log an artifact for a run.
        
        Args:
            local_path: Path to the artifact file
            artifact_path: Path within the artifacts directory (if None, uses filename)
            description: Optional description of the artifact
            run_id: Run ID (if None, uses active run)
            
        Returns:
            str: Path to the stored artifact
            
        Raises:
            ValueError: If no run ID provided and no active run
            FileNotFoundError: If the artifact file is not found
        """
        # Determine which run to use first
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging artifacts")
        
        # Then check if file exists
        path_obj = Path(local_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Artifact file not found: {local_path}")
        
        # Determine artifact destination path
        if artifact_path is None:
            artifact_path = path_obj.name
        
        # Create artifact directory for this run
        run_artifacts_dir = ensure_dir(self.artifacts_dir / run_id)
        
        # Create destination path
        dest_path = run_artifacts_dir / artifact_path
        ensure_dir(dest_path.parent)
        
        # Copy the artifact
        try:
            shutil.copy2(path_obj, dest_path)
            
            # Update artifacts list
            self._update_artifacts_list(
                run_id=run_id,
                artifact_name=artifact_path,
                artifact_path=str(dest_path),
                artifact_type="file",
                description=description
            )
            
            self.logger.debug(f"Logged artifact '{artifact_path}' for run {run_id}")
            return str(dest_path)
        except Exception as e:
            self.logger.error(f"Error logging artifact '{artifact_path}' for run {run_id}: {str(e)}")
            raise
    
    def log_figure(
        self,
        figure_data: bytes,
        name: str,
        description: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> str:
        """
        Log a figure artifact for a run.
        
        Args:
            figure_data: Figure data as bytes
            name: Figure name with extension (e.g., "plot.png")
            description: Optional description of the figure
            run_id: Run ID (if None, uses active run)
            
        Returns:
            str: Path to the stored figure
            
        Raises:
            ValueError: If no run ID provided and no active run
        """
        # Determine which run to use
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging figures")
        
        # Create artifact directory for this run
        run_artifacts_dir = ensure_dir(self.artifacts_dir / run_id / "figures")
        
        # Create destination path
        dest_path = run_artifacts_dir / name
        ensure_dir(dest_path.parent)
        
        # Write the figure data
        try:
            with open(dest_path, "wb") as f:
                f.write(figure_data)
            
            # Update artifacts list
            self._update_artifacts_list(
                run_id=run_id,
                artifact_name=name,
                artifact_path=str(dest_path),
                artifact_type="figure",
                description=description
            )
            
            self.logger.debug(f"Logged figure '{name}' for run {run_id}")
            return str(dest_path)
        except Exception as e:
            self.logger.error(f"Error logging figure '{name}' for run {run_id}: {str(e)}")
            raise
    
    def log_model(
        self,
        model_path: Union[str, Path],
        model_format: str,
        flavor: str = "custom",
        custom_properties: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None
    ) -> str:
        """
        Log a model artifact for a run.
        
        Args:
            model_path: Path to the model file
            model_format: Format of the model (e.g., "tflite", "onnx", "pytorch")
            flavor: Model flavor (e.g., "tensorflow", "pytorch", "custom")
            custom_properties: Additional properties for the model
            run_id: Run ID (if None, uses active run)
            
        Returns:
            str: Path to the stored model directory
            
        Raises:
            ValueError: If no run ID provided and no active run
            FileNotFoundError: If the model file is not found
        """
        # Check if file exists
        path_obj = Path(model_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Determine which run to use
        if run_id is None:
            run_id = self.active_run_id
            if run_id is None:
                raise ValueError("No active run for logging models")
        
        # Create model directory for this run
        run_models_dir = ensure_dir(self.artifacts_dir / run_id / "models")
        model_dir = ensure_dir(run_models_dir / path_obj.stem)
        
        # Copy the model file
        dest_path = model_dir / path_obj.name
        try:
            shutil.copy2(path_obj, dest_path)
            
            # Create model metadata
            model_meta = {
                "format": model_format,
                "flavor": flavor,
                "filename": path_obj.name,
                "timestamp": time.time(),
                "hash": calculate_file_hash(dest_path),
                "custom_properties": custom_properties or {}
            }
            
            # Save model metadata
            meta_path = model_dir / "model_info.json"
            save_json(model_meta, meta_path)
            
            # Update artifacts list
            self._update_artifacts_list(
                run_id=run_id,
                artifact_name=path_obj.name,
                artifact_path=str(model_dir),
                artifact_type="model",
                description=f"{flavor} model in {model_format} format"
            )
            
            self.logger.debug(f"Logged model '{path_obj.name}' for run {run_id}")
            return str(model_dir)
        except Exception as e:
            self.logger.error(f"Error logging model '{path_obj.name}' for run {run_id}: {str(e)}")
            raise
    
    def _update_artifacts_list(
        self,
        run_id: str,
        artifact_name: str,
        artifact_path: str,
        artifact_type: str,
        description: Optional[str] = None
    ) -> None:
        """
        Update the list of artifacts for a run.
        
        Args:
            run_id: Run ID
            artifact_name: Name of the artifact
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (file, figure, model)
            description: Optional description
        """
        # Get run directory
        run_dir = self.runs_dir / run_id
        
        # Load existing artifacts list or create new
        artifacts_path = run_dir / "artifacts.json"
        if artifacts_path.exists():
            try:
                artifacts = load_json(artifacts_path)
            except Exception:
                artifacts = []
        else:
            artifacts = []
        
        # Create artifact entry
        artifact_entry = {
            "name": artifact_name,
            "path": artifact_path,
            "type": artifact_type,
            "timestamp": time.time()
        }
        
        if description:
            artifact_entry["description"] = description
        
        # Add to artifacts list
        artifacts.append(artifact_entry)
        
        # Save artifacts list
        save_json(artifacts, artifacts_path)
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about a run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Dict[str, Any]: Run information
            
        Raises:
            ValueError: If run not found
        """
        # Get run directory
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run not found: {run_id}")
        
        # Load metadata
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Run metadata not found: {run_id}")
        
        metadata = load_json(metadata_path)
        
        # Load parameters if available
        params_path = run_dir / "params.json"
        if params_path.exists():
            try:
                params = load_json(params_path)
                metadata["params"] = params
            except Exception:
                metadata["params"] = {}
        else:
            metadata["params"] = {}
        
        # Load metrics if available
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = load_json(metrics_path)
                metadata["metrics"] = metrics
            except Exception:
                metadata["metrics"] = {}
        else:
            metadata["metrics"] = {}
        
        # Load artifacts if available
        artifacts_path = run_dir / "artifacts.json"
        if artifacts_path.exists():
            try:
                artifacts = load_json(artifacts_path)
                metadata["artifacts"] = artifacts
            except Exception:
                metadata["artifacts"] = []
        else:
            metadata["artifacts"] = []
        
        return metadata
    
    def list_runs(
        self,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        List all runs.
        
        Args:
            filter_func: Optional function to filter runs
            
        Returns:
            List[Dict[str, Any]]: List of run information
        """
        runs = []
        
        # Iterate through run directories
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            # Load metadata
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                metadata = load_json(metadata_path)
                
                # Apply filter if provided
                if filter_func is None or filter_func(metadata):
                    runs.append(metadata)
            except Exception as e:
                self.logger.warning(f"Error loading metadata for run {run_dir.name}: {str(e)}")
        
        return runs
    
    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run.
        
        Args:
            run_id: Run ID
            
        Returns:
            bool: True if deleted successfully
        """
        # Get run directory
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return False
        
        # Delete run directory
        try:
            shutil.rmtree(run_dir)
            
            # Also delete artifacts if they exist
            run_artifacts_dir = self.artifacts_dir / run_id
            if run_artifacts_dir.exists():
                shutil.rmtree(run_artifacts_dir)
            
            self.logger.info(f"Deleted run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting run {run_id}: {str(e)}")
            return False
    
    def backup_run(self, run_id: str) -> bool:
        """
        Backup a run.
        
        Args:
            run_id: Run ID
            
        Returns:
            bool: True if backup was successful
        """
        # Get run directory
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            self.logger.warning(f"Run not found for backup: {run_id}")
            return False
        
        # Create backup directory
        backup_dir = ensure_dir(self.backups_dir / run_id)
        
        try:
            # Copy run directory contents
            for item in run_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, backup_dir)
            
            # Copy artifacts if they exist
            run_artifacts_dir = self.artifacts_dir / run_id
            if run_artifacts_dir.exists():
                backup_artifacts_dir = ensure_dir(backup_dir / "artifacts")
                shutil.copytree(run_artifacts_dir, backup_artifacts_dir, dirs_exist_ok=True)
            
            self.logger.info(f"Backed up run {run_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error backing up run {run_id}: {str(e)}")
            return False
    
    def restore_run(self, run_id: str) -> bool:
        """
        Restore a run from backup.
        
        Args:
            run_id: Run ID
            
        Returns:
            bool: True if restore was successful
        """
        # Get backup directory
        backup_dir = self.backups_dir / run_id
        if not backup_dir.exists():
            self.logger.warning(f"Backup not found for run: {run_id}")
            return False
        
        try:
            # Create run directory
            run_dir = ensure_dir(self.runs_dir / run_id)
            
            # Copy backup files
            for item in backup_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, run_dir)
                elif item.name == "artifacts":
                    # Restore artifacts
                    shutil.copytree(item, self.artifacts_dir / run_id, dirs_exist_ok=True)
            
            self.logger.info(f"Restored run {run_id} from backup")
            return True
        except Exception as e:
            self.logger.error(f"Error restoring run {run_id}: {str(e)}")
            return False
    
    def export_to_mlflow_format(
        self,
        run_id: str,
        output_dir: Union[str, Path]
    ) -> bool:
        """
        Export a run to MLflow format.
        
        Args:
            run_id: Run ID
            output_dir: Directory for MLflow output
            
        Returns:
            bool: True if export was successful
        """
        # Get run info
        try:
            run_info = self.get_run_info(run_id)
        except ValueError as e:
            self.logger.error(f"Error getting run info: {str(e)}")
            return False
        
        # Create MLflow directory structure
        mlflow_run_dir = ensure_dir(Path(output_dir) / run_id)
        mlflow_params_dir = ensure_dir(mlflow_run_dir / "params")
        mlflow_metrics_dir = ensure_dir(mlflow_run_dir / "metrics")
        mlflow_artifacts_dir = ensure_dir(mlflow_run_dir / "artifacts")
        
        try:
            # Create meta.yaml
            meta = {
                "name": run_info.get("run_name", ""),
                "tags": run_info.get("tags", {}),
                "status": run_info.get("status", ""),
                "start_time": run_info.get("start_time", 0),
                "end_time": run_info.get("end_time", 0),
                "tinylcm.run_id": run_id
            }
            
            with open(mlflow_run_dir / "meta.yaml", "w") as f:
                f.write(yaml_format(meta))
            
            # Export parameters
            params = run_info.get("params", {})
            for param_name, param_value in params.items():
                # MLflow stores parameters as separate files with the value as content
                with open(mlflow_params_dir / param_name, "w") as f:
                    f.write(str(param_value))
            
            # Export metrics
            metrics = run_info.get("metrics", {})
            for metric_name, metric_values in metrics.items():
                for i, metric_entry in enumerate(metric_values):
                    # MLflow stores metrics as separate files with step and timestamp
                    timestamp = metric_entry.get("timestamp", 0)
                    step = metric_entry.get("step", i)
                    value = metric_entry.get("value", 0)
                    
                    metric_filename = f"{metric_name}-{timestamp:.0f}"
                    with open(mlflow_metrics_dir / metric_filename, "w") as f:
                        f.write(f"{value} {step} {timestamp:.0f}")
            
            # Export artifacts
            artifacts = run_info.get("artifacts", [])
            for artifact in artifacts:
                src_path = artifact.get("path")
                if not src_path:
                    continue
                
                name = artifact.get("name", "")
                dest_path = mlflow_artifacts_dir / name
                
                # Copy the artifact
                src_path_obj = Path(src_path)
                if src_path_obj.is_file():
                    ensure_dir(dest_path.parent)
                    shutil.copy2(src_path_obj, dest_path)
                elif src_path_obj.is_dir():
                    shutil.copytree(src_path_obj, dest_path, dirs_exist_ok=True)
            
            self.logger.info(f"Exported run {run_id} to MLflow format at {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting run {run_id} to MLflow format: {str(e)}")
            return False
    
    def close(self) -> None:
        """
        Close the training tracker and clean up resources.
        
        This method ensures all runs are properly ended before closing.
        """
        # End active run if there is one
        if self.active_run_id:
            self.logger.info(f"Auto-ending active run during close: {self.active_run_id}")
            self.end_run(status=STATUS_COMPLETED)  # Assume completed
        
        # End any runs in the stack (should not happen, but just in case)
        while self.run_stack:
            run_id = self.run_stack.pop()
            self.logger.warning(f"Auto-ending stacked run during close: {run_id}")
            self.end_run(run_id=run_id, status=STATUS_COMPLETED)
        
        self.logger.info("Closed training tracker")


def yaml_format(data: Dict[str, Any]) -> str:
    """
    Format dictionary as YAML string.
    
    This is a simple implementation for meta.yaml export.
    For a full implementation, consider using PyYAML.
    
    Args:
        data: Dictionary to format
        
    Returns:
        str: YAML-formatted string
    """
    lines = []
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    
    return "\n".join(lines)