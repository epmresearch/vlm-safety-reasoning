"""
Utility for saving run manifests (experiment receipts).
Captures configuration and environment state to ensure reproducibility.
"""
import os
import json
from datetime import datetime, timezone
from typing import Dict, Any

from core.logging import get_logger
from core.io import ensure_dir

logger = get_logger(__name__)

def save_run_manifest(output_dir: str, config_dict: Dict[str, Any], filename: str = "run_manifest.json") -> str:
    """
    Saves a snapshot of the current experiment configuration.
    
    Args:
        output_dir: Directory where the manifest should be saved.
        config_dict: Dictionary containing hyperparameters, model paths, prompt info, etc.
        filename: Name of the manifest file.
        
    Returns:
        The path to the saved manifest file.
    """
    ensure_dir(output_dir)
    manifest_path = os.path.join(output_dir, filename)
    
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_dict
    }
    
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"Run manifest saved to {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to save run manifest to {manifest_path}: {e}")
        
    return manifest_path
