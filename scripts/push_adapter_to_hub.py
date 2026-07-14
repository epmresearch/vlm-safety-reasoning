"""
Pushes a trained LoRA adapter from a local/Drive path to the HF Hub.
Usage:
    python scripts/push_adapter_to_hub.py --adapter_path <path> --hub_repo <org>/<repo-name>
"""
import argparse
import os
from huggingface_hub import HfApi, create_repo

from core.logging import get_logger

logger = get_logger(__name__)


def push_adapter(adapter_path: str, hub_repo: str, private: bool = True) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set in environment.")

    api = HfApi(token=token)
    create_repo(hub_repo, token=token, private=private, exist_ok=True)

    logger.info(f"Uploading {adapter_path} -> {hub_repo} (private={private})")
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=hub_repo,
        repo_type="model",
    )
    logger.info("Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--hub_repo", required=True)
    parser.add_argument("--public", action="store_false", dest="private", help="Make repo public (default is private)")
    args = parser.parse_args()

    push_adapter(args.adapter_path, args.hub_repo, args.private)