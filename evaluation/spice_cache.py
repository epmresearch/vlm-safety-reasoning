"""
Caches the Stanford CoreNLP models that pycocoevalcap's SPICE metric
downloads on first use (~2GB) into a Drive folder, so subsequent Colab
sessions restore them instead of re-downloading.

What gets cached: the `lib/` folder inside the installed pycocoevalcap
package's `spice/` module, where the SPICE jar's Java code downloads
Stanford CoreNLP .jar files on its first invocation.

Usage per session (in eval notebook, BEFORE calling compute_spice):
    from evaluation.spice_cache import restore_spice_cache, save_spice_cache
    from core.io import get_drive_path

    SPICE_CACHE_DIR = str(get_drive_path("tools", "spice_corenlp_cache"))
    restore_spice_cache(SPICE_CACHE_DIR)   # fast no-op if nothing cached yet

    ... run evaluation (first run will download if cache was empty) ...

    save_spice_cache(SPICE_CACHE_DIR)   # only needs to actually copy on
                                        # the FIRST successful run; a no-op
                                        # afterward since files already exist
"""
import shutil
from pathlib import Path

from core.logging import get_logger

logger = get_logger(__name__)


def _get_spice_lib_dir() -> Path:
    """Locates the local pycocoevalcap spice/lib directory where CoreNLP
    jars land after SPICE's first run."""
    import pycocoevalcap.spice.spice as spice_module
    spice_pkg_dir = Path(spice_module.__file__).resolve().parent
    return spice_pkg_dir / "lib"


def restore_spice_cache(drive_cache_dir: str) -> bool:
    """Copies previously-cached CoreNLP jars from Drive into the local
    package directory, skipping the ~2GB download this session.

    Call this once, early, before the first compute_spice() call.

    Returns:
        True if a cache was found and restored, False if this is the
        first-ever run (nothing to restore yet — SPICE will download
        normally, then call save_spice_cache() afterward).
    """
    drive_cache = Path(drive_cache_dir)
    local_lib_dir = _get_spice_lib_dir()

    if not drive_cache.exists() or not any(drive_cache.iterdir()):
        logger.info(f"No SPICE cache found at {drive_cache}. "
                     f"First run will download CoreNLP models (~2GB, one-time).")
        return False

    local_lib_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Restoring cached CoreNLP models: {drive_cache} -> {local_lib_dir}")

    copied = 0
    for item in drive_cache.iterdir():
        dest = local_lib_dir / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        copied += 1

    logger.info(f"SPICE cache restored ({copied} new item(s) copied).")
    return True


def save_spice_cache(drive_cache_dir: str) -> None:
    """Copies the local CoreNLP jars (freshly downloaded or already cached)
    to Drive so future sessions can restore them.

    Safe to call every session — it's a no-op if everything's already
    present in the Drive cache.
    """
    local_lib_dir = _get_spice_lib_dir()
    drive_cache = Path(drive_cache_dir)

    if not local_lib_dir.exists() or not any(local_lib_dir.iterdir()):
        logger.warning(f"Nothing to cache yet: {local_lib_dir} is empty. "
                         f"Run compute_spice() at least once first.")
        return

    drive_cache.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in local_lib_dir.iterdir():
        dest = drive_cache / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        copied += 1

    if copied:
        logger.info(f"Saved SPICE cache to Drive: {copied} new item(s) -> {drive_cache}")
    else:
        logger.info("SPICE cache on Drive already up to date.")