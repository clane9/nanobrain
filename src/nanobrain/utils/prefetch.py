import secrets
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
from fsspec.implementations.local import LocalFileSystem


def prefetch(
    root: str,
    paths: list[str],
    *,
    cache_dir: str | Path | None = None,
    max_workers: int = 1,
    prefetch_factor: int = 4,
    storage_options: dict | None = None,
):
    """Prefetch files from a remote source using fsspec.

    Yields tuples of (relative_path, local_path).
    """
    delete = cache_dir is None
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="prefetch-")

    fs, root_ = fsspec.url_to_fs(root, **(storage_options or {}))
    is_remote = not isinstance(fs, LocalFileSystem)

    def fn(path: str):
        full_path = f"{root_}/{path}"
        if is_remote:
            tmp_path = Path(cache_dir) / full_path.lstrip("/")
            if not tmp_path.exists():
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                get_file_atomic(fs, full_path, tmp_path)
            full_path = str(tmp_path)
        return path, full_path

    try:
        with ThreadPoolExecutor(max_workers) as executor:
            for path, f in executor.map(fn, paths, buffersize=prefetch_factor * max_workers):
                yield path, f

                if delete and is_remote:
                    Path(f).unlink(missing_ok=True)
    finally:
        if delete:
            shutil.rmtree(cache_dir)


def get_file_atomic(fs: fsspec.AbstractFileSystem, rpath: str, lpath: str | Path, *args, **kwargs):
    """fsspec get_file but safer."""
    lpath = Path(lpath)
    staging = lpath.with_name(f".tmp-{secrets.token_hex(3)}-{lpath.name}")
    try:
        fs.get_file(rpath, str(staging), *args, **kwargs)
        staging.rename(lpath)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
