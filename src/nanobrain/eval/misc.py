import subprocess
from pathlib import Path


def git_sha() -> str:
    kwargs = dict(cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kwargs).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], **kwargs).stdout.strip()
    return f"{sha}-dirty" if dirty else sha
