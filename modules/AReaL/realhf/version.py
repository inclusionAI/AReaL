import subprocess
from pathlib import Path

__version__ = "0.3.0-dev"
__branch__ = ""
__commit__ = ""
__is_dirty__ = False

try:
    __branch__ = (
        subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
        )
        .decode("utf-8")
        .strip()
    )
    __commit__ = (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
        )
        .decode("utf-8")
        .strip()
    )
    __is_dirty__ = False
    try:
        subprocess.check_call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
        )
    except subprocess.CalledProcessError:
        __is_dirty__ = True
except (subprocess.CalledProcessError, FileNotFoundError):
    pass


def get_full_version() -> str:
    version = __version__
    if __commit__ != "":
        version = f"{__version__}-{__commit__}"
    if __is_dirty__:
        version = f"{version}-dirty"
    return version


def get_full_version_with_dirty_description() -> str:
    version = get_full_version()
    if __is_dirty__:
        version = (
            f"{version} ('-dirty' means there are uncommitted code changes in git)"
        )
    return version
