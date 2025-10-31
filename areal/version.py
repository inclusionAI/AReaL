import subprocess
from pathlib import Path


class VersionInfo:
    def __init__(self):
        self.__version__ = "0.3.4"
        self.__branch__ = ""
        self.__commit__ = ""
        self.__is_dirty__ = False
        self._populate_version_info()

    @property
    def version(self) -> str:
        return self.__version__

    @property
    def branch(self) -> str:
        return self.__branch__

    @property
    def commit(self) -> str:
        return self.__commit__

    @property
    def is_dirty(self) -> bool:
        return self.__is_dirty__

    @property
    def full_version(self) -> str:
        version = self.version
        if self.commit != "":
            version = f"{self.version}-{self.commit}"
        if self.is_dirty:
            version = f"{version}-dirty"
        return version

    @property
    def full_version_with_dirty_description(self) -> str:
        version = self.full_version
        if self.is_dirty:
            version = (
                f"{version} ('-dirty' means there are uncommitted code changes in git)"
            )
        return version

    def _populate_version_info(self):
        try:
            self.__branch__ = (
                subprocess.check_output(
                    ["git", "branch", "--show-current"],
                    stderr=subprocess.DEVNULL,
                    cwd=Path(__file__).parent,
                )
                .decode("utf-8")
                .strip()
            )
            self.__commit__ = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    cwd=Path(__file__).parent,
                )
                .decode("utf-8")
                .strip()
            )
            try:
                subprocess.check_call(
                    ["git", "diff-index", "--quiet", "HEAD", "--"],
                    stderr=subprocess.DEVNULL,
                    cwd=Path(__file__).parent,
                )
            except subprocess.CalledProcessError:
                self.__is_dirty__ = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass


version_info = VersionInfo()

# for backward compatibility
__version__ = version_info.version
__branch__ = version_info.branch
__commit__ = version_info.commit
__is_dirty__ = version_info.is_dirty
