from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from neurofunctionx.core.BaseProcessor import BaseProcessor


class SubjectDataBase(BaseProcessor, ABC):
    """Domain-agnostic base for subject-data containers.

    Holds the subject name, a data directory and a nested ``files`` dict, and
    provides versioned access to stored file paths. Subclasses decide how a
    subject is discovered and loaded (BIDS store, local folder, ...).
    """

    def __init__(self, subject_name: str):
        self.subject_name: str = subject_name
        self.data_dir: Path = Path()
        self.files: dict = {}

    def _exists(self, filename):
        return (self.data_dir / filename).exists()

    @property
    def is_loaded(self):
        return len(self.files) > 0

    @staticmethod
    def _set_versioned_entry(container: dict, key: str, row, version: Optional[int]):
        versions = container.setdefault("_versions", {}).setdefault(key, {})

        if version is not None and 0 <= version <= 99:
            versions[version] = row
        else:
            versions[0] = row

        latest_version = max(versions.keys())
        container[key] = versions[latest_version]

    def get_file(self, *keys: str, version: Optional[int | str] = None):
        """Return a stored file path addressed by a dynamic path of keys.

        Works for raw entries and nested derivatives. When several versions
        (00..99) exist, returns the newest unless ``version`` is given. Returns
        a path relative to ``data_dir`` or None if not found.
        """
        if not keys:
            self._log_error("No keys provided to get_file().")
            return None

        current = self.files
        for key in keys[:-1]:
            if not isinstance(current, dict) or key not in current:
                self._log_error(f"Path '{'/'.join(keys)}' not found for subject {self.subject_name}")
                return None
            current = current[key]

        target = keys[-1]
        if not isinstance(current, dict):
            self._log_error(f"Container for '{'/'.join(keys[:-1])}' is not a dictionary.")
            return None

        versions_map = current.get("_versions", {}).get(target)

        if version is None:
            if versions_map:
                v = max(versions_map.keys())
                return versions_map[v]
            return current.get(target)

        try:
            v_int = int(version) if not (isinstance(version, str) and version.strip() == "") else None
        except (TypeError, ValueError):
            v_int = None

        if v_int is None or not (0 <= v_int <= 99):
            self._log_error(f"Invalid version '{version}'. Must be integer/string in range 00..99.")
            return None

        if versions_map and v_int in versions_map:
            return versions_map[v_int]

        self._log_error(f"Version '{v_int:02d}' for {'/'.join(keys)} not found.")
        return None

    def get_files(self, *keys: str, versions: bool = True, **tags) -> Optional[List[Path]]:
        """Return multiple file paths from ``files``.

        Versioned container -> all versions (or newest when ``versions`` is
        False); folder-like dict -> all contained files; single file -> a
        one-item list. Returns None if nothing matches.
        """
        if not keys:
            self._log_error("No keys provided to get_files().")
            return None

        current = self.files
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                self._log_error(f"Path '{'/'.join(keys)}' not found for subject {self.subject_name}")
                return None
            current = current[key]

        if isinstance(current, dict) and "_versions" in current:
            if not versions:
                versions_map = current["_versions"]
                if not versions_map:
                    return None
                latest = max(versions_map.keys())
                return [versions_map[latest]]

            all_versions = []
            for vmap in current["_versions"].values():
                all_versions.extend(vmap.values())
            return sorted(all_versions, key=lambda p: str(p))

        if isinstance(current, dict):
            files = [v for k, v in current.items() if k != "_versions" and isinstance(v, Path)]
            return files if files else None

        if isinstance(current, Path):
            return [current]

        self._log_error(f"Unsupported structure at '{'/'.join(keys)}'.")
        return None

    @abstractmethod
    def load_subject(self):
        """Populate ``files`` for this subject."""

    @staticmethod
    @abstractmethod
    def list_all(dataset: str):
        """Yield the subjects available in ``dataset``."""
