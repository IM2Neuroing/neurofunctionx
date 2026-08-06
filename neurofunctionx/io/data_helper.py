import hashlib
import json
import re
import shutil
from pathlib import Path


def _content_extension(name: str) -> str:
    """Return the part after the first dot, e.g. "sub.nii.gz" -> "nii.gz"."""
    return name.split(".", 1)[1] if "." in name else ""


def _is_path_payload(file) -> bool:
    """True when ``file`` names files on disk rather than holding content.

    Some producers hand back paths instead of objects -- ANTs writes its
    transforms itself and ``register_brains`` returns the filenames. Those need
    copying, not re-serialising.
    """
    if isinstance(file, (str, Path)):
        return Path(file).exists()
    if isinstance(file, (list, tuple)) and len(file) > 0:
        return all(isinstance(item, (str, Path)) for item in file)
    return False


def copy_existing_files(sources, destination) -> list:
    """
    Copy files that already exist on disk to ``destination``.

    ``destination`` may be a directory, in which case every file keeps its own
    name, or a single file path, which requires exactly one source. Returns the
    paths written.
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]
    sources = [Path(item) for item in sources]

    if not sources:
        raise ValueError("No files to copy")
    missing = [str(item) for item in sources if not item.exists()]
    if missing:
        raise FileNotFoundError(f"File(s) not found: {missing}")

    destination = Path(destination)
    if destination.is_dir() or destination.suffix == "":
        destination.mkdir(parents=True, exist_ok=True)
        targets = [destination / item.name for item in sources]
    else:
        if len(sources) != 1:
            raise ValueError(
                f"{len(sources)} files cannot be written to the single path "
                f"{destination} ({', '.join(item.name for item in sources)}). Pass a "
                f"directory instead. If these are ANTs transforms from "
                f"register_brains, re-run it with write_composite_transform=True to "
                f"get a single .h5 file."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        targets = [destination]

    for source, target in zip(sources, targets):
        shutil.copy2(source, target)
    return [str(item) for item in targets]


def save_any_file(file, file_path: Path):
    name = file_path.name
    if name.endswith(".json"):
        with open(str(file_path), "w+") as outfile:
            json.dump(file, outfile, sort_keys=True, indent=4)
        return

    if _is_path_payload(file):
        return copy_existing_files(file, file_path)

    ext = _content_extension(name)
    if ext in ("nii.gz", "mha", "nrrd"):
        from neurofunctionx.io.sitk.data_handler import save_volume
        save_volume(file, str(file_path))
    elif ext in ("h5", "hdf5", "tfm"):
        from neurofunctionx.io.sitk.data_handler import save_transform
        save_transform(file, str(file_path))
    elif ext == "vtu":
        from neurofunctionx.io.vtk.file_handler import write_vtk
        write_vtk(file, str(file_path))
    else:
        raise NotImplementedError(f"Unsupported file type: {name}")


def load_any_file(file_path):
    name = file_path.name
    if name.endswith(".json"):
        with open(str(file_path), "r") as f:
            return json.load(f)

    ext = _content_extension(name)
    if ext in ("nii.gz", "mha", "nrrd"):
        from neurofunctionx.io.sitk.data_handler import load_volume
        return load_volume(file_path)
    elif ext in ("h5", "hdf5", "tfm"):
        from neurofunctionx.io.sitk.data_handler import load_transform
        return load_transform(file_path)
    elif ext == "vtu":
        from neurofunctionx.io.vtk.file_handler import read_vtk
        return read_vtk(str(file_path))
    else:
        raise NotImplementedError(f"Unsupported file type: {name}")


def get_file_hash(path):
    if not path.exists():
        return None

    hasher = hashlib.new("sha256")
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_sidecar_name_of_file(file_path) -> str:
    return f"{str(file_path.name).split('.', 1)[0]}_sidecar.json"


_BIDS_NAME_PATTERN = re.compile(
    r"ses-(?P<session>[^_]+)"        # session (required)
    r"(?:_acq-(?P<acq>[^_]+))?"      # optional acq
    r"(?:_space-(?P<space>[^_]+))?"  # optional space
    r"(?:_run-(?P<run>[^_]+))?"      # optional run
    r"(?:_(?P<structure>[^_\.]+))?"  # optional structure
    r"_(?P<suffix>[^\.]+)"           # suffix (required)
    r"(?P<ext>\..+)$"                # extension (anything)
)


def get_bids_file_parts(file_name: str | Path):
    if isinstance(file_name, Path):
        file_name = file_name.name

    match = _BIDS_NAME_PATTERN.search(file_name)
    if not match:
        return None

    parts = match.groupdict()
    parts["file_name"] = (parts["structure"] + "_" if parts["structure"] else "") + parts["suffix"]
    return parts


def print_dict_tree(d, indent=0):
    for key in d:
        print("  " * indent + f"- {key}")
        if isinstance(d[key], dict):
            print_dict_tree(d[key], indent + 1)
