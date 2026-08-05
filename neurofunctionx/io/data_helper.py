import hashlib
import json
import re
from pathlib import Path


def _content_extension(name: str) -> str:
    """Return the part after the first dot, e.g. "sub.nii.gz" -> "nii.gz"."""
    return name.split(".", 1)[1] if "." in name else ""


def save_any_file(file, file_path: Path):
    name = file_path.name
    if name.endswith(".json"):
        with open(str(file_path), "w+") as outfile:
            json.dump(file, outfile, sort_keys=True, indent=4)
        return

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
