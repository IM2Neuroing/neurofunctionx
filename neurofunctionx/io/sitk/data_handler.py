import SimpleITK as sitk
from pathlib import Path
from typing import Union


def load_volume(path: Union[str, Path]):
    path = Path(path)
    path_str = str(path)
    if path.is_dir():
        return _load_dicom(path_str)
    elif path_str.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")):
        if Path(path).is_file():
            return sitk.ReadImage(path, outputPixelType=sitk.sitkFloat32)
        else:
            raise FileNotFoundError(f"File does not exist: {path_str}")
    else:
        raise ValueError(f"Unsupported image format or path: {path_str}")


def _load_dicom(dicom_dir: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

    if not series_ids:
        raise FileNotFoundError(f"No DICOM Series in {dicom_dir} found")

    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])

    reader.SetFileNames(series_file_names)
    return reader.Execute()


def save_volume(image: sitk.Image, path: str):
    sitk.WriteImage(image, path)


def load_transform(path) -> sitk.Transform:
    return sitk.ReadTransform(str(path))


def save_transform(transform: sitk.Transform, path: str):
    sitk.WriteTransform(transform, str(path))
