"""
High-quality registration of two skull-stripped brains.

Wraps ANTsPy (the Python bindings for ANTs) so a single call performs the same
rigid -> affine -> SyN deformable registration with the cross-correlation (CC)
metric that is the research-grade standard for brain-to-brain warping. Inputs
and outputs stay interoperable with the rest of the project: brains may be
``sitk.Image`` objects or any path readable by :func:`load_volume`, and warped
results are returned as ``sitk.Image``.

Requires the optional ``registration`` extra::

    pip install "neurofunctionx[registration]"      # installs antspyx
"""
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import SimpleITK as sitk

from neurofunctionx.io.sitk.data_handler import load_volume

ImageOrPath = Union[sitk.Image, str, Path]


@dataclass
class RegistrationResult:
    """
    Result of :func:`register_brains`.

    Attributes
    ----------
    warped_image : sitk.Image
        ``moving`` resampled into the ``fixed`` grid.
    forward_transforms : list[str]
        ANTs transform files mapping *moving -> fixed*. Pass to
        :func:`apply_registration` (or ``ants.apply_transforms``) to move
        associated images/labels from moving space into fixed space.
    inverse_transforms : list[str]
        Transform files for the reverse direction (*fixed -> moving*).
    """
    warped_image: sitk.Image
    forward_transforms: List[str]
    inverse_transforms: List[str]


def _require_ants():
    if platform.system() != "Linux":
        raise RuntimeError(
            "register_brains is supported on Linux only; current platform is "
            f"{platform.system()}. ANTsPy has no Windows wheels, so run this on "
            "Linux (e.g. WSL or a Linux container)."
        )
    try:
        import ants  # noqa: F401
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "register_brains requires ANTsPy. Install it with "
            '`pip install "neurofunctionx[registration]"` (or `pip install antspyx`).'
        ) from exc
    return ants


def _to_ants(image: ImageOrPath):
    """Load any supported input into an ANTs image via SimpleITK.

    Routing through :func:`load_volume` gives uniform format handling (NIfTI,
    NRRD, MHA, DICOM directories) and a consistent float32 cast; the on-disk
    NIfTI round-trip preserves the physical geometry (spacing/origin/direction)
    exactly, avoiding error-prone in-memory direction-matrix conversions.
    """
    ants = _require_ants()
    if not isinstance(image, sitk.Image):
        image = load_volume(image)
    image = sitk.Cast(image, sitk.sitkFloat32)

    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp.close()
    try:
        sitk.WriteImage(image, tmp.name)
        return ants.image_read(tmp.name)
    finally:
        os.remove(tmp.name)


def _ants_to_sitk(ants_image) -> sitk.Image:
    ants = _require_ants()
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp.close()
    try:
        ants.image_write(ants_image, tmp.name)
        return sitk.ReadImage(tmp.name)
    finally:
        os.remove(tmp.name)


def register_brains(
        fixed: ImageOrPath,
        moving: ImageOrPath,
        type_of_transform: str = "SyNCC",
        **kwargs,
) -> RegistrationResult:
    """
    Register two skull-stripped brains (``moving`` -> ``fixed``) with ANTs.

    Parameters
    ----------
    fixed, moving : sitk.Image | str | Path
        The reference brain and the brain to be aligned to it.
    type_of_transform : str
        ANTs transform preset. ``"SyNCC"`` (default) runs rigid + affine + SyN
        with the cross-correlation metric -- highest quality, slowest. Use
        ``"SyN"`` for a faster mutual-information variant, or ``"Affine"`` /
        ``"Rigid"`` for linear-only alignment.
    **kwargs
        Forwarded verbatim to ``ants.registration`` (e.g. ``reg_iterations``,
        ``grad_step``, ``mask``, ``moving_mask``, ``verbose=True``).

    Returns
    -------
    RegistrationResult
        The warped moving brain plus the forward/inverse transform file lists.
    """
    ants = _require_ants()

    fixed_ants = _to_ants(fixed)
    moving_ants = _to_ants(moving)

    reg = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=type_of_transform,
        **kwargs,
    )

    return RegistrationResult(
        warped_image=_ants_to_sitk(reg["warpedmovout"]),
        forward_transforms=reg["fwdtransforms"],
        inverse_transforms=reg["invtransforms"],
    )


def apply_registration(
        moving: ImageOrPath,
        reference: ImageOrPath,
        transforms: List[str],
        interpolator: str = "linear",
) -> sitk.Image:
    """
    Apply a transform list from :func:`register_brains` to another image.

    Use ``interpolator="linear"`` (or ``"bSpline"``) for intensity images and
    ``"nearestNeighbor"`` / ``"genericLabel"`` for label maps.

    Parameters
    ----------
    moving : sitk.Image | str | Path
        Image to resample (must live in the same space as the original moving
        brain, e.g. a label map of that subject).
    reference : sitk.Image | str | Path
        Image defining the target grid (typically the fixed brain).
    transforms : list[str]
        ``RegistrationResult.forward_transforms`` (moving -> fixed) or
        ``inverse_transforms`` (fixed -> moving).
    """
    ants = _require_ants()

    out = ants.apply_transforms(
        fixed=_to_ants(reference),
        moving=_to_ants(moving),
        transformlist=transforms,
        interpolator=interpolator,
    )
    return _ants_to_sitk(out)