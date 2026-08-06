import copy

import SimpleITK as sitk
import numpy as np

from neurofunctionx.io.data_helper import load_any_file
from neurofunctionx.io.sitk.data_handler import load_volume


def image_no_orient(sitk_image: sitk.Image) -> sitk.Image:
    sitk_image = copy.deepcopy(sitk_image)
    sitk_image.SetDirection(np.identity(3).reshape(1, -1).tolist()[0])
    sitk_image.SetOrigin([0, 0, 0])
    return sitk_image


def extract_nonzero_slices(sitk_image: sitk.Image) -> sitk.Image:
    return extract_roi(sitk_image, roi_nonzero_slices(sitk_image))


def resample_to_spacing(sitk_image: sitk.Image, spacing, interpolator=sitk.sitkLinear,
                        anti_alias: bool = True) -> sitk.Image:
    """
    Resample an image onto an isotropic (or per-axis) voxel size, in mm.

    The physical extent, origin and direction are preserved, so any transform
    computed on the result stays valid for the original image -- transforms map
    physical coordinates, not voxel indices.

    Mainly there to keep grids manageable: memory and runtime of deformable
    registration scale with the voxel count, which grows with the *cube* of the
    resolution, so going from 0.25 mm to 1 mm is a 64x reduction.

    Parameters
    ----------
    spacing : float | sequence of float
        Target voxel size in mm; a scalar means isotropic.
    interpolator : int
        ``sitk.sitkLinear`` for intensity images, ``sitk.sitkNearestNeighbor``
        for label maps.
    anti_alias : bool
        Low-pass filter before downsampling, so detail finer than the new voxel
        size folds back as blur rather than aliasing artefacts. Turn off for
        label maps (it would blend labels) and pair that with a nearest-neighbour
        interpolator.
    """
    dimension = sitk_image.GetDimension()
    if np.isscalar(spacing):
        new_spacing = [float(spacing)] * dimension
    else:
        new_spacing = [float(s) for s in spacing]
        if len(new_spacing) != dimension:
            raise ValueError(
                f"spacing needs {dimension} values for a {dimension}D image, got {len(new_spacing)}"
            )

    old_spacing = sitk_image.GetSpacing()
    old_size = sitk_image.GetSize()
    # Round up so the resampled grid still covers the full physical extent.
    new_size = [int(np.ceil(old_size[i] * old_spacing[i] / new_spacing[i])) for i in range(dimension)]

    source = sitk_image
    if anti_alias:
        # Nyquist: a sigma of half the added voxel size removes the detail the
        # coarser grid cannot represent. Zero on axes that are not downsampled.
        sigmas = [max(0.0, (new_spacing[i] ** 2 - old_spacing[i] ** 2) ** 0.5) / 2.0
                  for i in range(dimension)]
        if any(s > 0 for s in sigmas):
            source = sitk.SmoothingRecursiveGaussian(
                sitk.Cast(sitk_image, sitk.sitkFloat32), sigmas)
            source = sitk.Cast(source, sitk_image.GetPixelID())

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputPixelType(sitk_image.GetPixelID())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(source)


def roi_nonzero_slices(sitk_image: sitk.Image) -> tuple:
    mask = sitk.BinaryThreshold(sitk_image, lowerThreshold=0.1, upperThreshold=1e10)

    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(sitk_image, mask)
    bounding_box = stats.GetBoundingBox(1)  # 1 is the label value in the mask

    # Extract the min/max indices for each dimension
    min_x, max_x = bounding_box[0], bounding_box[1]
    min_y, max_y = bounding_box[2], bounding_box[3]
    min_z, max_z = bounding_box[4], bounding_box[5]

    # Add 1 to max coordinates because SimpleITK uses [min, max) convention
    max_x += 1
    max_y += 1
    max_z += 1

    roi = (int(min_x), int(min_y), int(min_z), int(max_x - min_x), int(max_y - min_y), int(max_z - min_z))

    return roi


def extract_roi(sitk_image: sitk.Image, roi: tuple) -> sitk.Image:
    index = roi[:3]
    size = roi[3:]
    roi_image = sitk.RegionOfInterest(sitk_image, size=size, index=index)
    new_origin = sitk_image.TransformIndexToPhysicalPoint(index)
    roi_image.SetOrigin(new_origin)
    return roi_image


def extract_deep_brain_cuboid(
        sitk_image: sitk.Image,
        offset_mm: tuple = (-0.1, 0.1, -4.4),
        size_mm: tuple = (26.1, 28.8, 53.6),
) -> sitk.Image:
    """
    Extracts the deep-brain cuboid from a T1 image using the universal placement
    spec derived from manual segmentations across subjects.

    offset_mm  – (x, y, z) shift from the brain centre in physical-space mm
    size_mm    – (x, y, z) cuboid dimensions in mm
    """
    brain_crop = extract_nonzero_slices(sitk_image)
    brain_size = np.array(brain_crop.GetSize(), dtype=float)
    brain_center_phys = np.array(brain_crop.TransformContinuousIndexToPhysicalPoint(
        ((brain_size - 1) / 2.0).tolist()
    ))

    target_phys = brain_center_phys + np.array(offset_mm)
    target_idx = np.array(sitk_image.TransformPhysicalPointToContinuousIndex(target_phys.tolist()))

    spacing = np.array(sitk_image.GetSpacing())
    half_voxels = np.array(size_mm) / 2.0 / spacing

    start = np.floor(target_idx - half_voxels).astype(int)
    end = np.floor(target_idx + half_voxels).astype(int)

    img_size = np.array(sitk_image.GetSize())
    start = np.clip(start, 0, img_size - 1)
    end = np.clip(end, 0, img_size - 1)

    roi_size = (end - start + 1).tolist()
    roi_start = start.tolist()

    roi = sitk.RegionOfInterest(sitk_image, size=roi_size, index=roi_start)
    roi.SetOrigin(sitk_image.TransformIndexToPhysicalPoint(roi_start))
    return roi


def extract_center_cube(sitk_volume, size_mm=100.0):
    # half-size in mm
    half_mm = size_mm / 2.0

    spacing = np.array(sitk_volume.GetSpacing())
    size = np.array(sitk_volume.GetSize())

    # Compute physical center in index coordinates
    index_center = (size - 1) / 2.0

    # Convert the half-size (mm) to voxel units in each direction
    half_voxels = np.round(half_mm / spacing).astype(int)

    start = np.floor(index_center - half_voxels).astype(int)
    end = np.floor(index_center + half_voxels).astype(int)

    # Clip to valid range
    start = np.maximum(start, 0)
    end = np.minimum(end, size - 1)

    # Convert index extent to SimpleITK RegionOfInterest size
    roi_size = (end - start + 1).tolist()
    roi_start = start.tolist()

    # Extract region
    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetSize(roi_size)
    extractor.SetIndex(roi_start)

    return extractor.Execute(sitk_volume)


def get_image_side(sitk_volume, axis=0, hemisphere="left"):
    """
    axis: 0 for X (Left/Right), 1 for Y, 2 for Z
    side: 'left' (or 'top'/'front'), 'right' (or 'bottom'/'back')
    """
    size = list(sitk_volume.GetSize())
    index = [0, 0, 0]

    # Calculate the midpoint of the chosen axis
    midpoint = size[axis] // 2

    if hemisphere.lower().startswith('r'):
        # Keep the first half: Index starts at 0, size is midpoint
        size[axis] = midpoint
        index[axis] = 0
    else:
        # Keep the second half: Index starts at midpoint, size is remainder
        size[axis] = size[axis] - midpoint
        index[axis] = midpoint

    cropper = sitk.RegionOfInterestImageFilter()
    cropper.SetSize(size)
    cropper.SetIndex(index)

    return cropper.Execute(sitk_volume)


def shift_image_by_intensity_center(sitk_volume: sitk.Image, percentile=20):
    """
    Shifts the image so the intensity center aligns with the geometric center,
    cropping the excess and padding with zero-value background.
    """
    arr = sitk.GetArrayFromImage(sitk_volume).astype(np.float32)

    # 1. Calculate centroid (same logic as before)
    thr = np.percentile(arr, percentile)
    weights = np.maximum(arr - thr, 0)

    total_weight = weights.sum()
    if total_weight == 0:
        return sitk_volume

    z, y, x = np.indices(arr.shape)
    cz = (weights * z).sum() / total_weight
    cy = (weights * y).sum() / total_weight
    cx = (weights * x).sum() / total_weight

    # 2. Calculate the shift required
    # Note: sitk indices are (x, y, z) while numpy indices are (z, y, x)
    centroid_index = np.array([cx, cy, cz])
    center_index = np.array(sitk_volume.GetSize()) / 2.0

    # We want to move the centroid to the center_index
    # The shift vector needed for the Transform
    shift_index = centroid_index - center_index
    shift_physical = shift_index * np.array(sitk_volume.GetSpacing())

    # 3. Configure Transform
    # To shift the image data, we apply the inverse transform to the grid
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(shift_physical)

    # 4. Resample with Nearest Neighbor
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_volume)
    resampler.SetDefaultPixelValue(0)  # Sets the padded areas to 0
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Prevents blurring/interpolation
    resampler.SetTransform(transform)
    resampled = resampler.Execute(sitk_volume)

    current_origin = np.array(sitk_volume.GetOrigin())
    new_origin = current_origin + shift_physical

    resampled.SetOrigin(new_origin)

    return resampled


def transform_to_another_image_space(image_path, reference_image_path, transform: sitk.Transform,
                                     interpolator=sitk.sitkLinear):
    image = load_volume(image_path)
    reference_image = load_volume(reference_image_path)

    return sitk.Resample(
        image,
        reference_image,  # (defines grid)
        transform,
        interpolator,  # sitkLinear for intensities, sitkNearestNeighbor for labels
        0.0,
        image.GetPixelID()
    )


# todo what to do with this
def transform_to_another_image(sitk_volume, transform: sitk.Transform):
    spacing = sitk_volume.GetSpacing()
    direction = sitk_volume.GetDirection()

    origin = sitk_volume.GetOrigin()

    size = sitk_volume.GetSize()

    return sitk.Resample(
        sitk_volume,
        size,
        transform,
        sitk.sitkLinear,
        origin,
        spacing,
        direction,
        0.0,
        sitk_volume.GetPixelID()
    )


def transform_trajectory_to_another_image_space(trajectory_path, transform_path):
    transform = sitk.ReadTransform(transform_path.as_posix()).GetInverse()

    data = load_any_file(trajectory_path)

    # transform control points
    for cp in data["markups"][0]["controlPoints"]:
        point = cp["position"]  # already LPS

        transformed = transform.TransformPoint(point)

        cp["position"] = list(transformed)

    return data


def convert_trajectory_to_index_space(input_markup, original_mri):
    data = load_any_file(input_markup)

    for markup in data["markups"]:
        for cp in markup["controlPoints"]:

            p = cp["position"]

            # physical → voxel index
            idx = original_mri.TransformPhysicalPointToIndex(p)

            cp["position"] = [float(idx[0]), float(idx[1]), float(idx[2])]

            # remove physical orientation
            if "orientation" in cp:
                del cp["orientation"]

        # remove physical metadata
        markup.pop("coordinateSystem", None)
        markup.pop("coordinateUnits", None)

    return data
