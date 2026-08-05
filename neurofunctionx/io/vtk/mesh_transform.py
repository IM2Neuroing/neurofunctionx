import numpy as np
import SimpleITK as sitk
from scipy.interpolate import griddata

# todo remove
def sample_to_image_interpolated(points, values, ref_img, method="nearest"):
    # --- create empty image in reference space ---
    img = sitk.Image(ref_img.GetSize(), sitk.sitkFloat32)
    img.CopyInformation(ref_img)

    # --- fill known samples ---
    for p, v in zip(points, values):
        try:
            idx = ref_img.TransformPhysicalPointToIndex(tuple(p))
            if all(0 <= idx[d] < ref_img.GetSize()[d] for d in range(3)):
                img[idx] = float(v)
        except RuntimeError:
            # point outside image
            continue

    # --- choose interpolator ---
    if method == "linear":
        interpolator = sitk.sitkLinear
    elif method == "nearest":
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError(f"Unsupported method: {method}")

    # --- resample (identity transform, but fills gaps via interpolation) ---
    resampled = sitk.Resample(
        img,
        ref_img,
        sitk.Transform(),
        interpolator,
        0.0,  # default value
        sitk.sitkFloat32
    )

    return resampled


def threshold_image(img, threshold=200):
    return sitk.BinaryThreshold(
        img,
        lowerThreshold=threshold,
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )


def sample_interpolated(mesh_points, mesh_values, ref_img):
    size = ref_img.GetSize()

    grid = np.indices(size[::-1]).reshape(3, -1).T
    phys_points = [ref_img.TransformIndexToPhysicalPoint(tuple(idx[::-1])) for idx in grid]

    interp_vals = griddata(mesh_points, mesh_values, phys_points, method='linear', fill_value=0)

    img_np = interp_vals.reshape(size[::-1])
    img = sitk.GetImageFromArray(img_np)
    img.CopyInformation(ref_img)
    return img


def transform_mesh_points(mesh, origin, direction, transform=None, scale=1000.0):
    """Move a VTK mesh's points into an image's physical space.

    Scales the mesh coordinates (``scale``, metres->mm by default), rotates them
    by the image ``direction`` and shifts by ``origin``; then optionally applies a
    further SimpleITK ``transform`` point-by-point. Returns the same mesh with its
    points replaced.
    """
    import vtk
    import vtkmodules.util.numpy_support as nps

    coords = nps.vtk_to_numpy(mesh.GetPoints().GetData())
    coords = (coords * scale) @ np.asarray(direction).reshape(3, 3).T + np.asarray(origin)
    if transform is not None:
        coords = np.array([transform.TransformPoint(p.tolist()) for p in coords])

    points = vtk.vtkPoints()
    points.SetData(nps.numpy_to_vtk(coords, deep=1))
    mesh.SetPoints(points)
    mesh.Modified()
    return mesh
