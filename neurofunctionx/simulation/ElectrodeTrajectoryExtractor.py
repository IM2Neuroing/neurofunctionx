from neurofunctionx.core.BaseProcessor import BaseProcessor
from neurofunctionx.io.data_helper import save_any_file
from neurofunctionx.io.sitk.image_transform import extract_center_cube, get_image_side, \
    shift_image_by_intensity_center, extract_roi, roi_nonzero_slices

import numpy as np
import scipy.ndimage as ndi
from sklearn.decomposition import PCA
import SimpleITK as sitk


def extract_electrode_trajectories(ct, hemispheres=("left", "right"), mask=None):
    """Extract electrode trajectories from a (post-op) CT image.

    Operates on direct images rather than BIDS references, so it is reusable
    outside a BIDS context. If ``mask`` (an image already in CT space) is given,
    the CT is masked and cropped to the mask ROI before fitting.

    Returns a dict mapping hemisphere -> Slicer trajectory (markup) dict for the
    hemispheres where an electrode was found.
    """
    if mask is not None:
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        masked_ct = sitk.Mask(ct, mask)
        ct = extract_roi(masked_ct, roi_nonzero_slices(masked_ct))

    extractor = ElectrodeTrajectoryExtractor(ct)
    trajectories = {}
    for hemisphere in hemispheres:
        extractor.fit(hemisphere=hemisphere)
        if hemisphere in extractor.electrodes:
            trajectories[hemisphere] = extractor.get_as_slicer_trajectory(hemisphere)
    return trajectories


class ElectrodeTrajectoryExtractor(BaseProcessor):

    def __init__(self, ct):
        self.ct = shift_image_by_intensity_center(ct)
        self.electrodes = {}

    def fit_all(self):
        self.fit(hemisphere="left")
        self.fit(hemisphere="right")

    def fit(self, hemisphere="left", electrode_offset=1.5):

        ct_roi = extract_center_cube(self.ct, 80)
        ct_roi = get_image_side(ct_roi, hemisphere=hemisphere)
        ct_np = sitk.GetArrayFromImage(ct_roi)

        # Metal threshold
        threshold = np.percentile(ct_np, 99.5)  # todo check for high intensities as well maybe before halfing
        metal_mask = ct_np > threshold

        try:
            electrode_mask = self._get_electrode_label(metal_mask)
        except RuntimeError:
            self._log("No electrode-like component found with 1 iteration of erosion. Retrying with 2.")

            try:
                electrode_mask = self._get_electrode_label(metal_mask, iterations=2)
            except RuntimeError:
                self._log_error("No electrode found for side " + hemisphere + ".")
                return

        coords = np.column_stack(np.where(electrode_mask))

        # Convert to LPS coordinates
        spacing = np.array(ct_roi.GetSpacing())
        origin = np.array(ct_roi.GetOrigin())

        direction_matrix = np.array(ct_roi.GetDirection()).reshape(3, 3)

        coords_xyz = coords[:, ::-1]
        scaled_indices = coords_xyz * spacing
        coords_phys = origin + (direction_matrix @ scaled_indices.T).T

        center, direction = self._fit_pca(coords_phys)

        # Project points onto trajectory
        v = coords_phys - center

        t = np.dot(v, direction)
        v_parallel = np.outer(t, direction)
        v_perp = v - v_parallel
        dist_from_line = np.linalg.norm(v_perp, axis=1)
        mask_radius_threshold = 1.5  # max thickness of the electrode (actual 1.27mm 3389)
        inlier_indices = dist_from_line < mask_radius_threshold
        coords_phys = coords_phys[inlier_indices]

        center, direction = self._fit_pca(coords_phys)

        t = np.dot(coords_phys - center, direction)

        t_min = np.percentile(t, 1)
        t_max = np.percentile(t, 99)

        p0 = center + t_min * direction
        p1 = center + t_max * direction

        # Entry is the more superior point
        if p0[2] < p1[2]:
            entry = p0
            tip_end = p1
        else:
            entry = p1
            tip_end = p0

        direction = tip_end - entry
        direction /= np.linalg.norm(direction)
        entry += electrode_offset * direction  # electrodes are starting from the first contact, not from the actual entry
        # Offset for first contact
        tip = entry + 20 * direction

        self.electrodes[hemisphere] = {"entry": entry, "tip": tip, "direction": direction, "center": center}
        self._log("Electrode for side " + hemisphere + " found")

    @staticmethod
    def _fit_pca(coords_phys):

        # Traj fit
        pca = PCA(n_components=1)
        pca.fit(coords_phys)

        direction = pca.components_[0]
        direction /= np.linalg.norm(direction)

        center = coords_phys.mean(axis=0)  # center of label

        return center, direction

    @staticmethod
    def _get_electrode_label(metal_mask: np.ndarray, iterations: int = 1):
        expanded = ndi.binary_dilation(metal_mask, iterations=iterations)
        labeled, num = ndi.label(expanded)  # Connected components

        best_label = None
        best_score = -np.inf

        for label in range(1, num + 1):

            coords = np.column_stack(np.where(labeled == label))

            # ignore small components
            if coords.shape[0] < 20:
                continue

            # PCA-based linearity score
            coords_centered = coords - coords.mean(axis=0)
            cov = np.cov(coords_centered, rowvar=False)

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

            linearity = eigvals[0] / (eigvals[1] + 1e-6)

            if linearity > best_score:
                best_score = linearity
                best_label = label

        if best_label is None:
            raise RuntimeError("No electrode-like component found")

        electrode_mask = labeled == best_label

        # todo maybe (adjust) some hemispheres don't contain an electrode
        if np.count_nonzero(electrode_mask) < 1000:
            raise RuntimeError("Not enough electrode-like component found")

        electrode_mask = ndi.binary_erosion(electrode_mask, iterations=iterations)  # decrease again
        return electrode_mask

    def get_as_slicer_trajectory(self, side="left"):

        """
        entry, tip: iterable of length 3 (RAS coordinates in mm)
        """
        entry = self.electrodes[side]["entry"]
        tip = self.electrodes[side]["tip"]

        data = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [
                {
                    "type": "Line",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "locked": False,
                    "fixedNumberOfControlPoints": False,
                    "labelFormat": "%N-%d",
                    "lastUsedControlPointNumber": 2,
                    "controlPoints": [
                        {
                            "id": "1",
                            "label": "Entry",
                            "position": list(entry),
                            "orientation": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                            "selected": True,
                            "locked": False,
                            "visibility": True,
                            "positionStatus": "defined"
                        },
                        {
                            "id": "2",
                            "label": "Target",
                            "position": list(tip),
                            "orientation": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                            "selected": True,
                            "locked": False,
                            "visibility": True,
                            "positionStatus": "defined"
                        }
                    ]
                }
            ]
        }
        return data
