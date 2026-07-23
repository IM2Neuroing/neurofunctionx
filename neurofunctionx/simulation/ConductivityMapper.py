import numpy as np

import SimpleITK as sitk
from neurofunctionx.core.BaseProcessor import BaseProcessor


class ConductivityMapper(BaseProcessor):

    def __init__(self, csf_cond, gm_cond, wm_cond):
        self.csf_cond = csf_cond
        self.gm_cond = gm_cond
        self.wm_cond = wm_cond

    @staticmethod
    def interpolate_conductivity(intensity, intensity1, intensity2, sigma1, sigma2):
        w = (intensity - intensity2) / (intensity1 - intensity2)
        sigma = w * sigma1 + (1 - w) * sigma2
        return sigma

    def map_conductivities(self, original_image: sitk.Image, label_volume: np.ndarray):
        self._log("Mapping conductivities")

        image_array = sitk.GetArrayFromImage(original_image)

        label_volume = label_volume.copy().astype(np.float32)

        mean_bg = image_array[(label_volume == 0)].mean()
        mean_csf = image_array[(label_volume == 1) & (image_array != 0)].mean()
        mean_gm = image_array[label_volume == 2].mean()
        mean_wm = image_array[label_volume == 3].mean()

        self._log(f"Average intensities for BG {mean_bg}, CSF {mean_csf}, GM {mean_gm}, WM {mean_wm}")
        self._log(f"Mapping with conductivities CSF {self.csf_cond}, GM {self.gm_cond}, WM {self.wm_cond}")

        third_csf_gm = (mean_gm - mean_csf) / 3
        third_gm_wm = (mean_wm - mean_gm) / 3

        middle_third_csf_gm_bottom = third_csf_gm + mean_csf
        middle_third_csf_gm_top = third_csf_gm * 2 + mean_csf

        middle_third_gm_wm_bottom = third_gm_wm + mean_gm
        middle_third_gm_wm_top = third_gm_wm * 2 + mean_gm

        label_volume[label_volume == 3] = self.wm_cond
        label_volume[label_volume == 2] = self.gm_cond
        label_volume[label_volume == 1] = self.csf_cond

        middle_third_csf_gm_mask = (image_array > middle_third_csf_gm_bottom) & (
                image_array < middle_third_csf_gm_top)

        middle_third_gm_wm_mask = (image_array > middle_third_gm_wm_bottom) & (
                image_array < middle_third_gm_wm_top)

        label_volume[middle_third_csf_gm_mask] = self.interpolate_conductivity(image_array[middle_third_csf_gm_mask],
                                                                               mean_csf, mean_gm, self.csf_cond,
                                                                               self.gm_cond)

        label_volume[middle_third_gm_wm_mask] = self.interpolate_conductivity(image_array[middle_third_gm_wm_mask],
                                                                              mean_gm, mean_wm, self.gm_cond,
                                                                              self.wm_cond)


        cond = sitk.GetImageFromArray(label_volume)
        cond.CopyInformation(original_image)

        return cond