"""Portable N4 bias-field correction + HD-BET skull stripping.

Operates on direct SimpleITK images and file-path lists.
HD-BET is imported lazily so importing this module
does not require it installed.
"""
from pathlib import Path

import SimpleITK as sitk

from neurofunctionx.core.BaseProcessor import BaseProcessor


class N4Striper(BaseProcessor):

    @staticmethod
    def bias_correct(image: sitk.Image) -> sitk.Image:
        """N4 bias-field correction, masking to the non-zero foreground."""
        mask_image = sitk.Cast(image > 0, sitk.sitkUInt8)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        return corrector.Execute(image, mask_image)

    @staticmethod
    def skull_strip(self, files, masks, overwrite=False):
        """Skull-strip each file with HD-BET and apply the resulting mask.

        ``files`` / ``masks`` are aligned lists of file paths (the mask paths are
        written by HD-BET). Applies each mask to its image and returns the list of
        brain paths (``_N4_`` -> ``_N4-Brain_``).
        """
        from HD_BET.checkpoint_download import maybe_download_parameters
        from HD_BET.hd_bet_prediction import get_hdbet_predictor, apply_bet

        maybe_download_parameters()
        predictor = get_hdbet_predictor()

        predictor.predict_from_files([[file] for file in files], masks, save_probabilities=False,
                                     overwrite=overwrite,
                                     num_processes_preprocessing=4, num_processes_segmentation_export=8,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

        brains = []
        for file, mask in zip(files, masks):
            brain = file.replace("_N4_", "_N4-Brain_")
            apply_bet(file, mask, brain)
            brains.append(brain)
        return brains
