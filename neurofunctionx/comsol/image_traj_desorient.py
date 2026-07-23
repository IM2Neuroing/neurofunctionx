import json
from pathlib import Path
from typing import List

import SimpleITK as sitk
import numpy as np

from neurofunctionx.io.data_helper import get_bids_file_parts
from neurofunctionx.io.sitk.image_transform import image_no_orient


def affine_matrix_from_points(p0, p1):
    """calculate an affine matrix that moves the vector [1,0,0] to p0p1 based on p0 and p1

    Args:
        p0 (np.array(3)): xyz coordinates of p0
        p1 (np.array(3)): xyz coordinates of p1

    Returns:
        np.array(4,4): affine transform (4x4)
    """
    x_axis = -(p1 - p0) / np.linalg.norm(p1 - p0)
    z_axis = np.cross(x_axis, np.array([0, 1, 0]))
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    translation_vector = np.array([[p0[0]], [p0[1]], [p0[2]]])

    affine_matrix = np.vstack(
        (np.column_stack((rotation_matrix, translation_vector)), [0, 0, 0, 1])
    )
    return affine_matrix


def sortPoints_targetFirst(pointPair):
    """sort two points on a trajectory target first and entry second

    Args:
        pointPair (list(list(3), list(3))): list of two points [[x0,y0,z1], [x1,y1,z1]]

    Returns:
        list(list(3), list(3)): list of the points reordered
    """
    return np.array(pointPair)[np.argsort(np.array(pointPair)[:, 2]), :].tolist()


def readTrajLPS(mrkJson_file):
    """Extract the points of the trajectories from a markupLine json saved in slicer.

    Args:
        mrkJson_file (str): path to markupline file

    Returns:
        list(list(3), list(3)): list of two points defining the list.
    """
    with open(mrkJson_file, "r") as f:
        traj_json = json.load(f)
    return [i["position"] for i in traj_json["markups"][0]["controlPoints"]]


def get_traj_json(desorient_filename: str, sitk_image: sitk.Image, trajs: List[Path]):
    no_orient_image = image_no_orient(sitk_image)

    key_map = {"L": "VIM links", "R": "VIM rechts", "L-T1Pre": "VIM links post",
               "R-T1Pre": "VIM rechts post"}

    # read the lines from the trajectories

    lines = {}

    for trajFile in trajs:
        name = Path(trajFile).name

        traj_parts = get_bids_file_parts(name)
        key = traj_parts["structure"]

        if key not in key_map:
            continue  # skip if the key is unknown

        # safe to add
        lines[key_map[key]] = readTrajLPS(str(trajFile))
    # reorder them
    points = {k: sortPoints_targetFirst(v) for k, v in lines.items()}
    # transform the points in the desoriented image
    trans_points = {
        k: [
            no_orient_image.TransformContinuousIndexToPhysicalPoint(
                sitk_image.TransformPhysicalPointToContinuousIndex(p)
            )
            for p in v
        ]
        for k, v in points.items()
    }
    # calculate the transformations from the transformed points
    transforms = {
        k: affine_matrix_from_points(np.array(v[0]), np.array(v[1])).tolist()
        for k, v in trans_points.items()
    }

    # json output
    return {
        "desorient_img": desorient_filename,
        "points": trans_points,
        "transforms": transforms,
    }
