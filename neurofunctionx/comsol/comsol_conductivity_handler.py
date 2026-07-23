import re

import SimpleITK as sitk
import numpy as np


def load_conductivity_from_file(file_path):
    with open(file_path, 'r') as fp:
        raw_data = fp.readlines()

    raw_grid = raw_data[1:4]
    raw_cond = raw_data[5:]

    # we store everything in numpy as yzx to have it as xyz later when converting later.

    number_pattern = r"[-+]?\d*\.\d+e[-+]?\d+" # old r"(\d+.\d+e[+|-]\d+)"

    grids = [np.array([float(i) * 1000 for i in re.findall(number_pattern, raw_grid[2])]),  # z -> I->S
             np.array([float(i) * 1000 for i in re.findall(number_pattern, raw_grid[1])]),  # y -> A->P
             np.array([float(i) * 1000 for i in re.findall(number_pattern, raw_grid[0])])]  # x -> R->L

    # the orientation of the image in ELMA is LPS.

    matrix_size = np.array([np.shape(i)[0] for i in grids])
    matrix_spacing = np.array([np.mean(np.diff(i)) for i in grids])
    matrix_origin = np.array([np.min(i) for i in grids]) - matrix_spacing

    cond_matrix = np.zeros(matrix_size, dtype=float)
    # WARNING: here we suppose the i dimension is in eachline, each new lines increments j, and every block of matrix_size[1] lines is a new slice in the k dimension
    j = 0
    k = 0
    for iLine in range(matrix_size[1] * matrix_size[0]):
        cond_matrix[k, j, :] = np.array([float(i) for i in re.findall(r"(\d+.\d+e[+|-]\d+)", raw_cond[iLine])])
        j += 1
        if j % matrix_size[1] == 0:
            k += 1
            j = 0

    outputImage = sitk.GetImageFromArray(cond_matrix)

    outputImage.SetOrigin(matrix_origin[::-1])  # flip to xyz
    outputImage.SetSpacing(matrix_spacing[::-1])  # spacing also needs to be flipped to xyz
    return outputImage


def save_conductivity_to_file(img: sitk.Image, file_path: str):
    # SimpleITK → numpy
    arr = sitk.GetArrayFromImage(img).astype(float)  # (z,y,x)
    K, J, I = arr.shape

    spacing = np.array(img.GetSpacing())  # in mm
    origin = np.array(img.GetOrigin())  # in mm

    # -----------------------------
    # Build COMSOL grids in meters
    # COMSOL rule:
    # min(grid) = origin_mm + spacing_mm
    # values = (origin + spacing) + n * spacing
    # -----------------------------
    def build_grid(origin_mm, spacing_mm, n):
        coords_mm = (origin_mm + spacing_mm) + np.arange(n) * spacing_mm
        coords_m = coords_mm / 1000.0
        return coords_m

    x_grid = build_grid(origin[0], spacing[0], I)  # R→L
    y_grid = build_grid(origin[1], spacing[1], J)  # A→P
    z_grid = build_grid(origin[2], spacing[2], K)  # I→S

    # Format helper
    def fmt_line(values):
        return "    ".join(f"{v:.5e}" for v in values)

    with open(file_path, "w") as f:
        f.write("% Grid\n")
        f.write(fmt_line(x_grid) + "\n")
        f.write(fmt_line(y_grid) + "\n")
        f.write(fmt_line(z_grid) + "\n")
        f.write("% Data\n")

        # COMSOL layout: for z then y, each line is all x
        for k in range(K):
            for j in range(J):
                f.write(fmt_line(arr[k, j, :]) + "\n")
