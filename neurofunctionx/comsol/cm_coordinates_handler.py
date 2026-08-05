from pathlib import Path
import SimpleITK as sitk
import numpy as np
from collections import Counter

def get_tissue_conductivity_values(img: sitk.Image):
    # 0 -> WM, 1 -> GM, 2 -> CSF
    arr = sitk.GetArrayFromImage(img)
    flat = arr.flatten().astype(np.float64)
    counted_values = Counter(flat)
    if 0.0 in counted_values:
        counted_values.pop(0.0) # ignore background if available
    top3_values = sorted([item for item, count in counted_values.most_common(3)])
    return np.round(top3_values, decimals=5)

def finalize_conductivity_cube(cond_img: sitk.Image, reference_img: sitk.Image) -> sitk.Image:
    """Prepare a raw conductivity image for COMSOL.

    Copies the reference image's orientation, crops to the centre cube and
    replaces background (0) voxels with the gray-matter conductivity, since
    COMSOL cannot handle zero-conductivity regions.
    """
    from neurofunctionx.io.sitk.image_transform import extract_center_cube

    cond_img.SetOrigin(reference_img.GetOrigin())
    cond_img.SetDirection(reference_img.GetDirection())

    tissue_conductivities = get_tissue_conductivity_values(cond_img)
    cond_img = extract_center_cube(cond_img)
    cond_img[cond_img == 0.0] = tissue_conductivities[1]  # background -> gray matter
    return cond_img


def create_cm_dict_from_sitk(img: sitk.Image):

    size = np.array(img.GetSize())
    spacing = np.array(img.GetSpacing())

    side_lengths = size * spacing

    center_index = (size - 1) / 2.0
    center_physical = img.TransformContinuousIndexToPhysicalPoint(center_index.tolist())

    arr = sitk.GetArrayFromImage(img)
    flat = arr.flatten().astype(np.float64)
    counted_values = Counter(flat)
    #counted_values.pop(0.0) # ignore background
    top3_values = sorted([item for item, count in counted_values.most_common(3)])
    top3_values = np.round(top3_values, decimals=5)

    dimensions = {
        "side_lengths": list(side_lengths),
        "center": list(center_physical)
    }

    return {
        "preop": dimensions,
        "postop": dimensions,
        "classification": {
            "csf_conductivity": top3_values[2],
            "gm_conductivity": top3_values[1],
            "wm_conductivity": top3_values[0],
        }
    }



def write_cm_coordinates_file(data: dict, path: str | Path):

    def fmt_triplet(v):
        return f" {v[0]:7.2f}  {v[1]:7.2f}  {v[2]:7.2f}" if v else ""

    def fmt_xyz(d):
        if not d:
            return ""
        return f"x: {d.get('x')}\ny: {d.get('y')}\nz: {d.get('z')}\n"

    def fmt_tilts(t):
        if not t:
            return ""
        return (
            f"Theta: {t['Theta']}\n"
            f"Phi: {t['Phi']}\n"
            f"Torsion: {t['Torsion']}\n"
        )

    def fmt_lead(d, keys):
        if not d:
            return ""
        out = []
        for k in keys:
            if k in d:
                out.append(f"{k}: {d[k]}")
        return "\n".join(out) + "\n"

    c = data.get("classification", {})

    # --- Build output text dynamically ---
    out = ["Side lengths and centre points in COMSOL Multiphysics \n", "Coordinate system in Comsol Multiphysics:\n",
           "x: from right to left\ny: from anterior to posterior\nz: from deeper to higher in the brain\n\n"]

    # --- Preop ---
    if "preop" in data:
        pre = data["preop"]
        if "center" in pre:
            out.append("Centre point in preop model (x,y,z) [mm]:\n")
            out.append(fmt_triplet(pre["center"]) + "\n\n")
        if "side_lengths" in pre:
            out.append("Side lengths in preop model (width,depth,height) [mm]:\n")
            out.append(fmt_triplet(pre["side_lengths"]) + "\n\n")

    # --- Postop ---
    if "postop" in data:
        post = data["postop"]
        if "center" in post:
            out.append("Centre point in postop model (x,y,z) [mm]:\n")
            out.append(fmt_triplet(post["center"]) + "\n\n")
        if "side_lengths" in post:
            out.append("Side lengths in postop model (width,depth,height) [mm]:\n")
            out.append(fmt_triplet(post["side_lengths"]) + "\n\n")

    # --- Fiducials ---
    fids = data.get("fiducials", {})
    if "preop" in fids and fids["preop"]:
        out.append("Center of fiducials in preoperative images (mm, [100, 100, 100] in Leksell coordinates)\n")
        out.append(fmt_xyz(fids["preop"]) + "\n")

    if "postop" in fids and fids["postop"]:
        out.append("Center of fiducials in postoperative images (mm, [100, 100, 100] in Leksell coordinates)\n")
        out.append(fmt_xyz(fids["postop"]) + "\n")

    # --- Tilts ---
    tilts = data.get("tilts", {})
    if "preop" in tilts and tilts["preop"]:
        out.append("Tilts in polar radians \nPreoperative\n")
        out.append(fmt_tilts(tilts["preop"]) + "\n")

    if "postop" in tilts and tilts["postop"]:
        out.append("Postoperative\n")
        out.append(fmt_tilts(tilts["postop"]) + "\n")

    # --- Leads ---
    lead1 = data.get("lead_1")
    if lead1:
        out.append("Lead 1 position\n")
        out.append(fmt_lead(lead1, ["x0", "y0", "z0", "x3", "y3", "z3"]) + "\n")

    lead2 = data.get("lead_2")
    if lead2:
        out.append("Lead 2 position\n")
        out.append(fmt_lead(lead2, ["x8", "y8", "z8", "x11", "y11", "z11"]) + "\n")

    # --- Classification ---
    if c:
        out.append("Classification parameters \n\n")

        def add_class(name, key):
            if key in c and c[key] is not None:
                out.append(f"{name}:\n{c[key]}\n")

        add_class("Image type", "image_type")
        add_class("Mean CSF intensity", "mean_csf_intensity")
        add_class("Mean grey matter intensity", "mean_gm_intensity")
        add_class("Mean blood intensity", "mean_blood_intensity")
        add_class("Mean white matter intensity", "mean_wm_intensity")
        add_class("Mean STN/SN/GP/RN intensity", "mean_stn_sn_gp_rn_intensity")
        add_class("Pulse frequency (Hz)", "pulse_frequency_hz")
        add_class("Pulse width (um)", "pulse_width_um")
        add_class("CSF conductivity (S/m)", "csf_conductivity")
        add_class("Grey matter conductivity (S/m)", "gm_conductivity")
        add_class("Blood conductivity (S/m)", "blood_conductivity")
        add_class("White matter conductivity (S/m)", "wm_conductivity")

    # --- Write file ---
    Path(path).write_text("".join(out))
