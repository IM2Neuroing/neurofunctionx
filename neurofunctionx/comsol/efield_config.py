"""Portable COMSOL e-field configuration building.

``build_efield_config`` is pure computation over a stimulation dataframe, and
``EFieldConfigurationWriter`` drives the whole per-subject job: it queries the
stimulation db and builds one config per hemisphere. Neither knows about BIDS —
the caller injects the conductivity-matrix name, the ELMA coordinates, and two
callables that provide the per-side desorient transform and the per-hemisphere
base file name (see neurodatax's processing step for the BIDS-facing wrapper).
"""
import re

import pandas as pd

from neurofunctionx.core.BaseProcessor import BaseProcessor
from neurofunctionx.comsol.simulation_parameters import (
    get_intra_simulations_parameters,
    get_post_simulations_parameters,
    get_chronic_simulations_parameters,
)

ELECTRODE_MODELS = {
    "Intraoperativ": "sub-elecMER_comsol-model_model.mph",
    "Medtronic 3389": "comsolModel3389.mph",
    "Abbott SJM 6170": "comsolModelAbbot6180.mph",
}


def parse_cm_coordinates(lines) -> dict:
    """Parse ELMA CM-coordinate text ``lines`` into pre/post centre & side vectors."""
    def numbers(line):
        return [float(x) for x in re.findall(r"(\d+\.\d+)+", line)]

    idx_center = [i for i, line in enumerate(lines) if "Centre point in" in line]
    idx_side = [i for i, line in enumerate(lines) if "Side lengths in" in line]

    return {
        "pre_center": numbers(lines[idx_center[0] + 1]),
        "post_center": numbers(lines[idx_center[1] + 1]),
        "pre_side": numbers(lines[idx_side[0] + 1]),
        "post_side": numbers(lines[idx_side[1] + 1]),
    }


def build_efield_config(hemisphere, df_side, elma_coords, sim_type, desorient_transform,
                        subject, cond_matrix_name, base_file_name, results_folder="/efields/",
                        electrode_models=None):
    """Build the COMSOL e-field config dict for one hemisphere.

    Pure computation over the per-hemisphere stimulation dataframe ``df_side``;
    every BIDS-resolved value (``desorient_transform``, ``subject``,
    ``cond_matrix_name``, ``base_file_name``) is passed in. Returns ``None`` when
    there is no desorient transform.
    """
    electrode_models = electrode_models or ELECTRODE_MODELS

    if desorient_transform is None:
        return None

    electrode_name = electrode_models["Intraoperativ"]

    if sim_type == "Intraoperativ":
        trajectories = {}
        for traj_type in df_side["trajectory_type"].unique().tolist():
            trajectories[traj_type] = (
                df_side[(df_side["trajectory_type"] == traj_type) & (df_side["amplitude"] != 0)]
                .groupby("position_on_trajectory")["amplitude"].apply(list).to_dict()
            )
    else:
        electrode_name = electrode_models[df_side["electrode_name"].unique()[0]]

        p = -0.75
        trajectories = {f"C{i}": {p: []} for i in range(1, 5)}
        for traj in trajectories.values():  # test values
            traj[p] = [i / 10 for i in range(2, 51, 2)]

        for _, row in df_side.iterrows():
            contact = int(row["contact_number"])
            amplitude = float(row["amplitude"])
            if amplitude == 0:  # skip 0 amplitudes
                continue
            if row["hemisphere"].startswith("L"):
                contact = contact - 4
            if amplitude not in trajectories["C" + str(contact + 1)][p]:
                trajectories["C" + str(contact + 1)][p].append(amplitude)

    config = {
        "subject": subject,
        "hemisphere": hemisphere.upper()[0],
        "usedElectrodes": [t.upper()[0] if sim_type != "Chronic" else t.upper() for t in trajectories.keys()],
        "stimulation_mode": df_side["stimulation_mode"].iloc[0].lower(),
        "trajectoryTransform": desorient_transform,
        "electrodePositions": [list(t.keys()) for t in trajectories.values()],
        "amplitudesForPositions": [list(t.values()) for t in trajectories.values()],
        "conductivityMatrixCenter": elma_coords["pre_center"],
        "conductivityMatrixExtend": elma_coords["pre_side"],
        "pathConductivityMatrix": f"./{cond_matrix_name}",
        "pathVTUFiles": f".{results_folder}",
        "pathComsolModel": electrode_name,
        "baseFileName": base_file_name,
    }

    if sim_type != "Chronic":
        config["electrodeOffsets"] = 2  # todo hardcode?
        config["guideTubePosition"] = -12  # todo hardcode?

    return config


_PARAMETER_QUERIES = {
    "Intraoperativ": get_intra_simulations_parameters,
    "Postoperativ": get_post_simulations_parameters,
    "Chronic": get_chronic_simulations_parameters,
}


class EFieldConfigurationWriter(BaseProcessor):
    """Build COMSOL e-field configs for one subject from the stimulation db.

    Self-contained (no BIDS/files/save): the caller passes the resolved
    conductivity-matrix name and ELMA coordinates, plus two callables:

    * ``desorient_for(side)``       -> the trajectory transform for "L"/"R".
    * ``base_name_for(hemisphere)`` -> the ``baseFileName`` for that hemisphere.

    ``sim_type`` must be the canonical ``"Intraoperativ"`` / ``"Postoperativ"`` /
    ``"Chronic"``. ``build()`` yields ``(hemisphere, config)`` pairs (config may
    be ``None`` when no desorient transform exists).
    """

    def __init__(self, db_path, subject_id, subject_name, sim_type, elma_coords,
                 cond_matrix_name, desorient_for, base_name_for, results_folder="/efields/"):
        self.db_path = db_path
        self.subject_id = subject_id
        self.subject_name = subject_name
        self.sim_type = sim_type
        self.elma_coords = elma_coords
        self.cond_matrix_name = cond_matrix_name
        self.desorient_for = desorient_for
        self.base_name_for = base_name_for
        self.results_folder = results_folder

    def build(self):
        query = _PARAMETER_QUERIES.get(self.sim_type)
        if query is None:
            raise ValueError(
                f"sim_type must be Intraoperativ, Postoperativ or Chronic, not {self.sim_type}")

        df_subject = query(self.db_path, self.subject_id)
        if df_subject.empty:
            self._log_error(f"No stimulation data for subject {self.subject_name} in session {self.sim_type}")
            return

        for hemisphere in df_subject["hemisphere"].unique():
            if self.sim_type != "Chronic":
                df_hemisphere = df_subject[df_subject["hemisphere"] == hemisphere].sort_values("position_on_trajectory")
            else:
                df_hemisphere = df_subject[df_subject["hemisphere"] == hemisphere]
            df_hemisphere = df_hemisphere[df_hemisphere["amplitude"] != pd.NA]

            if df_hemisphere.empty:
                self._log_error(
                    f"No stimulation data for subject {self.subject_name} on hemisphere {hemisphere}")

            try:
                config = build_efield_config(
                    hemisphere, df_hemisphere, self.elma_coords, self.sim_type,
                    desorient_transform=self.desorient_for(hemisphere[0].upper()),
                    subject=self.subject_name,
                    cond_matrix_name=self.cond_matrix_name,
                    base_file_name=self.base_name_for(hemisphere),
                    results_folder=self.results_folder,
                )
            except Exception as e:
                self._log_error(
                    f"Error building config for subject {self.subject_name} on hemisphere {hemisphere}: {e}")
                continue

            yield hemisphere, config
