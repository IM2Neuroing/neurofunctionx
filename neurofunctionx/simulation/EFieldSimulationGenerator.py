"""Portable e-field simulation batch runner.

Drives the discovery -> download -> simulate -> upload -> cleanup workflow over an
SMB share, delegating each COMSOL run to a ``ComsolSimulationRunner``. It holds no
BIDS/config knowledge: the SMB helper, the runner and all paths are injected by the
caller (see ``neurodatax`` for the BIDS-facing wrapper that resolves them).

``smb`` must expose: ``list_directory(path[, predicate])``, ``pull_file``,
``push_file``, ``download_conductivity_matrix``, ``upload_efields``,
``remote_efields_exist`` and ``create_remote_directory``.
"""
from datetime import datetime
import shutil
from pathlib import Path

from neurofunctionx.core.BaseProcessor import BaseProcessor
from neurofunctionx.io.data_helper import get_bids_file_parts


class EFieldSimulationGenerator(BaseProcessor):

    def __init__(self, runner, smb, remote_base_path: Path, local_sim_dir: Path,
                 only_subjects=None, only_session=None, overwrite=False):
        self.runner = runner
        self.smb = smb
        self.remote_base_path = Path(remote_base_path)
        self.local_sim_dir = Path(local_sim_dir)
        self.only_subjects = only_subjects
        self.only_session = only_session
        self.overwrite = overwrite

    def run_locally(self):
        """Iterate the patients found on the remote share and simulate each."""
        self._log("Running EFieldSimulationGenerator...")
        try:
            patient_entries = self.smb.list_directory(self.remote_base_path)
            if len(patient_entries) == 0:
                self._log_error(f"No patients found in remote directory: {self.remote_base_path}. ")
                return

            for entry in patient_entries:
                if not self._should_process_patient(entry):
                    continue
                self._process_single_patient(entry.filename, self.remote_base_path / entry.filename)
        except Exception as e:
            self._log_error(f"An error occurred: {e}")

        self._log("Exiting EFieldSimulationGenerator...")

    def _should_process_patient(self, entry):
        if not entry.isDirectory or entry.filename in (".", ".."):
            return False
        if self.only_subjects and entry.filename.lstrip("sub-") not in self.only_subjects:
            return False
        return True

    def _process_single_patient(self, patient_name, remote_patient_path: Path):
        """Discovery -> Setup -> Download -> Simulate -> Upload -> Cleanup for one patient."""
        remote_configs_dir = remote_patient_path / "Simulations"

        current_sim_dir = self.local_sim_dir / patient_name
        current_sim_dir.mkdir(parents=True, exist_ok=True)

        try:
            config_files = self.smb.list_directory(remote_configs_dir, lambda f: f.endswith("_config.json"))
        except Exception as e:
            self._log_error(f"Error finding configs for {patient_name}: {e}")
            self._update_log_entry_and_push(patient_name, "ERROR", f"Error finding configs: {e}",
                                            current_sim_dir, remote_configs_dir)
            return

        if not config_files:
            self._log_error(f"No config found for {patient_name}, skipping")
            self._update_log_entry_and_push(patient_name, "INFO", "No config found, skipping",
                                            current_sim_dir, remote_configs_dir)
            return

        # copy mph models to the local dir (matlab can't share a single .mph)
        models_dir = self.local_sim_dir.parent.parent / "models"  # todo copy only required model
        for mph_file in models_dir.glob("*.mph"):
            shutil.copy2(mph_file, current_sim_dir)

        try:
            self.smb.download_conductivity_matrix(remote_configs_dir, current_sim_dir)
        except Exception as e:
            self._log_error(f"Error finding Conductivity Matrix for {patient_name}: {e}")
            self._update_log_entry_and_push(patient_name, "ERROR", f"Error finding Conductivity Matrix: {e}",
                                            current_sim_dir, remote_configs_dir)
            return

        for config_file in config_files:
            file_parts = get_bids_file_parts(config_file)
            if self.only_session and file_parts["session"] not in self.only_session:
                continue
            self._log(f"Processing {patient_name} with config {file_parts['structure']}")

            self.smb.pull_file(remote_configs_dir / config_file, current_sim_dir / config_file)

            if not self.overwrite and self.smb.remote_efields_exist(remote_patient_path, file_parts["structure"]):
                self._update_log_entry_and_push(patient_name, "INFO", f"EFields for config {config_file} already exist",
                                                current_sim_dir, remote_configs_dir)
                self._log(f"Skipping {config_file}, efields already exist")
                continue

            self.runner.run(current_sim_dir, config_file)
            self._update_log_entry_and_push(patient_name, "SUCCESS", f"Simulation for {config_file} ran successfully",
                                            current_sim_dir, remote_configs_dir)

        self.smb.upload_efields(current_sim_dir / "efields", remote_configs_dir)

        shutil.rmtree(current_sim_dir)
        self._log(f"Cleaned up local directory: {current_sim_dir.parent}")

    def _update_log_entry_and_push(self, patient_name, status, message, local_sim_dir: Path,
                                   remote_base: Path, log_filename="efields.log"):
        status = status.upper()
        if status not in {"SUCCESS", "ERROR", "INFO"}:
            raise ValueError("status must be 'SUCCESS', 'INFO' or 'ERROR'")

        remote_efields_dir = remote_base / "efields"
        remote_log_path = remote_efields_dir / log_filename
        local_log_path = local_sim_dir / log_filename

        self.smb.create_remote_directory(remote_efields_dir)

        remote_filenames = {e.filename for e in self.smb.list_directory(remote_efields_dir) if not e.isDirectory}
        if log_filename in remote_filenames:
            self.smb.pull_file(remote_log_path, local_log_path)
        else:
            local_log_path.touch(exist_ok=True)

        timestamp = datetime.now().strftime("%H:%M:%S")
        with local_log_path.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp} - {patient_name} | {status} | {message}\n")

        self.smb.push_file(local_log_path, remote_log_path)
