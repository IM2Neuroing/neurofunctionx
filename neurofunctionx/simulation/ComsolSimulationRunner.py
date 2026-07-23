import subprocess
import time
from pathlib import Path

from neurofunctionx.core.BaseProcessor import BaseProcessor
from neurofunctionx.io.socket_helper import is_port_open, reserve_open_port


class ComsolSimulationRunner(BaseProcessor):
    """Portable execution core for a COMSOL e-field simulation.

    Given a container image (SIF), the MATLAB livelink scripts and a local
    working directory holding a config file, it starts an mphserver on a free
    port, runs the MATLAB entry point against it and shuts the server down.
    The BIDS/SMB orchestration that discovers configs and moves files lives in
    the caller (see neurodatax EFieldSimulationGenerator).
    """

    def __init__(self, sif_path, livelink_path, server_port=None):
        self.sif_path = Path(sif_path)
        self.livelink_path = livelink_path
        self.server_port = server_port
        self._license_checked = False

    def check_license(self):
        """Run a throwaway MATLAB session so the license is validated once."""
        if self._license_checked:
            return
        subprocess.run(
            ["apptainer", "exec", self.sif_path, "matlab", "-nosplash", "-nodesktop", "-r", "quit"],
            check=True,
        )
        self._log("Checked running Matlab")
        self._license_checked = True

    def run(self, local_sim_dir, config_file):
        bind_params = ["-B", "/rg:/rg", "-B", f"{local_sim_dir}:/simulation"]

        socket, port = reserve_open_port() if self.server_port is None else reserve_open_port(start=self.server_port)
        self._log(f"Using port {port} for mphserver")

        server_cmd = [
            "apptainer", "exec", *bind_params,
            self.sif_path, "comsol", "mphserver", "-port", str(port),
        ]
        server_proc = subprocess.Popen(server_cmd)
        self._log(f"Started mphserver (PID {server_proc.pid})")

        time.sleep(1)
        socket.close()
        try:
            while not is_port_open(port):
                time.sleep(1)
            self._log("mphserver ready")

            matlab_cmd = [
                "apptainer", "exec", *bind_params,
                self.sif_path, "matlab", "-nosplash", "-nodesktop", "-r",
                f"addpath('{self.livelink_path}'); Application('/simulation/{config_file}', {port}); quit",
            ]
            subprocess.run(matlab_cmd, check=True)
            self._log(f"Finished MATLAB simulation for {config_file}")
        finally:
            server_proc.terminate()
            server_proc.wait()
            self._log(f"Stopped mphserver (PID {server_proc.pid})\n")
