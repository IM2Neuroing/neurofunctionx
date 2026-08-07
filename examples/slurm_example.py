"""
Submitting work to SLURM with :class:`neurofunctionx.cluster.SlurmRunner`.

Run this on a **login node** -- it calls ``sbatch``, so it needs to live where
sbatch exists. It is a driver, not a job: it submits and exits (or waits, if you
ask it to).

    pip install "neurofunctionx[slurm]"
    python slurm_example.py

Edit the paths and the partition name at the top, then work through the
examples; each one is independent, so comment out what you do not want.
"""
import logging
from pathlib import Path

from neurofunctionx.cluster import SlurmRunner
from neurofunctionx.core.BaseProcessor import BaseProcessor
from neurofunctionx.io.sitk.data_handler import save_volume
from neurofunctionx.io.sitk.registration import register_brains

# One call sets up logging for the driver. Jobs configure their own on the
# compute node, so their output looks the same as this.
BaseProcessor.configure_logging(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

DATA = Path("/rg/data")
OUT = Path("/rg/derivatives/registration")

# log_dir MUST be on shared storage. submitit writes the pickled function there
# for the compute node to read back; a node-local path fails confusingly.
LOG_DIR = Path("/rg/logs/registration")

PARTITION = "cpu"          # None to let SLURM pick the default
SUBJECTS = ["sub-01", "sub-02", "sub-03"]


# --------------------------------------------------------------------------- #
# The work
# --------------------------------------------------------------------------- #
# These run on a compute node in a fresh interpreter. Arguments and return
# values make the trip as pickles, so paths and plain data are fine; open file
# handles, sockets and database connections are not.

def register_subject(subject: str) -> str:
    """Register one subject's T1 into their high-resolution scan."""
    result = register_brains(
        fixed=DATA / subject / "anat" / f"{subject}_highres.nii.gz",
        moving=DATA / subject / "anat" / f"{subject}_T1w.nii.gz",
        type_of_transform="SyNRA",
        verbose=True,          # ANTs' own progress, so a slow job looks different to a stuck one
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{subject}_warped.nii.gz"
    save_volume(result.warped_image, str(out_path))
    return str(out_path)


def count_outputs(directory: str) -> int:
    return len(list(Path(directory).glob("*_warped.nii.gz")))


def main():
    # A runner describes one shape of job. Reuse it for everything that fits
    # that shape; ask for a variant with with_resources().
    runner = SlurmRunner(
        log_dir=LOG_DIR,
        cpus=8,
        mem="16G",             # "512M", "64G" or a number of GiB
        time="04:00:00",       # or "4h", "90m", "1-12:00:00", or minutes
        partition=PARTITION,
    )

    # ---- 1. one job ------------------------------------------------------ #
    # Returns immediately with a handle; the job is now queued.
    job = runner.submit(register_subject, SUBJECTS[0])
    BaseProcessor.log(f"queued as {job.job_id}, state={job.state}")

    # .result() blocks until it finishes and gives you the actual return value.
    # An exception inside the job is re-raised here, with the worker traceback.
    BaseProcessor.log(f"first subject wrote {job.result()}")

    # ---- 2. one job array over many inputs ------------------------------- #
    # Much kinder to the scheduler than N separate submissions. Every element
    # gets the same resource request, so size it for the biggest one.
    jobs = runner.map(register_subject, SUBJECTS)

    # All of them start as soon as SLURM has room. To keep only a few running at
    # a time -- a busy shared partition, or work that hammers the fileserver --
    # cap the array. The whole array is still submitted at once; SLURM releases
    # the rest as earlier elements finish.
    #   polite = runner.with_resources(array_parallelism=4)
    #   jobs = polite.map(register_subject, SUBJECTS)

    # ---- 3. a step that waits for those to succeed ------------------------ #
    # afterok by default: it starts only if every element succeeded. Use
    # state="afterany" to run regardless of outcome (cleanup, reporting).
    tally = (
        runner
        .with_resources(cpus=1, mem="2G", time="10m")   # trivial step, trivial ask
        .after(*jobs)
        .submit(count_outputs, str(OUT))
    )

    # ---- 4. a heavier variant of the same work --------------------------- #
    # with_resources returns a NEW runner; `runner` above is unchanged. Any
    # keyword it does not recognise is passed to sbatch as-is, so
    # constraint="...", exclude="hpc11" or mail_type="FAIL" all work.
    big = runner.with_resources(mem="64G", time="12:00:00")
    big.submit(register_subject, SUBJECTS[-1])

    # ---- 5. collect ------------------------------------------------------ #
    # ignore_errors keeps one bad subject from throwing away the whole sweep;
    # failures are logged and come back as None.
    for path in runner.results(jobs, ignore_errors=True):
        BaseProcessor.log(f"got {path}")

    BaseProcessor.log(f"{tally.result()} warped volumes in {OUT}")

    # Nothing forces you to wait: drop the .result() calls and the driver exits
    # while the jobs keep running. Watch them with `squeue -u $USER`, and read
    # their output in LOG_DIR.


if __name__ == "__main__":
    main()
