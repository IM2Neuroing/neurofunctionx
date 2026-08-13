"""
What the current job is actually allowed to use.

A compute job sees the whole machine: ``os.cpu_count()`` reports every core on
the node, not the cores cgroups granted the allocation, and ``/tmp`` may be a
tmpfs sized for the hardware rather than for the job's memory limit. Sizing work
against the hardware instead of the allocation is the most common way a cluster
job dies, so everything needed to ask "what do *I* have?" lives here.

Nothing in this module imports a processing library, and nothing here is
expensive. That is deliberate: several libraries (ITK above all) size a thread
pool the first time they are touched, so their thread counts must be decided
*before* they are imported -- which means the code that answers "how many CPUs?"
must be safe to call first. Keep this module, and anything ``cluster/__init__``
pulls in, free of ITK, ANTs, VTK and friends.
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

from neurofunctionx.core.BaseProcessor import BaseProcessor


def job_id() -> Optional[str]:
    """The SLURM job id, or ``None`` when not running under SLURM."""
    return os.environ.get("SLURM_JOB_ID")


def allocated_cpus() -> int:
    """CPUs this job may actually use.

    ``SLURM_CPUS_PER_TASK`` is authoritative inside a job step; the affinity mask
    is the fallback. Both are far smaller than ``os.cpu_count()`` on a shared
    node.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(var)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def filesystem_type(path: Union[str, Path]) -> Optional[str]:
    """Filesystem type backing ``path``, e.g. ``"tmpfs"`` or ``"ext4"``.

    Returns ``None`` where ``/proc/mounts`` is unreadable (any non-Linux host).
    Worth checking before writing large intermediates: on a tmpfs they are RAM.
    """
    try:
        with open("/proc/mounts") as handle:
            mounts = handle.read()
        target = os.path.realpath(path)
    except OSError:
        return None

    # Longest matching mount point wins.
    best_len, best_type = -1, None
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point, fs_type = fields[1], fields[2]
        if (target == mount_point or target.startswith(mount_point.rstrip("/") + "/")) \
                and len(mount_point) > best_len:
            best_len, best_type = len(mount_point), fs_type
    return best_type


def describe_environment(label: str = "Job environment",
                         extra: Optional[Dict[str, object]] = None) -> dict:
    """Log (and return) the resources this process is really running under.

    Worth calling once at the top of a batch job: it puts the difference between
    the node's hardware and the job's allocation into the log, where it can be
    read back after the fact.

    Parameters
    ----------
    label : str
        Prefix for the logged line, so a job can say which stage it describes.
    extra : dict, optional
        Merged into both the logged line and the returned dict. For settings a
        caller derived from these numbers -- a library's thread count, say --
        which belong on the same line as the allocation they came from.
    """
    tmpdir = tempfile.gettempdir()
    info = {
        "node_cpus": os.cpu_count(),
        "allocated_cpus": allocated_cpus(),
        "tmpdir": tmpdir,
        "tmpdir_fstype": filesystem_type(tmpdir),
        "slurm_job_id": job_id(),
    }
    if extra:
        info.update(extra)

    extra_text = "".join(f"{key}={value} " for key, value in (extra or {}).items())
    BaseProcessor.log(
        f"{label}: job={info['slurm_job_id']} "
        f"node_cpus={info['node_cpus']} allocated_cpus={info['allocated_cpus']} "
        f"{extra_text}"
        f"tmpdir={tmpdir} ({info['tmpdir_fstype'] or 'unknown fs'})"
    )
    if info["tmpdir_fstype"] == "tmpfs":
        BaseProcessor.log_warn(
            f"Temp directory {tmpdir} is a tmpfs: files written there consume RAM "
            "and count against the job's memory limit. Export TMPDIR to node-local "
            "disk, or give whatever writes intermediates an explicit directory on "
            "scratch."
        )
    return info