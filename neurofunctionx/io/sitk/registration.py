"""
High-quality registration of two skull-stripped brains.

Wraps ANTsPy (the Python bindings for ANTs) so a single call performs the same
rigid -> affine -> SyN deformable registration with the cross-correlation (CC)
metric that is the research-grade standard for brain-to-brain warping. Inputs
and outputs stay interoperable with the rest of the project: brains may be
``sitk.Image`` objects or any path readable by :func:`load_volume`, and warped
results are returned as ``sitk.Image``.

Requires the optional ``registration`` extra::

    pip install "neurofunctionx[registration]"      # installs antspyx

Running on a cluster
--------------------
Registration is the most resource-hungry step in the stack; on SLURM two
failure modes dominate and both are handled here (see :func:`register_brains`):

* **Thread oversubscription.** ITK -- and therefore ANTs -- sizes its thread
  pool from ``std::thread::hardware_concurrency()``, which reports *all* cores
  of the compute node, not the cores cgroups actually granted the job. On a
  64-core node with ``--cpus-per-task=8`` ANTs starts 64 worker threads; the CC
  metric allocates per-thread neighborhood buffers, so peak memory grows with
  that (wrong) thread count while the 8 real cores thrash. The job looks stuck
  and then hits the memory limit. This module pins the thread count to the
  SLURM allocation before ANTsPy is imported.
* **Temporary files on tmpfs.** ANTs streams warp fields through ``outprefix``,
  which defaults to a name in ``$TMPDIR``/``/tmp``. Where that path is a tmpfs
  (common on diskless nodes) every written byte is RAM and is charged to the
  job. Pass ``work_dir=`` pointing at real scratch storage.

Call :func:`log_registration_environment` once at job start to get the resolved
thread count, cgroup memory limit and temp-directory filesystem in the log.
Logging goes through :class:`BaseProcessor` like everywhere else in the
package, so ``BaseProcessor.configure_logging()`` controls it.
"""
import os
import platform
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import SimpleITK as sitk

from neurofunctionx.core.BaseProcessor import BaseProcessor
from neurofunctionx.io.sitk.data_handler import load_volume

ImageOrPath = Union[sitk.Image, str, Path]

_GIB = 1024 ** 3

# Bytes of peak resident memory per voxel of the fixed image, measured for
# SyNCC at default settings. SyN/MI sits at roughly a third of this. Used only
# to emit an up-front warning, never to change behaviour.
_SYNCC_BYTES_PER_VOXEL = 400.0

_threads_configured = False


@dataclass
class RegistrationResult:
    """
    Result of :func:`register_brains`.

    Attributes
    ----------
    warped_image : sitk.Image
        ``moving`` resampled into the ``fixed`` grid.
    forward_transforms : list[str]
        ANTs transform files mapping *moving -> fixed*. Pass to
        :func:`apply_registration` (or ``ants.apply_transforms``) to move
        associated images/labels from moving space into fixed space.
    inverse_transforms : list[str]
        Transform files for the reverse direction (*fixed -> moving*).
    """
    warped_image: sitk.Image
    forward_transforms: List[str]
    inverse_transforms: List[str]


# --------------------------------------------------------------------------- #
# Resource introspection
# --------------------------------------------------------------------------- #

def _read_text(path: str) -> Optional[str]:
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def _process_memory() -> dict:
    """Current and peak resident set size of this process, in bytes.

    Read from ``/proc/self/status`` so no extra dependency (psutil) is needed;
    returns an empty dict on platforms without procfs.
    """
    status = _read_text("/proc/self/status")
    if status is None:
        return {}
    wanted = {"VmRSS": "rss", "VmHWM": "peak_rss", "VmSize": "vsize"}
    out = {}
    for line in status.splitlines():
        key, _, rest = line.partition(":")
        if key in wanted:
            parts = rest.split()
            if len(parts) >= 2 and parts[1] == "kB":
                out[wanted[key]] = int(parts[0]) * 1024
    return out


def cgroup_memory_limit() -> Optional[int]:
    """Memory limit the job is actually held to, in bytes, or ``None``.

    SLURM enforces ``--mem`` through cgroups, so this is the number the OOM
    killer compares against -- not the node's total RAM that ``free`` reports.
    """
    for path in (
            "/sys/fs/cgroup/memory.max",                    # cgroup v2
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        raw = _read_text(path)
        if raw is None or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # An unset v1 limit is reported as a huge sentinel value.
        if value < (1 << 62):
            return value
    return None


def _filesystem_type(path: Union[str, Path]) -> Optional[str]:
    """Filesystem type backing ``path``, e.g. ``"tmpfs"`` or ``"ext4"``."""
    mounts = _read_text("/proc/mounts")
    if mounts is None:
        return None
    try:
        target = os.path.realpath(path)
    except OSError:
        return None
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


def _available_cpus() -> int:
    """CPUs this job may actually use.

    ``SLURM_CPUS_PER_TASK`` is authoritative inside a job step; the CPU
    affinity mask is the fallback (and is what cgroups pin us to). Both are far
    smaller than ``os.cpu_count()`` on a shared node.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(var)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _configure_threads(num_threads: Optional[int] = None) -> int:
    """Pin ITK/OpenMP thread counts to the allocation, before ANTsPy loads.

    ITK caches its global default thread count the first time its multithreader
    is touched, which happens while the ANTs shared library initialises. The
    environment therefore has to be set *before* ``import ants`` -- afterwards
    it has no effect, so we warn instead of pretending it worked.
    """
    global _threads_configured

    threads = num_threads or _available_cpus()
    threads = max(1, int(threads))

    node_cpus = os.cpu_count() or threads
    if num_threads is None and node_cpus > threads:
        BaseProcessor.log(
            f"Limiting ITK/ANTs to {threads} thread(s); the node reports "
            f"{node_cpus} CPUs, and letting ITK size its pool from that would "
            f"oversubscribe the allocation and inflate peak memory."
        )

    already_imported = "ants" in sys.modules
    preset = os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")

    if preset is None:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    elif preset != str(threads):
        BaseProcessor.log(
            f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS is already set to {preset}; "
            f"leaving it as is (resolved allocation was {threads} thread(s))."
        )
        threads = int(preset) if preset.isdigit() else threads
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))

    if already_imported and preset is None:
        BaseProcessor.log_warn(
            "ANTsPy was imported before the thread limit could be applied, so "
            "ITK may still run with one thread per node core. Set "
            f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads} in the job script "
            "(or import neurofunctionx.io.sitk.registration first)."
        )

    _threads_configured = True
    return threads


def log_registration_environment() -> dict:
    """Log (and return) the resources this process is really running under.

    Worth calling once at the top of a batch job: it makes the difference
    between the node's hardware and the job's allocation visible in the log,
    which is where memory-overflow post-mortems usually start.
    """
    tmpdir = tempfile.gettempdir()
    info = {
        "node_cpus": os.cpu_count(),
        "allocated_cpus": _available_cpus(),
        "itk_threads": os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"),
        "cgroup_memory_limit": cgroup_memory_limit(),
        "tmpdir": tmpdir,
        "tmpdir_fstype": _filesystem_type(tmpdir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    info.update(_process_memory())

    limit = info["cgroup_memory_limit"]
    limit_text = f"{limit / _GIB:.1f} GiB" if limit else "unlimited/unknown"
    BaseProcessor.log(
        f"Registration environment: job={info['slurm_job_id']} "
        f"node_cpus={info['node_cpus']} allocated_cpus={info['allocated_cpus']} "
        f"itk_threads={info['itk_threads']} memory_limit={limit_text} "
        f"tmpdir={tmpdir} ({info['tmpdir_fstype'] or 'unknown fs'})"
    )
    if info["tmpdir_fstype"] == "tmpfs":
        BaseProcessor.log_warn(
            f"Temp directory {tmpdir} is a tmpfs: files written there consume "
            "RAM and count against the job's memory limit. ANTs warp fields are "
            "several hundred MB each -- pass work_dir=<scratch on disk> to "
            "register_brains, or export TMPDIR to node-local storage."
        )
    return info


def _log_memory(stage: str, verbose: bool = False) -> None:
    mem = _process_memory()
    if not mem:
        return
    limit = cgroup_memory_limit()
    share = f" ({100 * mem['rss'] / limit:.0f}% of limit)" if limit else ""
    message = (
        f"Memory after {stage}: rss={mem['rss'] / _GIB:.2f} GiB "
        f"peak={mem.get('peak_rss', mem['rss']) / _GIB:.2f} GiB{share}"
    )
    if verbose:
        BaseProcessor.log_verbose(message)
    else:
        BaseProcessor.log(message)


@contextmanager
def _heartbeat(stage: str, interval: float):
    """Log memory every ``interval`` seconds while the block runs.

    ANTs does its work inside a single opaque call. Without a heartbeat a job
    that is swapping, thrashing or simply slow is indistinguishable from one
    that has deadlocked -- the log just stops. Set ``interval<=0`` to disable.
    """
    if interval <= 0:
        yield
        return

    stop = threading.Event()
    started = time.monotonic()

    def _tick():
        while not stop.wait(interval):
            mem = _process_memory()
            elapsed = (time.monotonic() - started) / 60.0
            if mem:
                BaseProcessor.log(
                    f"{stage} still running after {elapsed:.1f} min "
                    f"(rss={mem['rss'] / _GIB:.2f} GiB, "
                    f"peak={mem.get('peak_rss', mem['rss']) / _GIB:.2f} GiB)"
                )
            else:
                BaseProcessor.log(f"{stage} still running after {elapsed:.1f} min")

    thread = threading.Thread(target=_tick, name="registration-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _describe(name: str, image: sitk.Image) -> int:
    """Log geometry of an input image and return its voxel count."""
    size = image.GetSize()
    voxels = 1
    for dim in size:
        voxels *= dim
    spacing = tuple(round(s, 4) for s in image.GetSpacing())
    BaseProcessor.log(
        f"{name} image: size={size} spacing={spacing} mm "
        f"voxels={voxels / 1e6:.1f}M ({voxels * 4 / _GIB:.2f} GiB as float32)"
    )
    return voxels


def _warn_if_oversized(voxels: int, type_of_transform: str) -> None:
    """Flag up front when the fixed grid is too large for the memory limit."""
    if not type_of_transform.startswith("SyN"):
        return

    factor = _SYNCC_BYTES_PER_VOXEL if "CC" in type_of_transform else _SYNCC_BYTES_PER_VOXEL / 3
    estimate = voxels * factor
    limit = cgroup_memory_limit()
    BaseProcessor.log(
        f"Estimated peak memory for {type_of_transform} on this grid: "
        f"~{estimate / _GIB:.1f} GiB (rough; scales linearly with voxel count, "
        f"and assumes the thread count is pinned to the allocation)."
    )
    if limit and estimate > 0.8 * limit:
        BaseProcessor.log_warn(
            f"Estimated peak (~{estimate / _GIB:.1f} GiB) is close to or above "
            f"the job's limit ({limit / _GIB:.1f} GiB). Mitigations, cheapest "
            "first: crop the images to the brain, resample to 1 mm, or switch "
            "type_of_transform to 'SyNRA' / 'SyN' (mutual information instead "
            "of the much heavier CC metric)."
        )


# --------------------------------------------------------------------------- #
# ANTs plumbing
# --------------------------------------------------------------------------- #

def _require_ants(num_threads: Optional[int] = None):
    if platform.system() != "Linux":
        raise RuntimeError(
            "register_brains is supported on Linux only; current platform is "
            f"{platform.system()}. ANTsPy has no Windows wheels, so run this on "
            "Linux (e.g. WSL or a Linux container)."
        )
    if not _threads_configured:
        _configure_threads(num_threads)
    try:
        import ants  # noqa: F401
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "register_brains requires ANTsPy. Install it with "
            '`pip install "neurofunctionx[registration]"` (or `pip install antspyx`).'
        ) from exc
    return ants


def _to_ants(image: ImageOrPath, work_dir: Optional[Path] = None, label: str = "image"):
    """Load any supported input into an ANTs image via SimpleITK.

    Routing through :func:`load_volume` gives uniform format handling (NIfTI,
    NRRD, MHA, DICOM directories) and a consistent float32 cast; the on-disk
    NIfTI round-trip preserves the physical geometry (spacing/origin/direction)
    exactly, avoiding error-prone in-memory direction-matrix conversions.
    """
    ants = _require_ants()
    started = time.monotonic()
    if not isinstance(image, sitk.Image):
        BaseProcessor.log(f"Loading {label} from {image}")
        image = load_volume(image)
    image = sitk.Cast(image, sitk.sitkFloat32)
    voxels = _describe(label, image)

    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False,
                                      dir=str(work_dir) if work_dir else None)
    tmp.close()
    try:
        sitk.WriteImage(image, tmp.name)
        ants_image = ants.image_read(tmp.name)
    finally:
        os.remove(tmp.name)
    BaseProcessor.log_verbose(
        f"Converted {label} to ANTs in {time.monotonic() - started:.1f} s"
    )
    _log_memory(f"loading {label}", verbose=True)
    return ants_image, voxels


def _ants_to_sitk(ants_image, work_dir: Optional[Path] = None) -> sitk.Image:
    ants = _require_ants()
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False,
                                      dir=str(work_dir) if work_dir else None)
    tmp.close()
    try:
        ants.image_write(ants_image, tmp.name)
        return sitk.ReadImage(tmp.name)
    finally:
        os.remove(tmp.name)


def _prepare_work_dir(work_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    if work_dir is None:
        tmpdir = tempfile.gettempdir()
        if _filesystem_type(tmpdir) == "tmpfs":
            BaseProcessor.log_warn(
                f"No work_dir given and {tmpdir} is a tmpfs; ANTs intermediates "
                "will be written to RAM and charged to the job's memory limit. "
                "Pass work_dir=<path on scratch> to avoid this."
            )
        return None
    path = Path(work_dir)
    path.mkdir(parents=True, exist_ok=True)
    BaseProcessor.log(f"Using work_dir {path} ({_filesystem_type(path) or 'unknown fs'})")
    return path


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def register_brains(
        fixed: ImageOrPath,
        moving: ImageOrPath,
        type_of_transform: str = "SyNCC",
        work_dir: Optional[Union[str, Path]] = None,
        num_threads: Optional[int] = None,
        verbose: bool = False,
        heartbeat: float = 300.0,
        **kwargs,
) -> RegistrationResult:
    """
    Register two skull-stripped brains (``moving`` -> ``fixed``) with ANTs.

    Parameters
    ----------
    fixed, moving : sitk.Image | str | Path
        The reference brain and the brain to be aligned to it.
    type_of_transform : str
        ANTs transform preset. ``"SyNCC"`` (default) runs rigid + affine + SyN
        with the cross-correlation metric -- highest quality, slowest, and by
        far the most memory-hungry. Use ``"SyNRA"`` or ``"SyN"`` for a much
        lighter mutual-information variant, or ``"Affine"`` / ``"Rigid"`` for
        linear-only alignment.
    work_dir : str | Path, optional
        Directory for intermediate NIfTI files and ANTs' transform output. Point
        this at real scratch storage on a cluster: the default temp directory is
        often a tmpfs, where intermediates are held in RAM and count towards the
        job's memory limit. The transform files referenced by the result live
        here, so the directory must outlive the returned object.
    num_threads : int, optional
        ITK/ANTs thread count. Defaults to the SLURM allocation
        (``SLURM_CPUS_PER_TASK``) or the CPU affinity mask -- *not*
        ``os.cpu_count()``, which reports every core on the node and would make
        ANTs oversubscribe and balloon its per-thread buffers.
    verbose : bool
        Forward ``verbose=True`` to ANTs so per-stage, per-iteration progress is
        printed. The single best way to tell a slow registration from a stalled
        one; costs nothing but log volume.
    heartbeat : float
        Seconds between "still running" log lines with current memory use.
        ``0`` disables it.
    **kwargs
        Forwarded verbatim to ``ants.registration`` (e.g. ``reg_iterations``,
        ``grad_step``, ``mask``, ``moving_mask``).

    Returns
    -------
    RegistrationResult
        The warped moving brain plus the forward/inverse transform file lists.
    """
    ants = _require_ants(num_threads)
    log_registration_environment()

    work_path = _prepare_work_dir(work_dir)
    if work_path is not None and "outprefix" not in kwargs:
        # Default outprefix is a mktemp name in $TMPDIR; keep warp fields with
        # the rest of the run instead, and off a possible tmpfs.
        kwargs["outprefix"] = str(work_path / f"reg_{os.getpid()}_")

    BaseProcessor.log(f"Starting {type_of_transform} registration (moving -> fixed)")
    fixed_ants, fixed_voxels = _to_ants(fixed, work_path, "fixed")
    moving_ants, _ = _to_ants(moving, work_path, "moving")
    _warn_if_oversized(fixed_voxels, type_of_transform)
    _log_memory("loading inputs")

    started = time.monotonic()
    with _heartbeat(f"{type_of_transform} registration", heartbeat):
        reg = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform=type_of_transform,
            verbose=verbose,
            **kwargs,
        )
    elapsed = (time.monotonic() - started) / 60.0
    BaseProcessor.log(f"{type_of_transform} registration finished in {elapsed:.1f} min")
    _log_memory("registration")

    result = RegistrationResult(
        warped_image=_ants_to_sitk(reg["warpedmovout"], work_path),
        forward_transforms=reg["fwdtransforms"],
        inverse_transforms=reg["invtransforms"],
    )
    BaseProcessor.log(f"Forward transforms: {result.forward_transforms}")
    return result


def apply_registration(
        moving: ImageOrPath,
        reference: ImageOrPath,
        transforms: List[str],
        interpolator: str = "linear",
        work_dir: Optional[Union[str, Path]] = None,
) -> sitk.Image:
    """
    Apply a transform list from :func:`register_brains` to another image.

    Use ``interpolator="linear"`` (or ``"bSpline"``) for intensity images and
    ``"nearestNeighbor"`` / ``"genericLabel"`` for label maps.

    Parameters
    ----------
    moving : sitk.Image | str | Path
        Image to resample (must live in the same space as the original moving
        brain, e.g. a label map of that subject).
    reference : sitk.Image | str | Path
        Image defining the target grid (typically the fixed brain).
    transforms : list[str]
        ``RegistrationResult.forward_transforms`` (moving -> fixed) or
        ``inverse_transforms`` (fixed -> moving).
    interpolator : str
        ANTs interpolator name; ``"linear"``/``"bSpline"`` for intensity images,
        ``"nearestNeighbor"``/``"genericLabel"`` for label maps.
    work_dir : str | Path, optional
        Directory for intermediate files; see :func:`register_brains`.
    """
    ants = _require_ants()
    work_path = _prepare_work_dir(work_dir)

    missing = [t for t in transforms if isinstance(t, (str, Path)) and not Path(t).exists()]
    if missing:
        raise FileNotFoundError(
            f"Transform file(s) no longer on disk: {missing}. ANTs writes them "
            "under its outprefix (a temp directory by default), so they can be "
            "cleaned up between jobs -- pass work_dir= to register_brains to "
            "keep them somewhere persistent."
        )

    BaseProcessor.log(
        f"Applying {len(transforms)} transform(s) with {interpolator} interpolation"
    )
    reference_ants, _ = _to_ants(reference, work_path, "reference")
    moving_ants, _ = _to_ants(moving, work_path, "moving")

    started = time.monotonic()
    out = ants.apply_transforms(
        fixed=reference_ants,
        moving=moving_ants,
        transformlist=transforms,
        interpolator=interpolator,
    )
    BaseProcessor.log(f"Applied transforms in {time.monotonic() - started:.1f} s")
    _log_memory("apply_transforms")
    return _ants_to_sitk(out, work_path)