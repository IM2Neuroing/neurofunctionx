"""
High-quality registration of two skull-stripped brains.

Wraps ANTsPy so a single call performs the rigid -> affine -> SyN deformable
registration with the cross-correlation (CC) metric that is the research-grade
standard for brain-to-brain warping. Brains may be ``sitk.Image`` objects or any
path readable by :func:`load_volume`; warped results come back as ``sitk.Image``.

Requires the optional ``registration`` extra::

    pip install "neurofunctionx[registration]"      # installs antspyx

Running on a cluster
--------------------
Two failure modes dominate on SLURM, and both are handled here:

* **Thread oversubscription.** ITK sizes its thread pool from the node's core
  count, not the cores cgroups granted the job, so a 64-core node with
  ``--cpus-per-task=8`` gets 64 ANTs worker threads. The CC metric allocates
  per-thread buffers, so peak memory grows with that wrong count while 8 real
  cores thrash. This module pins the thread count before ANTsPy is imported.
* **Temporary files on tmpfs.** ANTs streams warp fields through ``outprefix``,
  which defaults into ``$TMPDIR``/``/tmp``. Where that is a tmpfs, every byte is
  RAM charged to the job. Pass ``work_dir=`` pointing at real scratch.

The facts about the allocation itself -- CPU count, job id, which filesystem a
path is on -- come from :mod:`neurofunctionx.cluster.resources`, which knows
nothing about ITK or ANTs and is therefore safe to consult before they load.
This module only turns those numbers into ITK settings.

Call :func:`log_registration_environment` once at job start to get the resolved
thread count and temp-directory filesystem into the log. Logging goes through
:class:`BaseProcessor`, so ``BaseProcessor.configure_logging()`` controls it.
"""
import multiprocessing
import os
import platform
import select
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import SimpleITK as sitk

from neurofunctionx.cluster.resources import (
    allocated_cpus,
    describe_environment,
    filesystem_type,
)
from neurofunctionx.core.BaseProcessor import BaseProcessor
from neurofunctionx.io.sitk.data_handler import load_volume
from neurofunctionx.io.sitk.image_transform import resample_to_spacing

ImageOrPath = Union[sitk.Image, str, Path]

_GIB = 1024 ** 3

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
        :func:`apply_registration` to move associated images or labels from
        moving space into fixed space.
    inverse_transforms : list[str]
        Transform files for the reverse direction (*fixed -> moving*).
    """
    warped_image: sitk.Image
    forward_transforms: List[str]
    inverse_transforms: List[str]


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

def _configure_threads(num_threads: Optional[int] = None) -> int:
    """Pin ITK/OpenMP thread counts to the allocation, before ANTsPy loads.

    ITK caches its global default the first time its multithreader is touched,
    which happens as the ANTs shared library initialises. Setting the environment
    after ``import ants`` has no effect, so we warn instead of pretending.
    """
    global _threads_configured

    threads = max(1, int(num_threads or allocated_cpus()))

    node_cpus = os.cpu_count() or threads
    if num_threads is None and node_cpus > threads:
        BaseProcessor.log(
            f"Limiting ITK/ANTs to {threads} thread(s); the node reports "
            f"{node_cpus} CPUs, which would oversubscribe the allocation and "
            f"inflate peak memory."
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

    Worth calling once at the top of a batch job: it puts the difference between
    the node's hardware and the job's allocation into the log, with the resolved
    ITK thread count beside it.
    """
    info = describe_environment(
        label="Registration environment",
        extra={"itk_threads": os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")},
    )
    if info["tmpdir_fstype"] == "tmpfs":
        BaseProcessor.log_warn(
            "ANTs warp fields are several hundred MB each, so that applies to "
            "this run: pass work_dir=<scratch on disk> to register_brains."
        )
    return info


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this process when its parent dies.

    The last line of defence against a monitor that outlives the job. Every
    other stop signal -- the pipe, SIGTERM, the getppid() check -- needs this
    process to be running Python to act on it; ``PR_SET_PDEATHSIG`` does not.
    Linux-only and best effort: a failure here costs nothing, because the checks
    in the loop still apply.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # 1 == PR_SET_PDEATHSIG
    except Exception:  # pragma: no cover - platform / libc quirk
        pass


def _monitor_loop(target_pid: int, stage: str, interval: float,
                  stop_fd: int, parent_fd: int) -> None:
    """Body of the monitor process: report until the parent says stop.

    Every exit from here goes through ``os._exit``. Returning normally would
    hand control back to ``multiprocessing``'s bootstrap, which runs
    ``util._exit_function()`` -- and ``util._finalizer_registry`` is inherited
    through the fork and *not* cleared, so this process would run finalizers
    that belong to the parent (a pool's ``terminate``, a shared-memory unlink)
    against resources it does not own. Those block indefinitely, which is how a
    monitor that was told to stop ends up wedged instead of dead.
    """
    os.close(parent_fd)  # our copy of the write end, or we never see the EOF

    # A fork inherits the parent's signal handlers, and submitit installs one
    # that swallows SIGTERM so SLURM cannot kill a job mid-step. Inherited here
    # it makes process.terminate() a no-op: the monitor outlives the block,
    # multiprocessing's atexit handler joins it without a timeout, and the job
    # hangs until the wall clock kills it. Take the default handlers back.
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            try:
                signal.signal(signum, signal.SIG_DFL)
            except (OSError, ValueError):  # pragma: no cover - platform quirk
                pass
    _die_with_parent()

    started = time.monotonic()
    while True:
        # Wait on the pipe rather than the clock: the parent closes its end when
        # the registration returns, which wakes us at once and works even where
        # signals are being intercepted.
        try:
            if select.select([stop_fd], [], [], interval)[0]:
                os._exit(0)
        except (InterruptedError, OSError):
            os._exit(0)
        # The parent exiting reparents us to init; stop rather than log forever.
        if os.getppid() != target_pid or not os.path.exists(f"/proc/{target_pid}"):
            os._exit(0)
        elapsed = (time.monotonic() - started) / 60.0
        BaseProcessor.log(f"{stage} still running after {elapsed:.1f} min")


@contextmanager
def _heartbeat(stage: str, interval: float):
    """Log a line every ``interval`` seconds while the block runs.

    ANTs does its work inside a single opaque call, so without this the log just
    stops and a slow run is indistinguishable from a deadlocked one. Set
    ``interval<=0`` to disable.

    The monitor runs in a forked *process*, on the assumption that ANTsPy spends
    the whole registration in compiled code that never releases the GIL, so a
    thread would not be scheduled to report. A process is scheduled by the kernel
    either way, and inherits the logging handlers through the fork.

    That assumption has not been measured. If ANTsPy does release the GIL, a
    plain daemon thread would do the same job without a fork, and with it would
    go the inherited signal handlers, the stop pipe and the daemon-child join at
    interpreter exit -- every sharp edge in this file. Worth an afternoon.

    ``register_brains`` is the only caller and refuses to run off Linux, so fork
    is always available here; there is deliberately no fallback.
    """
    if interval <= 0:
        yield
        return

    context = multiprocessing.get_context("fork")
    sys.stdout.flush()
    sys.stderr.flush()
    stop_fd, parent_fd = os.pipe()
    process = context.Process(
        target=_monitor_loop,
        args=(os.getpid(), stage, interval, stop_fd, parent_fd),
        name="registration-heartbeat",
        daemon=True,
    )
    process.start()
    os.close(stop_fd)  # the monitor owns the read end from here on
    try:
        yield
    finally:
        # Closing the pipe is the primary stop signal; the escalation below only
        # matters if the monitor is wedged. Never leave this block with the child
        # alive: multiprocessing joins daemon children at exit without a timeout,
        # so a survivor hangs the whole job.
        os.close(parent_fd)
        process.join(timeout=5.0)
        for escalate in (process.terminate, process.kill):
            if not process.is_alive():
                break
            escalate()
            process.join(timeout=5.0)
        if process.is_alive():
            BaseProcessor.log_warn(
                "The heartbeat monitor did not exit. It is a daemon process, so "
                "the interpreter will try to join it at exit and may hang."
            )


# --------------------------------------------------------------------------- #
# ANTs plumbing
# --------------------------------------------------------------------------- #

def _effective_spacing(image: sitk.Image) -> float:
    """A single resolution figure for a possibly anisotropic voxel.

    The geometric mean of the sides -- the isotropic voxel of equal volume.
    ``min()`` would take the *best* axis as the image's resolution, which for a
    clinical 0.625 x 0.625 x 1.3 mm scan claims 0.625 mm detail that exists only
    in plane; the geometric mean calls that scan 0.80 mm, which is the honest
    number to compare grids with.
    """
    spacing = image.GetSpacing()
    volume = 1.0
    for side in spacing:
        volume *= float(side)
    return volume ** (1.0 / len(spacing))


def _resolve_registration_spacing(fixed: sitk.Image, moving: sitk.Image, requested):
    """Decide the grid the registration should run on.

    ``"moving"`` (the default) drops the fixed grid to the moving image's
    resolution. Registering a coarse image into a much finer grid cannot recover
    detail that is not there, but it makes every displacement field -- and the CC
    metric's per-voxel work -- scale with the fine grid.
    """
    if requested is None:
        return None
    if requested == "moving":
        fixed_spacing = _effective_spacing(fixed)
        moving_spacing = _effective_spacing(moving)
        if moving_spacing <= fixed_spacing * 1.5:
            return None  # resolutions already comparable
        return moving_spacing
    return requested


def _describe(name: str, image: sitk.Image) -> None:
    """Log the geometry of an input image."""
    size = image.GetSize()
    voxels = 1
    for dim in size:
        voxels *= dim
    spacing = tuple(round(s, 4) for s in image.GetSpacing())
    BaseProcessor.log(
        f"{name} image: size={size} spacing={spacing} mm "
        f"voxels={voxels / 1e6:.1f}M ({voxels * 4 / _GIB:.2f} GiB as float32)"
    )


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


def _load_sitk(image: ImageOrPath, label: str = "image") -> sitk.Image:
    if not isinstance(image, sitk.Image):
        BaseProcessor.log(f"Loading {label} from {image}")
        image = load_volume(image)
    return sitk.Cast(image, sitk.sitkFloat32)


def _to_ants(image: ImageOrPath, work_dir: Optional[Path] = None, label: str = "image"):
    """Load any supported input into an ANTs image via SimpleITK.

    Routing through :func:`load_volume` gives uniform format handling and a
    consistent float32 cast, and the on-disk NIfTI round-trip preserves the
    physical geometry exactly, avoiding in-memory direction-matrix conversions.

    The temporary file is deliberately *uncompressed*: gzip dominates the cost of
    the round-trip (25 s versus 1 s for a 288M-voxel volume) and the file is
    deleted a moment later.
    """
    ants = _require_ants()
    started = time.monotonic()
    image = _load_sitk(image, label)
    _describe(label, image)

    tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False,
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
    return ants_image


def _ants_to_sitk(ants_image, work_dir: Optional[Path] = None) -> sitk.Image:
    ants = _require_ants()
    tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False,
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
        BaseProcessor.log_warn(
            f"No work_dir given: ANTs will write the transform files under "
            f"{tmpdir}, which is wiped when the job ends. Use them before this "
            f"process exits, or pass work_dir=<path on scratch> so they persist."
        )
        if filesystem_type(tmpdir) == "tmpfs":
            BaseProcessor.log_warn(
                f"{tmpdir} is also a tmpfs, so those intermediates are held in RAM "
                "and charged to the job's memory limit."
            )
        return None
    path = Path(work_dir)
    path.mkdir(parents=True, exist_ok=True)
    BaseProcessor.log(f"Using work_dir {path} ({filesystem_type(path) or 'unknown fs'})")
    return path


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def register_brains(
        fixed: ImageOrPath,
        moving: ImageOrPath,
        type_of_transform: str = "SyNCC",
        registration_spacing: Union[str, float, None] = "moving",
        work_dir: Optional[Union[str, Path]] = None,
        num_threads: Optional[int] = None,
        verbose: bool = False,
        heartbeat: float = 60.0,
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
        with the cross-correlation metric -- highest quality, slowest, and by far
        the most memory-hungry. Use ``"SyNRA"`` or ``"SyN"`` for a much lighter
        mutual-information variant, or ``"Affine"`` / ``"Rigid"`` for linear-only
        alignment.
    registration_spacing : "moving" | float | None
        Voxel size, in mm, that the *registration* runs at. ``"moving"`` (the
        default) drops the fixed grid to the moving image's resolution whenever
        the fixed image is more than 1.5x finer, and does nothing otherwise.
        Both resolutions are measured as the geometric mean of the voxel sides,
        so an anisotropic scan is judged on its true detail rather than on its
        best axis.

        Cost scales with the fixed image's voxel count, which scales with the
        cube of the resolution: registering into a 0.25 mm grid instead of a 1 mm
        one is 64x the memory and time, and adds no accuracy at all if the moving
        image is 1 mm. Pass a float to choose the resolution yourself, or ``None``
        to register on the fixed image's native grid.

        The transforms come back in physical space, so they stay valid for the
        full-resolution image, and ``warped_image`` is always returned on the
        *original* fixed grid -- lowering this costs solver time and memory, not
        output resolution. Hand the original fixed image to
        :func:`apply_registration` as ``reference`` to warp anything else at
        full resolution too.
    work_dir : str | Path, optional
        Directory for intermediate NIfTI files and ANTs' transform output. Point
        this at real scratch storage on a cluster; the default temp directory is
        often a tmpfs. The transform files referenced by the result live here, so
        the directory must outlive the returned object.
    num_threads : int, optional
        ITK/ANTs thread count. Defaults to the SLURM allocation
        (``SLURM_CPUS_PER_TASK``) or the CPU affinity mask -- *not*
        ``os.cpu_count()``, which reports every core on the node.
    verbose : bool
        Forward ``verbose=True`` to ANTs for per-stage, per-iteration progress.
        The best way to tell a slow registration from a stalled one.
    heartbeat : float
        Seconds between "still running" log lines. ``0`` disables it.
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
    fixed_image = _load_sitk(fixed, "fixed")
    moving_image = _load_sitk(moving, "moving")

    spacing = _resolve_registration_spacing(fixed_image, moving_image, registration_spacing)
    registration_fixed = fixed_image
    if spacing is not None:
        BaseProcessor.log(
            f"Resampling the fixed grid from {_effective_spacing(fixed_image):.3g} mm to "
            f"{spacing:.3g} mm for registration. The transforms are in physical space, "
            f"so they stay valid for the full-resolution image, and the warped image is "
            f"returned on the original grid."
        )
        registration_fixed = resample_to_spacing(fixed_image, spacing)

    fixed_ants = _to_ants(registration_fixed, work_path, "fixed")
    moving_ants = _to_ants(moving_image, work_path, "moving")

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

    warped = reg["warpedmovout"]
    if spacing is not None:
        # Registering on a coarser grid saves work in the *solver*; it must not
        # cost the caller resolution in the output. The transforms are in
        # physical space, so pushing the moving image through them onto the
        # original fixed grid gives back exactly the volume a native-grid
        # registration would have produced, for one extra interpolation.
        BaseProcessor.log("Warping the moving image onto the original fixed grid")
        warped = ants.apply_transforms(
            fixed=_to_ants(fixed_image, work_path, "fixed (full resolution)"),
            moving=moving_ants,
            transformlist=reg["fwdtransforms"],
            interpolator="linear",
        )

    result = RegistrationResult(
        warped_image=_ants_to_sitk(warped, work_path),
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
    reference_ants = _to_ants(reference, work_path, "reference")
    moving_ants = _to_ants(moving, work_path, "moving")

    started = time.monotonic()
    out = ants.apply_transforms(
        fixed=reference_ants,
        moving=moving_ants,
        transformlist=transforms,
        interpolator=interpolator,
    )
    BaseProcessor.log(f"Applied transforms in {time.monotonic() - started:.1f} s")
    return _ants_to_sitk(out, work_path)