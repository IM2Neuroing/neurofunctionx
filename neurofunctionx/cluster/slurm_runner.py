"""
Run ordinary Python functions as SLURM jobs.

:class:`SlurmRunner` wraps `submitit <https://github.com/facebookincubator/submitit>`_
so that submitting work to the cluster looks like calling a function::

    runner = SlurmRunner(cpus=8, mem="16G", time="04:00:00")
    job = runner.submit(register_brains, fixed_path, moving_path)
    result = job.result()          # blocks until the job finishes

submitit pickles the function and its arguments (with cloudpickle, so closures
and locally-defined functions work), writes an sbatch script, submits it, and
un-pickles whatever the function returned -- exceptions included, re-raised on
the caller side with the worker's traceback attached.

Requires the optional ``slurm`` extra and a machine where ``sbatch`` exists,
i.e. a login node::

    pip install "neurofunctionx[slurm]"      # installs submitit

Three ways to submit
--------------------
``submit`` for one call, ``map`` for the same function over many inputs as a
single job array, and ``after`` to chain jobs::

    reg = runner.submit(register_brains, fixed, moving)
    per_subject = runner.map(process, subject_list)
    runner.after(*per_subject).submit(aggregate, output_dir)

Everything returns plain submitit ``Job`` objects, so the usual ``.job_id``,
``.result()``, ``.done()``, ``.state``, ``.stdout()`` and ``.cancel()`` are all
available and documented upstream.
"""
import logging
import os
import re
import socket
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from neurofunctionx.core.BaseProcessor import BaseProcessor

# Exported into every job's shell before the worker starts. ITK sizes its thread
# pool from the node's core count unless told otherwise, which on a 256-core node
# means 256 threads fighting over the 8 cores the job was actually given -- see
# neurofunctionx.io.sitk.registration for the full story.
DEFAULT_SETUP = [
    "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-1}",
    "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
]

_MEMORY_UNITS = {"k": 1 / 1024 ** 2, "m": 1 / 1024, "g": 1.0, "t": 1024.0}


def _require_submitit():
    try:
        import submitit  # noqa: F401
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "SlurmRunner requires submitit. Install it with "
            '`pip install "neurofunctionx[slurm]"` (or `pip install submitit`).'
        ) from exc
    return submitit


def parse_memory_gb(value: Union[str, int, float]) -> float:
    """``"16G"``, ``"512M"``, ``"16"``, ``16`` -> gibibytes as a float."""
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt])?i?b?\s*", str(value), re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot read {value!r} as a memory size (try '16G', '512M' or 16)")
    amount, unit = float(match.group(1)), (match.group(2) or "g").lower()
    return amount * _MEMORY_UNITS[unit]


def parse_minutes(value: Union[str, int, float]) -> int:
    """``"04:00:00"``, ``"1-12:00:00"``, ``"4h"``, ``"90m"``, ``90`` -> minutes.

    Deliberately stricter than sbatch, which reads a bare ``"10:00"`` as ten
    minutes and ``"10:00:00"`` as ten hours -- a difference of 60x that is very
    easy to get wrong. Use an explicit unit or the full ``HH:MM:SS`` form.
    """
    if isinstance(value, (int, float)):
        return max(1, int(round(value)))

    text = str(value).strip()
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{1,2}):(\d{1,2})", text)
    if match:
        days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
        return max(1, days * 1440 + hours * 60 + minutes + (1 if seconds else 0))

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([dhm])", text, re.IGNORECASE)
    if match:
        amount, unit = float(match.group(1)), match.group(2).lower()
        return max(1, int(round(amount * {"d": 1440, "h": 60, "m": 1}[unit])))

    raise ValueError(
        f"Cannot read {value!r} as a time limit. Use 'HH:MM:SS', 'D-HH:MM:SS', "
        f"'4h', '90m', or a number of minutes."
    )


def _job_ids(jobs: Iterable[Any]) -> List[str]:
    """Parent job ids, de-duplicated, for use in a --dependency list.

    Array elements report ``"12345_3"``; a dependency has to name the array as a
    whole, so everything after the underscore is dropped.
    """
    ids = []
    for job in jobs:
        raw = job if isinstance(job, (str, int)) else job.job_id
        ids.append(str(raw).split("_")[0])
    return list(dict.fromkeys(ids))


def _stdout_path(job) -> Optional[str]:
    """Where submitit will write this job's stdout, if it will say."""
    try:
        return str(job.paths.stdout)
    except AttributeError:
        return None


def _describe_call(fn: Callable, args: Sequence, kwargs: Dict) -> str:
    shown = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    rendered = ", ".join(shown)
    if len(rendered) > 120:
        rendered = rendered[:117] + "..."
    return f"{getattr(fn, '__name__', fn)}({rendered})"


class _ConfiguredCall:
    """Picklable wrapper that sets the package's logging up inside the job.

    The worker is a fresh interpreter on a compute node, so nothing the driver
    script configured applies there. Without this every ``BaseProcessor`` message
    from the submitted function would land at the default level with the default
    format, and the useful ones would be invisible.
    """

    def __init__(self, fn: Callable, level: int):
        self.fn = fn
        self.level = level

    def __call__(self, *args, **kwargs):
        BaseProcessor.configure_logging(level=self.level)
        BaseProcessor.log(
            f"Running {getattr(self.fn, '__name__', self.fn)} on {socket.gethostname()} "
            f"(job {os.environ.get('SLURM_JOB_ID', '?')})"
        )
        return self.fn(*args, **kwargs)


class SlurmRunner(BaseProcessor):
    """
    Submit Python callables to SLURM with a fixed resource profile.

    One runner describes one shape of job. Ask for a different shape with
    :meth:`with_resources`, which returns a new runner and leaves this one
    alone, so a single runner can be shared safely.

    Parameters
    ----------
    log_dir : str | Path
        Where submitit keeps its pickles and each job's stdout/stderr. Must be
        visible from the compute nodes -- a shared filesystem, not node-local
        scratch. ``%j`` in the filenames is expanded by SLURM.
    cpus : int
        ``--cpus-per-task``. Also becomes ``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS``
        and ``OMP_NUM_THREADS`` inside the job.
    mem : str | float
        ``--mem``, as ``"16G"``, ``"512M"`` or a number of gibibytes.
    time : str | int
        Wall-clock limit: ``"04:00:00"``, ``"1-12:00:00"``, ``"4h"``, ``"90m"``,
        or minutes. Jobs are killed at this limit, so overestimate.
    partition, account, qos : str, optional
        Passed through to sbatch when set.
    gpus : int
        ``--gpus-per-node``; omitted entirely when 0.
    name : str, optional
        Job name. Defaults to the submitted function's ``__name__``.
    setup : list[str], optional
        Shell lines run before the worker starts. Defaults to
        :data:`DEFAULT_SETUP` (the thread pinning). Pass your own list to replace
        it -- e.g. to add ``module load`` lines -- and include the exports if you
        still want them.
    log_level : int
        Logging level configured inside the job.
    **slurm_extra
        Any other sbatch flag, as ``--flag=value`` pairs without the dashes:
        ``constraint="cascadelake"``, ``exclude="hpc11"``, ``mail_type="FAIL"``.
    """

    def __init__(
            self,
            log_dir: Union[str, Path] = "slurm_logs",
            cpus: int = 8,
            mem: Union[str, float] = "16G",
            time: Union[str, int] = "04:00:00",
            partition: Optional[str] = None,
            account: Optional[str] = None,
            qos: Optional[str] = None,
            gpus: int = 0,
            nodes: int = 1,
            name: Optional[str] = None,
            setup: Optional[Sequence[str]] = None,
            log_level: int = logging.INFO,
            dependency: Optional[str] = None,
            **slurm_extra,
    ):
        self.log_dir = Path(log_dir)
        self.cpus = cpus
        self.mem = mem
        self.time = time
        self.partition = partition
        self.account = account
        self.qos = qos
        self.gpus = gpus
        self.nodes = nodes
        self.name = name
        self.setup = list(DEFAULT_SETUP if setup is None else setup)
        self.log_level = log_level
        self.dependency = dependency
        self.slurm_extra = slurm_extra

    # ----------------------------------------------------------------- #
    # Building variants
    # ----------------------------------------------------------------- #

    def with_resources(self, **overrides) -> "SlurmRunner":
        """A copy of this runner with some settings changed.

        ``runner.with_resources(mem="64G", time="12:00:00")`` for the one step
        that needs more, without disturbing the shared runner.
        """
        settings = dict(
            log_dir=self.log_dir, cpus=self.cpus, mem=self.mem, time=self.time,
            partition=self.partition, account=self.account, qos=self.qos,
            gpus=self.gpus, nodes=self.nodes, name=self.name, setup=self.setup,
            log_level=self.log_level, dependency=self.dependency,
        )
        extra = dict(self.slurm_extra)
        for key, value in overrides.items():
            if key in settings:
                settings[key] = value
            else:
                extra[key] = value
        return SlurmRunner(**settings, **extra)

    def after(self, *jobs, state: str = "afterok") -> "SlurmRunner":
        """A copy of this runner whose jobs wait for ``jobs`` to finish.

        ``state`` is the sbatch dependency type: ``afterok`` (default) starts
        only if every listed job *succeeded*, ``afterany`` starts regardless of
        outcome, ``afternotok`` only on failure -- useful for a cleanup step.

        Note that a job waiting on a dependency that can never be satisfied sits
        in the queue forever with reason ``DependencyNeverSatisfied``; SLURM does
        not cancel it for you.
        """
        ids = _job_ids(jobs)
        if not ids:
            return self.with_resources(dependency=None)
        return self.with_resources(dependency=f"{state}:" + ":".join(ids))

    # ----------------------------------------------------------------- #
    # Submitting
    # ----------------------------------------------------------------- #

    def _executor(self, default_name: Optional[str] = None):
        submitit = _require_submitit()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        executor = submitit.AutoExecutor(folder=str(self.log_dir))
        parameters: Dict[str, Any] = {
            "name": self.name or default_name or "neurofunctionx",
            "timeout_min": parse_minutes(self.time),
            "cpus_per_task": self.cpus,
            "mem_gb": parse_memory_gb(self.mem),
            "nodes": self.nodes,
            "tasks_per_node": 1,
            "slurm_setup": self.setup,
        }
        if self.partition:
            parameters["slurm_partition"] = self.partition
        if self.account:
            parameters["slurm_account"] = self.account
        if self.gpus:
            parameters["gpus_per_node"] = self.gpus

        additional = dict(self.slurm_extra)
        if self.qos:
            additional["qos"] = self.qos
        if self.dependency:
            additional["dependency"] = self.dependency
        if additional:
            parameters["slurm_additional_parameters"] = additional

        executor.update_parameters(**parameters)
        return executor

    def _resources_text(self) -> str:
        bits = [f"{self.cpus} cpus", f"{self.mem}", f"{self.time}"]
        if self.gpus:
            bits.append(f"{self.gpus} gpu(s)")
        if self.partition:
            bits.append(self.partition)
        if self.dependency:
            bits.append(self.dependency)
        return ", ".join(str(b) for b in bits)

    def submit(self, fn: Callable, *args, **kwargs):
        """
        Submit one call and return its job straight away, without blocking.

        ``fn``, ``args`` and ``kwargs`` are pickled, so they must survive the
        trip to a compute node: file paths travel fine, open file handles and
        database connections do not. Call ``job.result()`` when you want the
        return value.
        """
        executor = self._executor(default_name=getattr(fn, "__name__", None))
        job = executor.submit(_ConfiguredCall(fn, self.log_level), *args, **kwargs)
        self._log(
            f"Submitted job {job.job_id}: {_describe_call(fn, args, kwargs)} "
            f"[{self._resources_text()}]"
        )
        self._log(f"Logs: {_stdout_path(job) or self.log_dir}")
        return job

    def map(self, fn: Callable, *iterables) -> List[Any]:
        """
        Submit one job array running ``fn`` once per element.

        Mirrors the builtin ``map``: with several iterables the function is
        called with one item from each, and the shortest one wins::

            jobs = runner.map(register_one, subject_ids)
            jobs = runner.map(register_pair, fixed_paths, moving_paths)

        One array is far kinder to the scheduler than N separate submissions,
        and every element gets the same resource request -- size it for the
        biggest one.
        """
        columns = [list(iterable) for iterable in iterables]
        if not columns:
            raise ValueError("map needs at least one iterable of arguments")
        length = min(len(column) for column in columns)
        columns = [column[:length] for column in columns]
        if length == 0:
            self._log_warn("map called with no arguments; nothing submitted")
            return []

        executor = self._executor(default_name=getattr(fn, "__name__", None))
        jobs = executor.map_array(_ConfiguredCall(fn, self.log_level), *columns)
        self._log(
            f"Submitted array of {len(jobs)} jobs "
            f"({getattr(fn, '__name__', fn)}, ids {jobs[0].job_id}..{jobs[-1].job_id}) "
            f"[{self._resources_text()}]"
        )
        return jobs

    # ----------------------------------------------------------------- #
    # Collecting
    # ----------------------------------------------------------------- #

    def results(self, jobs: Sequence[Any], ignore_errors: bool = False) -> List[Any]:
        """
        Wait for every job and return their results in order.

        With ``ignore_errors=True`` a failed job contributes ``None`` and its
        error is logged instead of raised, so one bad subject does not throw
        away the rest of a sweep.
        """
        results = []
        for index, job in enumerate(jobs, start=1):
            try:
                results.append(job.result())
                self._log(f"Job {job.job_id} finished ({index}/{len(jobs)})")
            except Exception as exc:
                if not ignore_errors:
                    raise
                self._log_error(f"Job {job.job_id} failed: {exc}")
                results.append(None)
        return results

    @staticmethod
    def cancel(jobs: Union[Any, Sequence[Any]]) -> None:
        """Cancel one job or a sequence of them."""
        for job in (jobs if isinstance(jobs, (list, tuple)) else [jobs]):
            job.cancel()
            BaseProcessor.log(f"Cancelled job {job.job_id}")
