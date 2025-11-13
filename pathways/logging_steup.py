# pathways/logging_setup.py
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

from .filesystem_constants import USER_LOGS_DIR

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _ensure_dir(p: Path) -> None:
    """Create a directory (and parents) if it does not yet exist.

    :param p: Directory path to create.
    :type p: pathlib.Path
    :returns: ``None``
    :rtype: None
    """

    p.mkdir(parents=True, exist_ok=True)


def configure_logging(
    *,
    mode: str = "per-run",  # "per-run" | "rotating"
    level: int = logging.INFO,
    console: bool = True,
    max_bytes: int = 5_000_000,  # only used in "rotating" mode
    backup_count: int = 5,  # only used in "rotating" mode
    run_tag: str | None = None,  # appended to per-run filenames
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> Path:
    """Configure application logging and return the active log file path.

    :param mode: ``"per-run"`` for timestamped files or ``"rotating"`` for a rolling handler.
    :type mode: str
    :param level: Root logger level to apply.
    :type level: int
    :param console: Whether to emit log messages to ``stderr`` in addition to the file.
    :type console: bool
    :param max_bytes: Maximum file size for rotating mode before rollover.
    :type max_bytes: int
    :param backup_count: Number of rotated files to retain.
    :type backup_count: int
    :param run_tag: Optional suffix appended to per-run filenames.
    :type run_tag: str | None
    :param fmt: Logging format string to use.
    :type fmt: str
    :param datefmt: Date formatting string for log records.
    :type datefmt: str
    :raises ValueError: If an unsupported ``mode`` is given.
    :returns: Path to the log file that is configured.
    :rtype: pathlib.Path
    """
    _ensure_dir(USER_LOGS_DIR)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers if called twice
    _remove_pathways_handlers(root)

    # File handler
    if mode == "per-run":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_tag}" if run_tag else ""
        log_file = USER_LOGS_DIR / f"pathways_{ts}{suffix}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
    elif mode == "rotating":
        log_file = USER_LOGS_DIR / "pathways.log"
        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
    else:
        raise ValueError("mode must be 'per-run' or 'rotating'")

    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    fh.addFilter(_PathwaysOnceFilter())
    root.addHandler(fh)

    # Optional console
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        ch.addFilter(_PathwaysOnceFilter())
        root.addHandler(ch)

    logging.getLogger(__name__).info("Logging initialized. File: %s", log_file)
    return log_file


class _PathwaysOnceFilter(logging.Filter):
    """Filter tagging handlers installed by Pathways so they can be removed safely."""

    pass


def _remove_pathways_handlers(root: logging.Logger) -> None:
    """Detach and close any handlers previously installed by :func:`configure_logging`.

    :param root: Logger whose handlers should be inspected.
    :type root: logging.Logger
    :returns: ``None``
    :rtype: None
    """

    to_remove = [
        h
        for h in root.handlers
        if any(isinstance(f, _PathwaysOnceFilter) for f in h.filters)
    ]
    for h in to_remove:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
