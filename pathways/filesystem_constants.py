"""
This module contains constants for the filesystem paths used by Pathways.
"""

from pathlib import Path

import platformdirs
import yaml


def load_var_file():
    """Load optional path overrides from ``variables.yaml`` in the working directory.

    :returns: Parsed YAML contents if present, otherwise ``None``.
    :rtype: dict | None
    """
    var_file = Path.cwd() / "variables.yaml"
    if var_file.exists():
        with open(var_file, "r") as f:
            return yaml.safe_load(f)
    else:
        return None


VARIABLES = load_var_file() or {}

# Directories for data which comes with Pathways
if "DATA_DIR" in VARIABLES:
    DATA_DIR = Path(VARIABLES["DATA_DIR"])
else:
    DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Directories for user-created data
if "USER_DATA_BASE_DIR" in VARIABLES:
    USER_DATA_BASE_DIR = Path(VARIABLES.get("USER_DATA_BASE_DIR"))
else:
    USER_DATA_BASE_DIR = platformdirs.user_data_path(
        appname="pathways", appauthor="pylca"
    )
USER_DATA_BASE_DIR.mkdir(parents=True, exist_ok=True)

if "DIR_CACHED_DB" in VARIABLES:
    DIR_CACHED_DB = Path(VARIABLES.get("DIR_CACHED_DB"))
else:
    DIR_CACHED_DB = USER_DATA_BASE_DIR / "cache"
DIR_CACHED_DB.mkdir(parents=True, exist_ok=True)

if "USER_LOGS_DIR" in VARIABLES:
    USER_LOGS_DIR = Path(VARIABLES["USER_LOGS_DIR"])
else:
    USER_LOGS_DIR = platformdirs.user_log_path(appname="pathways", appauthor="pylca")
USER_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# STATS_DIR = USER_DATA_BASE_DIR / "stats"
if "STATS_DIR" in VARIABLES:
    STATS_DIR = Path(VARIABLES["STATS_DIR"])
else:
    STATS_DIR = Path.cwd() / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)
