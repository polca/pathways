from pathlib import Path

__version__ = (0, 0, 1)
__all__ = ("__version__", "DATA_DIR", "Pathways")

DATA_DIR = Path(__file__).resolve().parent / "data"

from .pathways import Pathways
