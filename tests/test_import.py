import pathways
from pathways import __version__


def test_import():
    assert pathways.__version__ == __version__
