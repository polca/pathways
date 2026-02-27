__version__ = (1, 0, 4)
__all__ = ("__version__", "Pathways", "run_gsa", "configure_logging")


def __getattr__(name):
    if name == "Pathways":
        from .pathways import Pathways

        return Pathways
    if name == "run_gsa":
        from .stats import run_gsa

        return run_gsa
    if name == "configure_logging":
        from .logging_steup import configure_logging

        return configure_logging
    raise AttributeError(f"module 'pathways' has no attribute {name!r}")
