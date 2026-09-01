"""B-ACE 2.0 Python package."""

from bace.env import BaceEnv, BaceGymEnv, BaceVecEnv, BaceVecGym, make_env

__version__ = "2.0.0"
__all__ = [
    "BaceEnv",
    "BaceGymEnv",
    "BaceVecEnv",
    "BaceVecGym",
    "make_env",
    "run_experiment",
    "__version__",
]


def __getattr__(name: str):
    if name == "run_experiment":
        from bace.experiment import run_experiment as _run

        return _run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
