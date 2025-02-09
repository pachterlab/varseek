"""varseek package initialization module."""
import logging

from .utils import *  # noqa: F401, F403  # only imports what is in __all__ in .utils/__init__.py
from .varseek_build import build  # noqa: F401
from .varseek_clean import clean  # noqa: F401
from .varseek_count import count  # noqa: F401
from .varseek_fastqpp import fastqpp  # noqa: F401
from .varseek_filter import filter  # noqa: F401
from .varseek_info import info  # noqa: F401
from .varseek_ref import ref  # noqa: F401
from .varseek_sim import sim  # noqa: F401
from .varseek_summarize import summarize  # noqa: F401

# Mute numexpr threads info
logging.getLogger("numexpr").setLevel(logging.WARNING)

__version__ = "0.1.0"
__author__ = "Joseph Rich"
__email__ = "josephrich98@gmail.com"
