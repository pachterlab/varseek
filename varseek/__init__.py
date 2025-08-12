"""varseek package initialization module."""

import logging

from .utils import *  # only imports what is in __all__ in .utils/__init__.py
from .varseek_build import build
from .varseek_clean import clean
from .varseek_count import count
from .varseek_fastqpp import fastqpp
from .varseek_filter import filter
from .varseek_info import info
from .varseek_ref import ref
from .varseek_sim import sim
from .varseek_summarize import summarize

# # a possible alternative to avoid running imports from each function - but I already run all imports from utils functions, so it doesn't really matter
# def build(*args, **kwargs):
#     return importlib.import_module("varseek.varseek_build").build(*args, **kwargs)


# Mute numexpr threads info
logging.getLogger("numexpr").setLevel(logging.WARNING)

__version__ = "0.1.1"
__author__ = "Joseph Rich"
__email__ = "josephrich98@gmail.com"
