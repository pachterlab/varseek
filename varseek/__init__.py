from .varseek_build import build
from .varseek_summarize import summarize
from .varseek_clean import clean
from .varseek_info import info
from .varseek_sim import sim
from .varseek_filter import filter
from .varseek_fastqpp import fastqpp
from .varseek_ref import ref
from .varseek_count import count

from .utils import *  # only imports what is in __all__ in .utils/__init__.py

import logging

# Mute numexpr threads info
logging.getLogger("numexpr").setLevel(logging.WARNING)

__version__ = "0.1.0"
__author__ = "Joseph Rich"
__email__ = "josephrich98@gmail.com"
