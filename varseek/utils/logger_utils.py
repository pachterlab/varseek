import os
import logging
from datetime import datetime
import time
from IPython.core.magic import register_cell_magic

# Mute numexpr threads info
logging.getLogger("numexpr").setLevel(logging.WARNING)


def set_up_logger(logging_level_name=None, save_logs=False, log_dir=None):
    if logging_level_name is None:
        logging_level_name = os.getenv("VARSEEK_LOGLEVEL", "INFO")
    logging_level = logging.getLevelName(
        logging_level_name
    )  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    if type(logging_level) != int:  # unknown log level
        logging_level = logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)

    if not logger.hasHandlers():
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if save_logs:
            if log_dir is None:
                package_dir = os.path.dirname(os.path.abspath(__file__))
                log_dir = os.path.join(package_dir, "logs")

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_file = os.path.join(
                log_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# * DEPRECATED - use %%time instead
def report_time(running_total=None):
    if running_total is None:
        running_total = time.time()
    elapsed_time = time.time() - running_total
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"RUNTIME: {minutes}m, {seconds:.2f}s")
    running_total = time.time()
    return running_total


try:
    from IPython import get_ipython

    ip = get_ipython()
except ImportError:
    ip = None

if ip:

    @register_cell_magic
    def cell_runtime(
        line, cell
    ):  # best version - slight overhead (~0.15s per bash command in a cell), but works on multiline bash commands with variables
        start_time = time.time()
        get_ipython().run_cell(cell)  # type: ignore
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"RUNTIME: {minutes}m, {seconds:.2f}s")

    def load_ipython_extension(ipython):
        ipython.register_magic_function(cell_runtime, "cell")

else:

    def cell_runtime(*args):
        pass

    def load_ipython_extension(ipython):
        pass


# # unused code for utilizing rich - rprint will work like rich.print if rich is True else it will work like print; logger will have rich output if rich is True else normal
# # to use these functions, I must write these 4 lines in each module
# use_rich = True  # or environment variable USE_RICH_VARSEEK = True
# rprint = define_rprint(use_rich)  # for rprint
# logger = set_up_logger(logging_level_name = None, save_logs = False, rich=use_rich)  # for logger
# add_color_to_logger(logger, rich=use_rich)  # if I want to use color in the logger message as well - can uncomment if use_rich is False AND all of my log statements do not have a color argument

# def define_rprint(use_rich = None):
#     if use_rich is None:
#         use_rich = os.getenv("USE_RICH_VARSEEK", "false").lower() == "true"
#     if use_rich:
#         try:
#             from rich import print as rich_print
#         except ImportError:
#             def rprint(message, color=None, bold=False):
#                 print(message)
#             return rprint
#         def rprint(message, color=None, bold=False):
#             if color:
#                 if bold:
#                     rich_print(f"[bold {color}]{message}[/bold {color}]")
#                 else:
#                     rich_print(f"[{color}]{message}[/{color}]")
#             elif bold:
#                 rich_print(f"[bold]{message}[/bold]")
#             else:
#                 rich_print(message)
#     else:
#         def rprint(message, *args, **kwargs):
#             print(message)
#     return rprint


# # Mute numexpr threads info
# logging.getLogger("numexpr").setLevel(logging.WARNING)

# def set_up_logger(logging_level_name=None, save_logs=False, log_dir=None, rich=None):
#     if rich is None:
#         rich = os.getenv("USE_RICH_VARSEEK", "false").lower() == "true"
#     if rich:
#         try:
#             from rich.logging import RichHandler  # Import RichHandler
#         except ImportError:
#             rich = False
#     if logging_level_name is None:
#         logging_level_name = os.getenv("VARSEEK_LOGLEVEL", "INFO")
#     logging_level = logging.getLevelName(logging_level_name)  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
#     if type(logging_level) != int:  # unknown log level
#         logging_level = logging.INFO

#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging_level)

#     if not logger.hasHandlers():
#         formatter = logging.Formatter(
#             "%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
#         )

#         # Add RichHandler if rich=True
#         if rich:
#             console_handler = RichHandler(markup=True, rich_tracebacks=True)
#         else:
#             console_handler = logging.StreamHandler()
#             console_handler.setFormatter(formatter)

#         logger.addHandler(console_handler)

#         if save_logs:
#             if log_dir is None:
#                 package_dir = os.path.dirname(os.path.abspath(__file__))
#                 log_dir = os.path.join(package_dir, 'logs')

#             if not os.path.exists(log_dir):
#                 os.makedirs(log_dir)

#             log_file = os.path.join(
#                 log_dir, f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
#             )

#             file_handler = logging.FileHandler(log_file)
#             file_handler.setFormatter(formatter)
#             logger.addHandler(file_handler)

#     return logger

# def add_color_to_logger(logger, rich = False):
#     """
#     Wraps logger methods to support 'color' argument for rich markup.

#     Args:
#         logger: The logger object to wrap.
#     """
#     def create_wrapper(log_func):
#         def wrapper(message, *args, color=None, **kwargs):
#             if color and rich:
#                 message = f"[{color}]{message}[/{color}]"
#             log_func(message, *args, **kwargs)
#         return wrapper

#     for level in ['debug', 'info', 'warning', 'error', 'critical']:
#         log_func = getattr(logger, level)
#         setattr(logger, level, create_wrapper(log_func))
