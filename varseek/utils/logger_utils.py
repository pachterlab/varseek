"""varseek logger utilities."""

import base64
import getpass
import inspect
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
from gget.gget_cosmic import is_valid_email

from varseek.constants import default_filename_dict

# Mute numexpr threads info
logging.getLogger("numexpr").setLevel(logging.WARNING)


def set_up_logger(logger, logging_level=None, save_logs=False, log_dir=None):
    # type checking
    if not isinstance(save_logs, bool):
        raise TypeError(f"save_logs must be a boolean, got {type(save_logs)}")
    if log_dir is not None and not isinstance(log_dir, (str, Path)):
        raise TypeError(f"log_dir must be a string or Path or None, got {type(log_dir)}")
    if log_dir is not None:
        save_logs = True  # if someone provides a log_dir, they want to save logs

    # retrieve logging_level and check value
    if logging_level is None:
        logging_level = os.getenv("VARSEEK_LOGGING_LEVEL", "INFO")
    if str(logging_level) not in {"NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60"}:  # unknown log level
        print(f"Unknown log level: {logging_level}. Defaulting to INFO.")
        logging_level = logging.INFO

    if logging_level in {"0", "10", "20", "30", "40", "50", "60"}:
        logging_level = int(logging_level)

    # logger = logging.getLogger(__name__)  # leave commented out and run in each module individually
    logger.setLevel(logging_level)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        # console_handler.setLevel(logging_level)  # redundant
        logger.addHandler(console_handler)

        if save_logs:
            if log_dir is None:
                package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                log_dir = os.path.join(package_dir, "logs")

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            function_name = inspect.stack()[1].function  # gets the name of the function that called it (eg build, info, filter, etc)
            dt = datetime.now()
            log_file = os.path.join(log_dir, f"logs_{function_name}_date_{dt:%Y_%m_%d}_time_{dt:%H_%M_%S}.txt")

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            # file_handler.setLevel(logging.DEBUG)  # Capture all logs regardless of logger's level
            logger.addHandler(file_handler)

    return logger


def check_file_path_is_string_with_valid_extension(file_path, variable_name, file_type, required=False):
    valid_extensions = {
        "json": {".json"},
        "yml": {".yaml", ".yml"},
        "yaml": {".yaml", ".yml"},
        "csv": {".csv"},
        "tsv": {".tsv"},
        "txt": {".txt"},
        "fasta": {".fasta", ".fa", ".fna", ".ffn"},
        "fastq": {".fastq", ".fq"},
        "bam": {".bam"},
        "bed": {".bed"},
        "vcf": {".vcf"},
        "gtf": {".gtf"},
        "t2g": {".txt"},
        "index": {".idx"},
        "h5ad": {".h5ad", ".h5"},
        "adata": {".h5ad", ".h5"},
    }
    if file_path is not None:  # skip if None or empty string, as I will provide the default path in this case
        # check if file_path is a dataframe or AnnData with the correct extension
        if isinstance(file_path, pd.DataFrame) and file_type in {"csv", "tsv"}:
            pass
        elif isinstance(file_path, ad.AnnData) and file_type in {"h5ad"}:
            pass
        else:
            # check if file_path is a string
            if not isinstance(file_path, (str, Path)):
                raise ValueError(f"{variable_name} must be a string or Path, got {type(file_path)}")

            # check if file_type is a single value or list of values
            if isinstance(file_type, str):
                valid_extensions_for_file_type = valid_extensions.get(file_type)
            elif isinstance(file_type, (list, set, tuple)):
                valid_extensions_for_file_type = set()
                for ft in file_type:
                    valid_extensions_for_file_type.update(valid_extensions.get(ft))
            else:
                raise ValueError(f"file_type must be a string or a list, got {type(file_type)}")

            # check if file has valid extension
            file_path = str(file_path)
            if not any(file_path.lower().endswith((ext, f"{ext}.zip", f"{ext}.gz")) for ext in valid_extensions_for_file_type):
                raise ValueError(f"Invalid file extension for {variable_name}. Must be one of {valid_extensions_for_file_type}")
    else:
        if required:
            raise ValueError(f"{file_type} file path is required")


def make_function_parameter_to_value_dict(levels_up=1, explicit_only=False):
    # Collect parameters in a dictionary
    params = OrderedDict()

    # Get the caller's frame (one level up in the stack)
    frame = inspect.currentframe()

    for _ in range(levels_up):
        if frame is None:
            break
        frame = frame.f_back

    function_args, varargs, varkw, values = inspect.getargvalues(frame)

    # handle explicit function arguments
    for arg in function_args:
        params[arg] = values[arg]

    if not explicit_only:
        # handle *args
        if varargs:
            params[varargs] = values[varargs]

        # handle **kwargs
        if varkw:
            for key, value in values[varkw].items():
                params[key] = value

    return params


def report_time_elapsed(start_time, logger=None, function_name=None):
    elapsed = time.perf_counter() - start_time
    function_name_message = f" for vk {function_name}" if function_name else ""
    time_elapsed_message = f"Total runtime{function_name_message}: {int(elapsed // 60)}m, {elapsed % 60:.2f}s"
    if logger:
        logger.info(time_elapsed_message)
    else:
        print(time_elapsed_message)


def convert_value_for_json(value):
    if isinstance(value, Path):
        return str(value)
    elif isinstance(value, pd.DataFrame):
        return "DataFrame Object"
    elif isinstance(value, pd.Series):
        return "Series Object"
    elif isinstance(value, ad.AnnData):
        return "AnnData Object"
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32)):
        return float(value)
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, (set, frozenset)):
        return list(value)
    elif isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    elif isinstance(value, logging.Logger):
        return f"Logger Object with name {value.name}"
    # Add more conversions as needed
    else:
        return value


def save_params_to_config_file(params=None, out_file="run_config.json", remove_passwords=True):
    out_file_directory = os.path.dirname(out_file)
    if not out_file_directory:
        out_file_directory = "."
    else:
        os.makedirs(out_file_directory, exist_ok=True)

    # Collect parameters in a dictionary
    if not params:
        params = make_function_parameter_to_value_dict(levels_up=2)

    if remove_passwords:
        for key in params.keys():
            if "password" in key.lower():
                params[key] = "********"

    params = {key: convert_value_for_json(value) for key, value in params.items()}  # avoid json serialization errors

    # Write to JSON
    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(params, file, indent=4)


def return_kb_arguments(command, remove_dashes=False):
    # Run the help command and capture the output
    result = subprocess.run(["kb", command, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

    help_output = result.stdout

    # # Regex pattern to match options (e.g., -x, --long-option)
    # options_pattern = r"(--?\w[\w-]*)"

    # # Find all matches in the help output
    # arguments = re.findall(options_pattern, help_output)

    line_pattern = r"\n  (--?.*)"

    # Categorize flags based on whether they have an argument
    store_true_flags = []
    flags_with_args = []

    # Find all matching lines
    matching_lines = re.findall(line_pattern, help_output)

    # Determine the last character of the contiguous string for each match
    for line in matching_lines:
        # Split the line by the first space to isolate the flag
        flag = line.split()[0]
        if flag[-1] == ",":
            len_first_flag = len(flag) + 1  # accounts for length of flag, comma, and space
            flag = line.split()[1]  # only the long flag is valid
        else:
            len_first_flag = 0
        # Get the last character of the contiguous string
        if flag == "--help":
            continue
        last_char_pos = len_first_flag + len(flag) - 1
        positional_arg_pos = last_char_pos + 2
        if (positional_arg_pos >= len(line)) or (line[positional_arg_pos] == " "):
            store_true_flags.append(flag)
        else:
            flags_with_args.append(flag)

    kb_arguments = {}
    kb_arguments["store_true_flags"] = store_true_flags
    kb_arguments["flags_with_args"] = flags_with_args

    # remove dashes
    if remove_dashes:
        kb_arguments["store_true_flags"] = [argument.lstrip("-").replace("-", "_") for argument in kb_arguments["store_true_flags"]]

        kb_arguments["flags_with_args"] = [argument.lstrip("-").replace("-", "_") for argument in kb_arguments["flags_with_args"]]

    return kb_arguments


def print_varseek_dry_run(params, function_name=None, remove_passwords=True):
    if function_name:
        if function_name not in {"build", "info", "filter", "fastqpp", "clean", "summarize", "ref", "count"}:
            raise ValueError(f"function_name must be one of build, info, filter, fastqpp, clean, summarize, ref, count. Got {function_name}")
        print(f"varseek.varseek_{function_name}.{function_name}(", end="\n  ")
    len_params = len(params)
    for i, (param_key, param_value) in enumerate(params.items()):
        if isinstance(param_value, str):
            param_value = f'"{param_value}"'

        if "password" in param_key.lower() and remove_passwords:
            param_value = "******"

        end = ",\n  " if i < len_params - 1 else "\n"  # Normal newline for last entry
        print(f"{param_key} = {param_value}", end=end)
    if function_name:
        print(")")


def assign_output_file_name_for_download_varseek_files(response, out, filetype):
    output_file = os.path.join(out, default_filename_dict[filetype])
    # content_disposition = response.headers.get("Content-Disposition", "")
    # filename = (
    #     content_disposition.split("filename=")[-1].strip('"')
    #     if "filename=" in content_disposition
    #     else "unknown"
    # )
    # if filename:
    #     filename = filename.split('";')[0]
    #     output_file = os.path.join(out, filename)
    # else:
    #     output_file = os.path.join(out, default_filename_dict[filetype])
    return output_file


def download_varseek_files(urls_dict, out=".", verbose=True):
    filetype_to_filename_dict = {}
    for filetype, url in urls_dict.items():
        os.makedirs(out, exist_ok=True)

        response = requests.get(url, stream=True, timeout=(10, 90))

        # Check for successful response
        if response.status_code == 200:
            # Extract the filename from the Content-Disposition header
            output_file_path = assign_output_file_name_for_download_varseek_files(response=response, out=out, filetype=filetype)

            with open(output_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            filetype_to_filename_dict[filetype] = output_file_path
            if verbose:  # setting verbose so that I can turn it off in vk ref
                print(f"File downloaded successfully as '{output_file_path}'")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")    # no need to toggle with verbose - I only do above for vk ref to avoid confusion (will say it downloaded to the original file path, but vk ref moves it)

    return filetype_to_filename_dict


def is_program_installed(program):
    return shutil.which(program) is not None or os.path.exists(program)


# # if I want to remove shell below:
# command = ["/usr/bin/time", time_flag, "python3", script_path]
# if argparse_flags:
# command.extend(shlex.split(argparse_flags))
# try:
# result = subprocess.run(command, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)


def report_time_and_memory_of_script(script_path, argparse_flags=None, output_file=None):
    # Run the command and capture stderr, where `/usr/bin/time -l` outputs its results
    system = os.uname().sysname
    time_flag = "-v" if system == "Linux" else "-l"
    command = f"/usr/bin/time {time_flag} python3 {script_path}"

    if argparse_flags:
        command += f" {argparse_flags}"

    try:
        result = subprocess.run(command, shell=True, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
    except Exception as e:
        print(f"Error running command {command}: {e}")
        return None

    if system == "Linux":
        time_re = r"(?:Elapsed \(wall clock\) time \(.*\):\s+)?(\d+:)?(\d+):(\d+(?:\.\d+)?)"
        time_group = 1
        memory_re = r"Maximum resident set size \(kbytes\): (\d+)"
    else:
        time_re = r"(\d+\.\d+)\s+(?:real|user|sys)"
        time_group = 1
        memory_re = r"\s+(\d+)\s+maximum resident set size"

    match_time = re.search(time_re, result.stderr)
    if match_time:
        if system == "Linux":
            hours = float(match_time.group(1)) if match_time.group(1) else 0
            minutes = int(float(match_time.group(2)) + hours * 60)
            seconds = float(match_time.group(3))
            runtime = minutes * 60 + seconds
        else:
            runtime = float(match_time.group(time_group))  # Return real time in seconds
            minutes = int(runtime // 60)
            seconds = runtime % 60
        time_message = f"Runtime: {minutes} minutes, {seconds:.2f} seconds"
        print(time_message)

    # Extract the "maximum resident set size" line using a regex
    match_memory = re.search(memory_re, result.stderr)
    if match_memory:
        peak_memory = int(match_memory.group(1))  # Capture the numeric value
        # Determine units (bytes or KB)
        if "kbytes" in match_memory.group(0):
            peak_memory *= 1024  # Convert KB to bytes
        peak_memory_readable_units = peak_memory / (1024**2)  # MB
        unit = "MB"
        if peak_memory_readable_units > 1000:
            peak_memory_readable_units = peak_memory_readable_units / 1024  # GB
            unit = "GB"
        memory_message = f"Peak memory usage: {peak_memory_readable_units:.2f} {unit}"
        print(memory_message)

    if not match_time:
        raise ValueError("Failed to find 'real' time in output.")
    if not match_memory:
        raise ValueError("Failed to find 'maximum resident set size' in output.")

    if output_file:
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(time_message + "\n")
            f.write(memory_message + "\n")

    return (runtime, peak_memory)  # Return the runtime and peak memory usage in bytes


def make_positional_arguments_list_and_keyword_arguments_dict():
    args = sys.argv[1:]

    # Initialize storage
    args_dict = {}
    positional_args = []

    # Parse arguments
    i = 0
    while i < len(args):
        if args[i].startswith("--"):  # Handle flags
            key = args[i].lstrip("--")
            # Check if a value exists and doesn't start with "--"
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                args_dict[key] = args[i + 1]
                i += 1  # Skip the value
            else:
                args_dict[key] = True  # Store True for flags without values
        else:  # Handle positional arguments
            positional_args.append(args[i])
        i += 1

    return positional_args, args_dict


def run_command_with_error_logging(command, verbose=True, track_time=False):
    if track_time:
        start_time = time.time()
    if isinstance(command, str):
        shell = True
    elif isinstance(command, list):
        shell = False
    else:
        raise ValueError("Command must be a string or a list.")

    command_string = command if shell else " ".join(command)

    try:
        if verbose:
            print(f"Running command: {command_string}")
        subprocess.run(command, check=True, shell=shell)
    except subprocess.CalledProcessError as e:
        # Log the error for failed commands
        print(f"Command failed with exit code {e.returncode}")
        print(f"Command: {command_string}")
    except FileNotFoundError:
        print("Error: Command not found. Ensure the command or executable exists.")
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")

    if track_time:
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        if verbose:
            print(f"Command runtime: {minutes}m, {seconds:.2f}s")
        return minutes, seconds


def download_box_url(url, output_folder=".", output_file_name=None, verbose=True):
    if not output_file_name:
        output_file_name = url.split("/")[-1]
    if "/" not in output_file_name:
        output_file_path = os.path.join(output_folder, output_file_name)
    else:
        output_file_path = output_file_name
        output_folder = os.path.dirname(output_file_name)
    os.makedirs(output_folder, exist_ok=True)

    # Download the file
    response = requests.get(url, stream=True, timeout=(10, 90))
    if response.status_code == 200:
        with open(output_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        if verbose:  
            print(f"File downloaded successfully to {output_file_path}")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")


# # * DEPRECATED - use %%time instead
# def report_time(running_total=None):
#     if running_total is None:
#         running_total = time.time()
#     elapsed_time = time.time() - running_total
#     minutes = int(elapsed_time // 60)
#     seconds = elapsed_time % 60
#     print(f"RUNTIME: {minutes}m, {seconds:.2f}s")
#     running_total = time.time()
#     return running_total


def get_set_of_parameters_from_function_signature(func):
    signature = inspect.signature(func)
    # Extract the parameter names, excluding **kwargs
    return {name for name, param in signature.parameters.items() if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}


def get_set_of_allowable_kwargs(func):
    """
    Extracts the argument names following the line '# Hidden arguments'.
    Requires a docstring line of the following format:
    - NON-CAPTURED_ARG1     (TYPE1 or TYPE2 or ...) DESCRIPTION
    OPTIONAL EXTRA DESCRIPTION LINE 1
    OPTIONAL EXTRA DESCRIPTION LINE 2
    ...
    - NON-CAPTURED_ARG2    (TYPE1 or TYPE2 or ...) DESCRIPTION
    ...
    # Hidden arguments
    - CAPTURED_ARG1    (TYPE1 or TYPE2 or ...) DESCRIPTION
    - CAPTURED_ARG2    (TYPE1 or TYPE2 or ...) DESCRIPTION
    ...
    """
    docstring = inspect.getdoc(func)

    # Initialize variables
    hidden_args_section_found = False
    hidden_args = set()

    # Loop through each line in the docstring
    for line in docstring.splitlines():
        # Check if we've reached the "# Hidden arguments" section
        if "# Hidden arguments" in line:
            hidden_args_section_found = True
            continue  # Skip the header line

        # If in the hidden arguments section, look for argument patterns
        if hidden_args_section_found:
            # Match lines starting with a dash followed by a valid argument name and type in parentheses
            match = re.match(r"-\s*([a-zA-Z_]\w*)\s*\(.*?\)", line)
            if match:
                # Extract and append the argument name
                hidden_args.add(match.group(1))

    return hidden_args


def is_valid_int(value, threshold_type=None, threshold_value=None, optional=False, min_value_inclusive=None, max_value_inclusive=None):
    """
    Check if value is an integer or a string representation of an integer.
    Optionally, apply a threshold comparison.

    Parameters:
    - value: The value to check.
    - threshold_value (int, optional): The threshold to compare against. This is the threshold for which the value **is** valid. (eg threshold_type='>=', threshold_value=1 means that >=1 returns True, and <1 returns False)
    - threshold_type (str, optional): Comparison type ('<', '<=', '>', '>=')
    - optional (bool, optional): If True, the value can be None.

    Returns:
    - True if value is a valid integer and meets the threshold condition (if specified).
    - False otherwise.
    """
    # Check for optional
    if optional and value is None:
        return True

    # Check if value is an int or a valid string representation of an int
    if not (isinstance(value, int) or (isinstance(value, str) and value.isdigit())):
        return False

    # Convert to integer
    value = int(value)

    # If no threshold is given, just return True
    if threshold_value is None:
        if min_value_inclusive and not max_value_inclusive:
            threshold_value = min_value_inclusive
        elif max_value_inclusive and not min_value_inclusive:
            threshold_value = max_value_inclusive
        elif min_value_inclusive and max_value_inclusive:
            threshold_value = min_value_inclusive
        else:
            return True

    # Apply threshold comparison
    if threshold_type == "<":
        return value < threshold_value
    elif threshold_type == "<=":
        return value <= threshold_value
    elif threshold_type == ">":
        return value > threshold_value
    elif threshold_type == ">=":
        return value >= threshold_value
    elif threshold_type == "between":
        return min_value_inclusive <= value <= max_value_inclusive
    elif threshold_type is None:  # No threshold comparison
        return True
    else:
        raise ValueError(f"Invalid threshold_type: {threshold_type}. Must be one of '<', '<=', '>', '>=', 'between'.")


try:
    from IPython import get_ipython
    from IPython.core.magic import register_cell_magic

    ip = get_ipython()
except ImportError:
    ip = None

if ip:

    @register_cell_magic
    def cell_runtime(line, cell):  # best version - slight overhead (~0.15s per bash command in a cell), but works on multiline bash commands with variables
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


def extract_documentation_file_blocks(file_path, start_pattern, stop_pattern):
    """
    Extract blocks of text from a file based on start and stop regex patterns.

    :param file_path: Path to the file.
    :param start_pattern: Regex pattern to identify the start of a block.
    :param stop_pattern: Regex pattern to identify the stop condition.
    :return: List of extracted text blocks.
    """
    extracted_blocks = []
    current_block = []
    capturing = False  # Flag to track if we are inside a block

    start_regex = re.compile(start_pattern)
    stop_regex = re.compile(stop_pattern)

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip()  # Remove trailing newlines but preserve content

            if capturing:
                if stop_regex.match(line):  # Stop capturing if the stop pattern matches
                    extracted_blocks.append("\n".join(current_block))
                    current_block = []
                    capturing = False
                else:
                    current_block.append(line)

            if start_regex.match(line):  # Start capturing if the start pattern matches
                capturing = True
                current_block.append(line)

    # Capture any remaining block if the file ends without a newline
    if current_block:
        extracted_blocks.append("\n".join(current_block))

    return extracted_blocks


# from gget cosmic
def authenticate_cosmic_credentials(email=None, password=None):
    if not email:
        email = input("Please enter your COSMIC email: ")
    if not is_valid_email(email):
        raise ValueError("The email address is not valid.")
    if not password:
        password = getpass.getpass("Please enter your COSMIC password: ")

    # Concatenate the email and password with a colon
    input_string = f"{email}:{password}\n"

    encoded_bytes = base64.b64encode(input_string.encode("utf-8"))
    encoded_string = encoded_bytes.decode("utf-8")
    curl_command = [
        "curl",
        "-H",
        f"Authorization: Basic {encoded_string}",
        "https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted?path=grch37/cmc/v101/CancerMutationCensus_AllData_Tsv_v101_GRCh37.tar&bucket=downloads",  # COSMIC CMC - doesn't really matter what it is
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True, check=True)

    try:
        response_data = json.loads(result.stdout)
        true_download_url = response_data.get("url")
        return True
    except AttributeError:
        print("Invalid username or password.")
        return False


def encode_cosmic_credentials(email=None, password=None):
    """Encodes COSMIC email and password into a base64 authentication token."""
    if not email:
        email = input("Please enter your COSMIC email: ")
    if not password:
        password = getpass.getpass("Please enter your COSMIC password: ")

    input_string = f"{email}:{password}\n"
    encoded_bytes = base64.b64encode(input_string.encode("utf-8"))
    return encoded_bytes.decode("utf-8")


def authenticate_cosmic_credentials_via_server(encoded_token):
    """Sends the encoded authentication token to the server for verification."""
    server_url = "https://your-secure-server.com/verify_cosmic"  #!!! modify - see varseek_server/validate_cosmic.py

    response = requests.post(server_url, json={"encoded_token": encoded_token}, timeout=20)

    if response.status_code == 200 and response.json().get("authenticated"):
        return True
    else:
        print("Invalid credentials.")
        return False


def get_python_function_call():
    # Get the calling frame
    frame = inspect.currentframe().f_back.f_back.f_back.f_back  # goes 4 up - 1 to get_python_function_call, 1 to get_python_or_cli_function_call, 1 to save_run_info, and 1 to the function of interest
    function_call = inspect.getsource(frame).strip()

    return function_call


def get_python_or_cli_function_call():
    len_sys_argv = len(sys.argv)
    if len_sys_argv == 1:  # Python script
        function_call = get_python_function_call()
    elif len_sys_argv == 2 and "ipykernel" in sys.argv[0]:  # Jupyter notebook python
        function_call = get_python_function_call()
    else:  # command line (terminal or Jupyter with '!')
        function_call = " ".join(sys.argv)
    return function_call


def save_run_info(out_file="run_info.txt", remove_passwords=True):
    from varseek import (
        __version__,  # keep internal to this function to avoid circular import
    )

    out_file_directory = os.path.dirname(out_file)
    if not out_file_directory:
        out_file_directory = "."
    else:
        os.makedirs(out_file_directory, exist_ok=True)

    function_call = get_python_or_cli_function_call()

    if remove_passwords:
        # for python calls
        python_password_pattern = re.compile(r'(\b\w*password\w*\s*=\s*)(["\'].*?["\'])', re.IGNORECASE)
        function_call = python_password_pattern.sub(r'\1"******"', function_call)

        # for CLI calls
        cli_password_pattern = re.compile(r"(--\w*password\w*\s+)(\S+)", re.IGNORECASE)
        function_call = cli_password_pattern.sub(r"\1******", function_call)

    # Get the current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    version = __version__

    # Write everything to the file
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Time of execution: {timestamp}\n\n")  # Write the date and time
        f.write(f"varseek version: {version}\n\n")  # Write the package version
        f.write(function_call + "\n")  # Write the function call


def check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(path1, path2):
    if path1 is not None and path2 is not None and path1 != path2:
        raise ValueError(f"{path1} and {path2} must be the same path")
    elif path1 is not None and path2 is None:
        path2 = path1
    elif path1 is None and path2 is not None:
        path1 = path2
    return path1, path2


def get_printlog(verbose=True, logger=None):
    """
    if verbose=False --> print/log nothing
    if verbose=True and logger --> logger.info
    if verbose=True and not logger --> print
    """
    return (lambda *args, **kwargs: None) if not verbose else (print if logger is None else logger.info)


def splitext_custom(file_path):
    if not isinstance(file_path, pathlib.PosixPath):
        file_path = pathlib.Path(file_path)
    base = str(file_path).replace("".join(file_path.suffixes), "")
    ext = "".join(file_path.suffixes)
    return base, ext


def get_file_name_without_extensions_or_full_path(file_path):
    return str(file_path).split("/")[-1].split(".")[0]


def check_memory_of_all_items_in_scope(scope_items, threshold=0, units='MB'):
    # scope_items should be globals().items() or locals().items()
    # threshold is the MB threshold for printing
    # beware that in my varseek functions, I might periodically delete objects from memory - if I want to see the impact of these, do a replace-all of 'del ' with '# del ' (without quotes), do the memory-checking, and then replace-all the reverse to restore back to original

    from pympler import asizeof

    memory_usage = []  # Store (name, size) tuples

    for name, obj in scope_items:
        try:
            size = asizeof.asizeof(obj)
            if units.upper() == 'BYTES':
                pass
            if units.upper() == 'KB':
                size = size / 1024
            elif units.upper() == 'MB':
                size = size / (1024 ** 2)
            elif units.upper() == 'GB':
                size = size / (1024 ** 3)
            else:
                raise ValueError("Invalid units. Must be 'bytes, 'KB', 'MB', or 'GB'.")
                
            if size >= threshold:
                memory_usage.append((name, size))
        except Exception as e:
            print(f"Could not get size of {name}: {e}")

    # Sort by size in descending order
    memory_usage.sort(key=lambda x: x[1], reverse=True)

    # Print results
    for name, size in memory_usage:
        print(f"{name}: {size:.3f} {units}")
