from datetime import datetime
import logging
import re
import os
import psutil
import requests
import subprocess
import time
import tracemalloc
import inspect
import json
from collections import OrderedDict

from bs4 import BeautifulSoup
import pandas as pd

from varseek.constants import default_filename_dict

# Mute numexpr threads info
logging.getLogger("numexpr").setLevel(logging.WARNING)

from IPython.core.magic import register_cell_magic


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
    }
    if file_path:  # skip if None or empty string, as I will provide the default path in this case
        # check if file_path is a string
        if not isinstance(file_path, str):
            raise ValueError(f"{variable_name} must be a string, got {type(file_path)}")
        
        # check if file_type is a single value or list of values
        if isinstance(file_type, str):
            valid_extensions_for_file_type = valid_extensions.get(file_type)
        elif isinstance(file_type, list) or isinstance(file_type, set) or isinstance(file_type, tuple):
            valid_extensions_for_file_type = set()
            for ft in file_type:
                valid_extensions_for_file_type.update(valid_extensions.get(ft))
        else:
            raise ValueError(f"file_type must be a string or a list, got {type(file_type)}")
        
        # check if file has valid extension
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions_for_file_type):
            raise ValueError(
                f"Invalid file extension for {variable_name}. Must be one of {valid_extensions_for_file_type}"
            )
    else:
        if required:
            raise ValueError(f"{file_type} file path is required")

def load_params(file):
    if file.endswith(".json"):
        with open(file, "r") as f:
            return json.load(f)
    elif file.endswith(".yaml") or file.endswith(".yml"):
        import yaml
        with open(file, "r") as f:
            return yaml.safe_load(f)
    else:
        print("config file format not recognized. currently supported are json and yaml.")
        return {}
    
def make_function_parameter_to_value_dict(levels_up = 1):
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
    
    # handle *args
    if varargs:
        params["*args"] = values[varargs]
    
    # handle **kwargs
    if varkw:
        for key, value in values[varkw].items():
            params[key] = value
    
    return params

def report_time_elapsed(start_time, logger = None):
    elapsed = time.perf_counter() - start_time
    time_elapsed_message = f"Total runtime for vk build\n: {int(elapsed // 60)}m, {elapsed % 60:.2f}s"
    if logger:
        logger.info(time_elapsed_message)
    else:
        print(time_elapsed_message)

def save_params_to_config_file(out_file="run_config.json"):
    out_file_directory = os.path.dirname(out_file)
    if not out_file_directory:
        out_file_directory = "."
    else:
        os.makedirs(out_file_directory, exist_ok=True)

    # Collect parameters in a dictionary
    params = make_function_parameter_to_value_dict(levels_up = 2)

    # Write to JSON
    with open(out_file, "w") as file:
        json.dump(params, file, indent=4)



def return_kb_arguments(command, remove_dashes = False):
    # Run the help command and capture the output
    result = subprocess.run(["kb", command, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    help_output = result.stdout

    # Regex pattern to match options (e.g., -x, --long-option)
    options_pattern = r"(--?\w[\w-]*)"

    # Find all matches in the help output
    arguments = re.findall(options_pattern, help_output)

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
        if flag == '--help':
            continue
        last_char_pos = len_first_flag + len(flag) - 1
        positional_arg_pos = last_char_pos + 2
        if (positional_arg_pos >= len(line)) or (line[positional_arg_pos] == ' '):
            store_true_flags.append(flag)
        else:
            flags_with_args.append(flag)

    kb_arguments = {}
    kb_arguments['store_true_flags'] = store_true_flags
    kb_arguments['flags_with_args'] = flags_with_args

    # remove dashes
    if remove_dashes:
        kb_arguments['store_true_flags'] = [
            argument.lstrip('-').replace('-', '_') for argument in kb_arguments['store_true_flags']
        ]

        kb_arguments['flags_with_args'] = [
            argument.lstrip('-').replace('-', '_') for argument in kb_arguments['flags_with_args']
        ]

    return kb_arguments


def print_varseek_dry_run(params, function_name=None):
    if function_name:
        assert function_name in {"build", "info", "filter", "fastqpp", "clean", "summarize", "ref", "count"}
        end="\n  "
        print(f"varseek.varseek_{function_name}.{function_name}(", end=end)
    for param_key, param_value in params.items():
        print(f"{param_key} = {param_value}", end=end)
    if function_name:
        print(")")

def assign_output_file_name_fordownload_varseek_files(response, out, filetype):
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

def download_varseek_files(urls_dict, out="."):
    filetype_to_filename_dict = {}
    for filetype, url in urls_dict.items():
        os.makedirs(out, exist_ok=True)

        response = requests.get(url, stream=True)

        # Check for successful response
        if response.status_code == 200:
            # Extract the filename from the Content-Disposition header
            output_file_path = assign_output_file_name_fordownload_varseek_files(response=response, out=out, filetype=filetype)
            
            with open(output_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            filetype_to_filename_dict[filetype] = output_file_path
            
            print(f"File downloaded successfully as '{output_file_path.name}'")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

    return filetype_to_filename_dict


def report_time_and_memory_of_script(script_path, argparse_flags=None, output_file=None):
    # Run the command and capture stderr, where `/usr/bin/time -l` outputs its results
    system = os.uname().sysname
    time_flag = "-v" if system == "Linux" else "-l"
    command = f"/usr/bin/time {time_flag} python3 {script_path}"
    if argparse_flags:
        command += f" {argparse_flags}"
    try:
        result = subprocess.run(command, shell=True, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    except Exception as e:
        print(f"Error running command {command}: {e}")
        return

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
            minutes = int(float(match_time.group(2)) + hours*60)
            seconds = float(match_time.group(3))
            runtime = minutes*60 + seconds
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
        with open(output_file, "w") as f:
            f.write(time_message + "\n")
            f.write(memory_message + "\n")

    return (runtime, peak_memory)  # Return the runtime and peak memory usage in bytes

def make_positional_arguments_list_and_keyword_arguments_dict():
    import sys
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


def run_command_with_error_logging(command, verbose=True):
    if isinstance(command, str):
        shell = True
    elif isinstance(command, list):
        shell = False
    else:
        raise ValueError("Command must be a string or a list.")
    
    command_string = command if shell else ' '.join(command)
    
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

def download_box_url(url, output_folder = ".", output_file_name = None):
    if not output_file_name:
        output_file_name = url.split("/")[-1]
    if "/" not in output_file_name:
        output_file_path = os.path.join(output_folder, output_file_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
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
    return {
        name for name, param in signature.parameters.items() 
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }

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



def is_valid_int(value, threshold_type=None, threshold_value=None, optional=False):
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
    elif threshold_type is None:  # No threshold comparison
        return True
    else:
        raise ValueError(f"Invalid threshold_type: {threshold_type}. Must be one of '<', '<=', '>', '>='.")






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


def find_key_with_trace(data, target_key, path=""):
    """
    Recursively search for a key in a nested JSON object (dict or list) and return its full path and values.

    Parameters:
        data (dict or list): The JSON object to search.
        target_key (str): The key to find.
        path (str): Tracks the current path in the JSON object.

    Returns:
        list: A list of tuples containing the path to the key and the associated value.
    """
    found = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if key == target_key:
                found.append((current_path, value))
            # Recursively search if the value is a nested structure
            if isinstance(value, (dict, list)):
                found.extend(find_key_with_trace(value, target_key, current_path))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            current_path = f"{path}[{index}]"
            # Recursively search each item
            found.extend(find_key_with_trace(item, target_key, current_path))

    return found

def get_experiment_links(search_url, base_url):
    response = requests.get(search_url)
    response.raise_for_status()  # Ensure the request was successful
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all experiment links
    links = [
        base_url + link['href']
        for link in soup.find_all('a', href=True)
        if '/experiments/' in link['href']
    ]
    return list(set(links))  # Remove duplicates if any

def make_entex_df():
    print("making entex df")
    base_url = "https://www.encodeproject.org"
    search_url = "https://www.encodeproject.org/search/?type=Experiment&status=released&internal_tags=ENTEx&files.file_type=fastq&control_type!=*&limit=200&assay_title=total+RNA-seq"
    experiment_links = get_experiment_links(search_url=search_url, base_url=base_url)

    entex_list_of_dicts = []

    for experiment_link in experiment_links:
        entex_experiment_dict = {}
        experiment_id = experiment_link.split("/")[-2]
        json_data_url = f"{experiment_link}/?format=json"

        response = requests.get(json_data_url)
        response.raise_for_status()
        experiment_data = response.json()

        # donor_values = find_key_with_trace(experiment_data, 'donor')  # to find the trace in the json - returns as list of key_path, value pairs

        description = experiment_data['description']
        tissue = experiment_data['biosample_ontology']['term_name']
        organ_slims = experiment_data['biosample_ontology']['organ_slims']
        age = experiment_data['replicates'][0]['library']['biosample']['donor']['age']  # same as donor_values[0][1]['age']
        sex = experiment_data['replicates'][0]['library']['biosample']['donor']['sex']  # same as donor_values[0][1]['sex']

        # Extract FASTQ file links
        fastq_files_metadata = [
            file
            for file in experiment_data.get('files', [])
            if file['file_type'] == 'fastq' and file['replicate']['technical_replicate_number'] == 1
        ]

        if len(fastq_files_metadata) > 2:
            date_created_to_match = fastq_files_metadata[0]['date_created'].split("T")[0]
            fastq_files_metadata = [metadata for metadata in fastq_files_metadata if metadata['date_created'].split("T")[0] == date_created_to_match]

        fastq_files = [file['href'] for file in fastq_files_metadata]
        read_lengths = [file['read_length'] for file in fastq_files_metadata]
        paired_ends = [file['paired_end'] for file in fastq_files_metadata]
        date_created = [file['date_created'] for file in fastq_files_metadata]

        # reorder fastq_files according to paired_ends
        fastq_files = [fastq_files[int(i) - 1] for i in paired_ends]

        if len(set(read_lengths)) != 1:
            print(f"skipping experiment {experiment_id} due to having {len(set(read_lengths))} read lengths: {set(read_lengths)}")
        
        read_length = read_lengths[0]

        fastq_links = [base_url + link for link in fastq_files]

        if len(fastq_links) != 2:
            print(f"skipping experiment {experiment_id} due to having {len(fastq_links)} fastq links: {fastq_links}")

        entex_experiment_dict['experiment_id'] = experiment_id
        entex_experiment_dict['description'] = description
        entex_experiment_dict['tissue'] = tissue
        entex_experiment_dict['organ_slims'] = organ_slims
        entex_experiment_dict['age'] = age
        entex_experiment_dict['sex'] = sex
        entex_experiment_dict['read_length'] = read_length
        entex_experiment_dict['fastq_link_pair_1'] = fastq_links[0]
        entex_experiment_dict['fastq_link_pair_2'] = fastq_links[1]

        experiment_data['biosample_ontology']['term_name']

        entex_list_of_dicts.append(entex_experiment_dict)

    entex_df = pd.DataFrame(entex_list_of_dicts)

    return entex_df

def download_entex_fastq_links(entex_df, tissue = None, data_download_base = "."):
    print("downloading fastq files")
    if tissue:
        entex_df_tissue_selection = entex_df.loc[entex_df['tissue'] == tissue].reset_index(drop=True)
    else:
        entex_df_tissue_selection = entex_df

    # iterate through rows of entex_df_tissue_selection
    number_of_samples = len(entex_df_tissue_selection)
    for index, row in entex_df_tissue_selection.iterrows():
        tissue_underscore_separated = row['tissue'].replace(" ", "_")
        sample_base_dir = os.path.join(data_download_base, tissue_underscore_separated, row['experiment_id'])
        for i in [1, 2]:
            pair_dir = f"{sample_base_dir}/pair{i}"
            os.makedirs(pair_dir, exist_ok=True)

            link = row[f'fastq_link_pair_{i}']

            download_command = f"wget -c --tries=20 --retry-connrefused -P {pair_dir} {link}"

            if not os.path.exists(f"{pair_dir}/{link.split('/')[-1]}"):
                try:
                    print(f"Downloading sample {index + 1}/{number_of_samples}, pair {i}/2 to {pair_dir}")
                    result = subprocess.run(download_command, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error downloading {link} to {pair_dir}")
                    print(e)
            else:
                print(f"File {pair_dir}/{link.split('/')[-1]} already exists, skipping download")