from datetime import datetime
import logging
import os
import psutil
import requests
import subprocess
import time
import tracemalloc

from bs4 import BeautifulSoup
import pandas as pd


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


# os.environ["REPORT_TIME_AND_MEMORY"] = "TRUE"
# os.environ["REPORT_TIME_AND_MEMORY_TOTAL_ONLY"] = "FALSE"

# final call, global → print normal
# not final call, global → skip
# final call, not global → print the list
# not final call, not global → print normal

def convert_to_mb(value):
    return value / 1024 / 1024

def report_time_and_memory(start=None, peaks_list=None, final_call=False, logger=None, report=True, process_name="the process", dfs = None, cols=False):
    """
    Reports elapsed time if `start` is provided, otherwise starts a timer.

    Args:
        start (float): The starting time (from `time.perf_counter()`).
        logger (logging.Logger): Optional logger to log messages. Falls back to `print` if None.
        verbose (bool): Whether to display the message.
        dfs (dict): Optional dictionary of DataFrames to report the shape of. name:df
        cols: False for no columns, True for all columns, int for top int columns

    Returns:
        float: The new start time (from `time.perf_counter()`).
    """
    
    if os.environ.get("REPORT_TIME_AND_MEMORY") != "TRUE":
        return None, None
    
    if (not final_call and os.environ.get("REPORT_TIME_AND_MEMORY_TOTAL_ONLY") == "TRUE"):
        if start is None:
            start = time.perf_counter()
            tracemalloc.start()
        return start, peaks_list

    if report:
        if start is None and peaks_list is None:
            message = "Starting timer and memory tracking"
            tracemalloc.start()
            peaks_list = []
        else:
            elapsed = time.perf_counter() - start
            minutes = int(elapsed // 60)
            seconds = elapsed % 60

            process = psutil.Process()

            if (not final_call and not os.environ.get("REPORT_TIME_AND_MEMORY_TOTAL_ONLY") == "TRUE") or (final_call and os.environ.get("REPORT_TIME_AND_MEMORY_TOTAL_ONLY") == "TRUE"):
                current, peak = tracemalloc.get_traced_memory()
                current_mb = convert_to_mb(current)
                peak_mb = convert_to_mb(peak)
                if isinstance(peaks_list, list):
                    peaks_list.append(peak_mb)

                message = f"Time and memory information for {process_name}\nRuntime: {minutes}m, {seconds:.2f}s\nMemory: current={current_mb:.3f}MB, peak={peak_mb:.3f}MB, total process={convert_to_mb(process.memory_info().rss):.3f}MB"
            elif (final_call and not os.environ.get("REPORT_TIME_AND_MEMORY_TOTAL_ONLY") == "TRUE"):
                message = f"Total runtime: {minutes}m, {seconds:.2f}s\nHighest peak memory usage: {max(peaks_list):.3f}MB\nTotal process memory: {convert_to_mb(process.memory_info().rss):.3f} MB"

            if isinstance(dfs, dict) and dfs:
                for name, df in dfs.items():
                    # check if df exists as a dataframe
                    if df is not None and hasattr(df, "shape"):
                        df_memory_usage = df.memory_usage(deep=True)
                        message += f"\ndf {name}: {convert_to_mb(df_memory_usage.sum()):.3f}MB, {df.shape[0]:,} rows, {df.shape[1]} columns"
                        if cols:
                            top_columns = df_memory_usage[1:].sort_values(ascending=False)
                            if isinstance(cols, int):
                                top_columns = top_columns.head(cols)
                            for col, usage in top_columns.items():
                                message += f"\n{col}: {convert_to_mb(usage):.3f} MB"
            
            tracemalloc.stop()
            if not final_call:
                tracemalloc.start()
        
        # Log the message
        if logger:
            logger.debug(message)
        else:
            print(message)

    else:
        # so that memory tracking gets reset if report=False
        if start is None and peaks_list is None:
            tracemalloc.start()
            peaks_list = []
        else:
            tracemalloc.stop()
            if not final_call:
                tracemalloc.start()

    # return the new start time
    return time.perf_counter(), peaks_list

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

    # iterate through rows of entex_df_tissue_selection
    number_of_samples = len(entex_df_tissue_selection)
    for index, row in entex_df_tissue_selection.iterrows():
        tissue_underscore_separated = row['tissue'].replace(" ", "_")
        sample_base_dir = os.path.join(data_download_base, row['experiment_id'], tissue_underscore_separated)
        for i in [1, 2]:
            pair_dir = f"{sample_base_dir}/pair{i}"
            os.makedirs(pair_dir, exist_ok=True)

            link = row[f'fastq_link_pair_{i}']

            download_command = f"wget -c --tries=20 --retry-connrefused -P {pair_dir} {link}"
            try:
                print(f"Downloading sample {index + 1}/{number_of_samples}, pair {i}/2 to {pair_dir}")
                result = subprocess.run(download_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error downloading {link} to {pair_dir}")
                print(e)