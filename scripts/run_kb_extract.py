import os
import shutil
import subprocess
import threading
import argparse
from rich.console import Console
console = Console()
from varseek.utils import find_genes_with_aligned_reads_for_kb_extract

def get_args():
    parser = argparse.ArgumentParser(description="Run kb extract with specified options")

    # Define the arguments
    parser.add_argument('-k', '--k', required=True, help='Value for the argument -k')
    parser.add_argument('-t', '--threads', type=int, required=True, help='Number of threads to use')
    parser.add_argument('--parallel', type=int, required=True, help='Number of parallel processes')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk size for processing')
    parser.add_argument('--kallisto', required=True, help='Path to kallisto executable')
    parser.add_argument('--adata', required=True, help='Path to adata file')
    parser.add_argument('--fastq', required=True, help='Path to synthetic reads fastq file')
    parser.add_argument('-i', '--index', required=True, help='Path to mutation index file')
    parser.add_argument('-g', '--t2g', required=True, help='Path to mutation t2g file')
    parser.add_argument('--no-mm-fast-out', required=True, help='Directory for non-multimapped output')
    parser.add_argument('--mm-fast-out', required=True, help='Directory for multimapped output')
    parser.add_argument('--slow-out', required=True, help='Directory for slow output')
    parser.add_argument('-s', '--strand', type=str, required=True, help='Strand')

    # Parse the arguments
    return parser.parse_args()



args = get_args()

k = int(args.k)
threads = args.threads
number_of_parallel_processes = args.parallel
chunk_size = args.chunk  # toward the max that kb extract can handle

kb_extract_out_total = args.no_mm_fast_out
kb_extract_out_total_mm = args.mm_fast_out
kb_extract_slow_base = args.slow_out

kallisto_large_k_path = args.kallisto
adata_path = args.adata
index = args.index
t2g = args.t2g
fastq = args.fastq

strand = args.strand


# kb extract all
print("Running kb extract all")
kb_extract_crude_command = f"kb extract --extract_all_fast --strand {strand} --verbose -k {str(k)} -t {threads} -o {kb_extract_out_total} -i {index} -g {t2g} {fastq}"
try:
    result = subprocess.run(kb_extract_crude_command, shell=True, check=True, text=True)
except subprocess.CalledProcessError as e:
    print("An error occurred:")
    print(e)
    print("Standard Error Output:")
    print(e.stderr)
print("Finished running kb extract all")


# kb extract all mm
print("Running kb extract all mm")
kb_extract_crude_opposite_mm_command = f"kb extract --extract_all_fast --strand {strand} --mm --verbose -k {str(k)} -t {threads} -o {kb_extract_out_total_mm} -i {index} -g {t2g} {fastq}"
try:
    result = subprocess.run(kb_extract_crude_opposite_mm_command, shell=True, check=True, text=True)
except subprocess.CalledProcessError as e:
    print("An error occurred:")
    print(e)
    print("Standard Error Output:")
    print(e.stderr)
print("Finished running kb extract all mm")


# kb extract slow mm
print("Running kb extract slow mm")
mapped_mutations_string = find_genes_with_aligned_reads_for_kb_extract(adata_path)
mcrs_list_with_aligned_reads = list(set(mapped_mutations_string.split()))
list_length = len(mcrs_list_with_aligned_reads)

# Function to run the command and monitor the output
def run_kb_extract_command(kb_extract_command, event):
    try:
        with subprocess.Popen(kb_extract_command, shell=True, stderr=subprocess.PIPE, text=True) as proc:
            for line in proc.stderr:
                # print(line.strip())  #* Uncomment to see the output in real-time
                if "Error:" in line:
                    console.print(f"[bold red]An error occurred: {line.strip()}[/bold red]")
                    break
                # Check for the specific line to trigger the next task
                if "Extracting reads for following transcripts for gene" in line:
                    event.set()  # Signal that the line has been found
            # Wait for the process to complete
            proc.wait()
    except Exception as e:
        print("An error occurred:")
        print(e)

# List to hold running threads and events
threads_list = []
events_list = []
semaphore = threading.Semaphore(number_of_parallel_processes)

outer_loop_counter = 0
inner_loop_counter = 0

# Loop over chunks and launch each extraction process in a new thread
for i in range(0, list_length, chunk_size):
    console.print(f"[bold green]Starting outer loop {outer_loop_counter}, inner loop {inner_loop_counter}[/bold green]")
    inner_loop_counter += 1
    semaphore.acquire()  # Limit the number of parallel processes
    mcrs_string_with_aligned_reads_chunk_i = mcrs_list_with_aligned_reads[i:i + chunk_size]
    mcrs_string_with_aligned_reads_chunk_i_string = ' '.join(f"'{x}'" for x in mcrs_string_with_aligned_reads_chunk_i)

    output_folder_chunk_i = f"{kb_extract_slow_base}/chunk_{i}"

    kb_extract_command = f"kb extract --strand {strand} --verbose -k {str(k)} --mm --kallisto {kallisto_large_k_path} -t {threads} --targets {mcrs_string_with_aligned_reads_chunk_i_string} -o {output_folder_chunk_i} -i {index} -g {t2g} {fastq}"

    # Create an event for synchronization
    event = threading.Event()
    events_list.append(event)

    # Create and start a new thread for each subprocess
    t = threading.Thread(target=run_kb_extract_command, args=(kb_extract_command, event))
    threads_list.append(t)
    t.start()

    # Wait for the event to be set before starting the next iteration
    event.wait()

    if inner_loop_counter == number_of_parallel_processes:
        # Wait for all threads to finish before starting the next round of multi-threading
        for t in threads_list:
            t.join()

        # Reset the lists and counters for the next batch
        threads_list.clear()
        events_list.clear()
        
        inner_loop_counter = 0
        outer_loop_counter += 1

# After the loop, wait for any remaining threads to finish
for t in threads_list:
    t.join()

print("Finished running kb extract slow mm")

# Loop through the level 1 directories
for level_1_dir in os.listdir(kb_extract_slow_base):
    level_1_path = os.path.join(kb_extract_slow_base, level_1_dir)
    
    # Check if it's a directory
    if os.path.isdir(level_1_path):
        
        # Loop through the level 2 directories inside the level 1 directories
        for level_2_dir in os.listdir(level_1_path):
            level_2_path = os.path.join(level_1_path, level_2_dir)
            
            # Move level 2 directories to level 0
            if os.path.isdir(level_2_path):
                shutil.move(level_2_path, kb_extract_slow_base)
                # print(f"Moved {level_2_path} to {kb_extract_slow_base}")

        # After moving, remove the now-empty level 1 directory
        if not os.listdir(level_1_path):  # If the directory is empty
            os.rmdir(level_1_path)
            # print(f"Removed empty directory {level_1_path}")

print("Finished moving directories and removing temp directories for kb extract slow.")
print("All kb extract tasks have been completed.")


