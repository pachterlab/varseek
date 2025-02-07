import argparse
import concurrent.futures
import json
import os
import subprocess
from pdb import set_trace as st

from varseek.utils import (
    concatenate_fastqs,
    download_t2t_reference_files,
    get_header_set_from_fastq,
    replace_low_quality_base_with_N,
    split_fastq_reads_by_N,
    trim_edges_and_adaptors_off_fastq_reads,
)


def check_for_successful_downloads(ccle_data_out_base):
    bad_samples = []

    for subfolder in os.listdir(ccle_data_out_base):
        subfolder_path = os.path.join(ccle_data_out_base, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Paths to the required subdirectories
            mutation_index_path = os.path.join(subfolder_path, "kb_count_out_mutation_index")
            standard_index_path = os.path.join(subfolder_path, "kb_count_out_standard_index")
            
            # Check if both subdirectories exist and are non-empty
            mutation_exists = os.path.exists(mutation_index_path) and os.listdir(mutation_index_path)
            standard_exists = os.path.exists(standard_index_path) and os.listdir(standard_index_path)
            
            if not mutation_exists and standard_exists:
                bad_samples.append(subfolder)

    return bad_samples

def download_ccle_total(
    record,
    mutation_index,
    mutation_t2g,
    standard_index = None,
    standard_t2g = None,
    ccle_data_out_base = ".",
    k = 31,
    k_standard = 31,
    seqtk = "seqtk",
    threads_per_task = 1,
    trim_edges_off_reads = False,
    minimum_base_quality_trim_reads = 13,
    replace_low_quality_bases_with_N = False,
    minimum_base_quality_replace_with_N = 20,
    split_reads_by_Ns = False,
    save_fastq_headers = False,
    save_fastq_files = False,
    max_retries = 5
):
    sample_accession = record.get('sample_accession')
    experiment_accession = record.get('experiment_accession')
    run_accession = record.get('run_accession')
    fastq_ftp = record.get('fastq_ftp')
    experiment_alias = record.get('experiment_alias')

    fastq_links = fastq_ftp.split(';')

    experiment_alias_underscores_only = experiment_alias.replace("-", "__")

    sample = f"{experiment_alias_underscores_only}___{sample_accession}___{experiment_accession}___{run_accession}"

    sample_out_folder = os.path.join(ccle_data_out_base, sample)

    kb_count_out_mutation_index = os.path.join(sample_out_folder, "kb_count_out_mutation_index")
    kb_count_out_standard_index = os.path.join(sample_out_folder, "kb_count_out_standard_index")

    if not os.path.exists(kb_count_out_mutation_index):
        os.makedirs(kb_count_out_mutation_index)
    
    if not os.path.exists(kb_count_out_standard_index):
        os.makedirs(kb_count_out_standard_index)

    fastq_files = []
    download_success = set()
    fastq_links_updated = []

    for link in fastq_links:
        rnaseq_fastq_file = os.path.join(sample_out_folder, os.path.basename(link))
        fastq_files.append(rnaseq_fastq_file)
        if not link.startswith(('ftp://', 'http://')):
            link = 'ftp://' + link
            fastq_links_updated.append(link)
    
    if not os.path.exists(fastq_files[0]) or not os.path.exists(fastq_files[1]):
        temp_file = os.path.join(sample_out_folder, "temp_input_files.txt")
        with open(temp_file, "w") as file:
            for item in fastq_links_updated:
                file.write(f"{item}\n")
        
        # download_command = f"curl --connect-timeout 60 --speed-time 30 --speed-limit 10000 -o {rnaseq_fastq_file} {link}"
        # download_command = f"wget -c --tries={max_retries} --retry-connrefused -O {rnaseq_fastq_file} {link}"  # --limit-rate=1m  (or some other rate)
        download_command = f"aria2c -x 16 -j 2 -d {sample_out_folder} -c -i {temp_file}"

        try:
            result = subprocess.run(download_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {fastq_links_updated}")
            print(e)
            if not save_fastq_files:
                for fastq in fastq_files:
                    os.remove(fastq)
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return
        
        if os.path.exists(temp_file):
            os.remove(temp_file)

    for rnaseq_fastq_file in fastq_files:
        # Validate the file with gzip -t
        gzip_check_command = f"gzip -t {rnaseq_fastq_file}"
        
        try:
            subprocess.run(gzip_check_command, shell=True, check=True)
            download_success.add(True)
        except subprocess.CalledProcessError:
            download_success.add(False)
            if not save_fastq_files:
                for fastq in fastq_files:
                    os.remove(fastq)
            return
    
    print(f"Downloaded all files for run accession {run_accession}")

    if len(fastq_files) == 2:
        rnaseq_fastq_file = fastq_files[0]
        rnaseq_fastq_file_2 = fastq_files[1]
    else:
        rnaseq_fastq_file = fastq_files[0]
        rnaseq_fastq_file_2 = None

    if trim_edges_off_reads:
        rnaseq_fastq_file_original, rnaseq_fastq_file_2_original = rnaseq_fastq_file, rnaseq_fastq_file_2
        rnaseq_fastq_file, rnaseq_fastq_file_2 = trim_edges_and_adaptors_off_fastq_reads(filename = rnaseq_fastq_file, filename_r2 = rnaseq_fastq_file_2, filename_filtered = None, filename_filtered_r2 = None, cut_mean_quality = minimum_base_quality_trim_reads, qualified_quality_phred = None, unqualified_percent_limit = None, n_base_limit = None, length_required = k)
        fastq_files_total = [rnaseq_fastq_file_original, rnaseq_fastq_file_2_original, rnaseq_fastq_file, rnaseq_fastq_file_2]
        fastq_files = [rnaseq_fastq_file, rnaseq_fastq_file_2]
    else:
        fastq_files_total = fastq_files

    try:
        fastqc_command = f"fastqc {rnaseq_fastq_file}"
        if rnaseq_fastq_file_2:
            fastqc_command += f" {rnaseq_fastq_file_2}"
        subprocess.run(fastqc_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running fastqc for sample {sample}")
        print(e)

    # try:
    #     multiqc_command = f"multiqc --filename multiqc --outdir {sample_out_folder} {sample_out_folder}/*fastqc*"
    #     subprocess.run(multiqc_command, shell=True, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error running multiqc for sample {sample}")
    #     print(e)

    # run kb count standard with paired end, before any further fastq modification
    if not os.path.exists(kb_count_out_standard_index) or not os.listdir(kb_count_out_standard_index):
        kb_count_standard_index_command = ["kb", "count", "-t", str(threads_per_task), "-k", str(k_standard), "-i", standard_index, "-g", standard_t2g, "-x", "bulk", "--num", "--h5ad", "--parity", "paired", "-o", kb_count_out_standard_index] + fastq_files
        try:
            subprocess.run(kb_count_standard_index_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running kb count for standard index for run accession {run_accession}")
            print(e)

    print("Done with kb count standard")

    fastq_files_copy = []

    for fastq_file in fastq_files:
        # # not supported with paired-end
        if replace_low_quality_bases_with_N:
            print(f"Replacing low-quality bases with 'N' for {fastq_file}")
            fastq_file = replace_low_quality_base_with_N(fastq_file, seqtk = seqtk, minimum_base_quality = minimum_base_quality_replace_with_N)
            fastq_files_total.append(fastq_file)

        if split_reads_by_Ns:
            print(f"Splitting reads by 'N' for {fastq_file}")
            fastq_file = split_fastq_reads_by_N(fastq_file, minimum_sequence_length = k)  # TODO: would need a way of postprocessing to make sure I don't double-count fragmented reads - I would need to see where each fragmented read aligns - perhaps with kb extract or pseudobam
            fastq_files_total.append(fastq_file)

        fastq_files_copy.append(fastq_file)

    print("Done with read preprocessing")

    fastq_files = fastq_files_copy

    if len(fastq_files) > 1:
        print(f"Concatenating fastq files for sample {sample}")
        fastq_files = concatenate_fastqs(*fastq_files)
        fastq_files_total.append(fastq_files)
    
    if save_fastq_headers:
        i = 1
        for fastq_file in fastq_files:
            fastq_header_list = get_header_set_from_fastq(fastq_file, output_format="list")
            with open(f"{sample_out_folder}/fastq_headers_{i}.txt", "w") as f:
                f.write("\n".join(fastq_header_list))
            i += 1

    if type(fastq_files) == str:
        fastq_files = [fastq_files]

    # now run single-end on mutation index
    if not os.path.exists(kb_count_out_mutation_index) or not os.listdir(kb_count_out_mutation_index):
        kb_count_mutation_index_command = ["kb", "count", "-t", str(threads_per_task), "-k", str(k), "-i", mutation_index, "-g", mutation_t2g, "-x", "bulk", "--num", "--h5ad", "--parity", "single", "-o", kb_count_out_mutation_index] + fastq_files

        if not save_fastq_headers and not save_fastq_files:
            kb_count_mutation_index_command.remove("--num")  # num only useful for keeping read header indices in bus file - but if I'm not saving read headers, then simply remove this

        try:
            subprocess.run(kb_count_mutation_index_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running kb count for mutation index for run accession {run_accession}")
            print(e)

    if not save_fastq_files:
        for fastq in fastq_files_total:
            os.remove(fastq)
        

def main(args):
    # Paths and settings from arguments
    data_to_use = args.data_to_use
    mutation_index = args.mutation_index
    mutation_t2g = args.mutation_t2g
    standard_index = args.standard_index
    standard_t2g = args.standard_t2g
    json_path = args.json_path
    ccle_data_out_base = args.ccle_data_out_base
    seqtk = args.seqtk
    k = args.k
    k_standard = args.k_standard
    threads_per_task = args.threads_per_task
    number_of_tasks_total = args.number_of_tasks_total
    trim_edges_off_reads = args.trim_edges_off_reads
    minimum_base_quality_trim_reads = args.minimum_base_quality_trim_reads
    replace_low_quality_bases_with_N = args.replace_low_quality_bases_with_N
    minimum_base_quality_replace_with_N = args.minimum_base_quality_replace_with_N
    split_reads_by_Ns = args.split_reads_by_Ns
    save_fastq_headers = args.save_fastq_headers
    save_fastq_files = args.save_fastq_files
    max_retries = args.max_retries

    if not os.path.exists(ccle_data_out_base):
        os.makedirs(ccle_data_out_base, exist_ok=True)

    if not json_path:
        json_path = "./ccle_metadata.json"

    if not os.path.exists(json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        ccle_metadata_download_command = f"wget -O {json_path} 'https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA523380&result=read_run&fields=study_accession,sample_accession,experiment_accession,run_accession,scientific_name,library_strategy,experiment_title,experiment_alias,fastq_bytes,fastq_ftp,sra_ftp&format=json&download=true&limit=0'"
        subprocess.run(ccle_metadata_download_command, shell=True, check=True)

    # Loop through json file and download fastqs
    with open(json_path, 'r') as file:
        data = json.load(file)

    rnaseq_data = [study for study in data if study['library_strategy'] == 'RNA-Seq']
    wxs_data = [study for study in data if study['library_strategy'] == 'WXS']
    rnaseq_study_accessions = {study['study_accession'] for study in rnaseq_data}
    wxs_data_with_corresponding_rnaseq_sample = [study for study in wxs_data if study['sample_accession'] in rnaseq_study_accessions]

    if data_to_use.lower() == "rnaseq":
        data_list_to_run = rnaseq_data
    elif data_to_use.lower() == "wxs":
        data_list_to_run = wxs_data
    elif data_to_use.lower() == "wxs_with_corresponding_rnaseq_sample":
        data_list_to_run = wxs_data_with_corresponding_rnaseq_sample
    elif data_to_use.lower() == "rnaseq_and_wxs" or data_to_use.lower() == "wxs_and_rnaseq":
        data_list_to_run = rnaseq_data + wxs_data
    else:
        raise ValueError("data_to_use must be one of 'rnaseq', 'wxs', 'wxs_with_corresponding_rnaseq_sample', 'rnaseq_and_wxs', or 'wxs_and_rnaseq'")

    if not standard_index:
        standard_index = "index_standard.idx"
    if not standard_t2g:
        standard_t2g = "t2g_standard.txt"

    if not os.path.exists(standard_index) or not os.path.exists(standard_t2g):
        kb_ref_standard_command = f"kb ref -d human -i {standard_index} -g {standard_t2g} -t {number_of_tasks_total}"
        raise ValueError("Standard reference files not found. Please download them using kb ref command.")  #!!! delete this and uncomment the below later
        # result = subprocess.run(kb_ref_standard_command, shell=True, check=True)

    # t2t_folder = "t2t"
    # ref_dlist_fa_genome, ref_dlist_fa_cdna, ref_dlist_gtf = download_t2t_reference_files(t2t_folder)
    # kb_ref_standard_command = f"kb ref -k {k} -i {standard_index} -g {standard_t2g} -f1 {standard_f1} -t {number_of_tasks_total} {ref_dlist_fa_genome} {ref_dlist_gtf}"
    # result = subprocess.run(kb_ref_standard_command, shell=True, check=True)

    for record in data_list_to_run:
        download_ccle_total(record=record,
                            mutation_index=mutation_index,
                            mutation_t2g=mutation_t2g,
                            standard_index=standard_index,
                            standard_t2g=standard_t2g,
                            ccle_data_out_base=ccle_data_out_base,
                            k=k,
                            k_standard=k_standard,
                            seqtk=seqtk,
                            threads_per_task=threads_per_task,
                            trim_edges_off_reads=trim_edges_off_reads,
                            minimum_base_quality_trim_reads=minimum_base_quality_trim_reads,
                            replace_low_quality_bases_with_N=replace_low_quality_bases_with_N,
                            minimum_base_quality_replace_with_N=minimum_base_quality_replace_with_N,
                            split_reads_by_Ns=split_reads_by_Ns,
                            save_fastq_headers=save_fastq_headers,
                            save_fastq_files=save_fastq_files,
                            max_retries=max_retries
                            )

    try:
        multiqc_total_command = f"find {ccle_data_out_base} -type f -name '*fastqc.zip*' | xargs multiqc --filename multiqc_total --outdir {ccle_data_out_base}"
        subprocess.run(multiqc_total_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running multiqc_total")
        print(e)

    bad_samples = check_for_successful_downloads(ccle_data_out_base)
    
    print(f"Samples with failed downloads: {bad_samples}")
    
    print("Program complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA-seq processing script with configurable parameters.")
    
    # Define arguments with argparse
    parser.add_argument("--mutation_index", type=str, required=True, help="Path to mutation index file.")
    parser.add_argument("--mutation_t2g", type=str, required=True, help="Path to mutation transcript-to-gene mapping file.")
    parser.add_argument("--data_to_use", type=str, default="rnaseq", help="Data to use (default: rnaseq) Options: 'rnaseq', 'wxs', 'wxs_with_corresponding_rnaseq_sample', 'rnaseq_and_wxs'.")
    parser.add_argument("--standard_index", type=str, default="", help="Path to standard reference index file.")
    parser.add_argument("--standard_t2g", type=str, default="", help="Path to standard reference transcript-to-gene mapping file.")
    parser.add_argument("--json_path", type=str, default="", help="Path to JSON file containing metadata.")
    parser.add_argument("--ccle_data_out_base", type=str, default=".", help="Base directory for kb output.")
    parser.add_argument("-k", "--k", type=int, default=31, help="Length of k-mer for kb count (default: 31).")
    parser.add_argument("--k_standard", type=int, default=31, help="Length of k-mer for standard index (default: 31).")
    parser.add_argument("--seqtk", type=str, default="seqtk", help="Path to seqtk executable.")
    parser.add_argument("--threads_per_task", type=int, default=1, help="Number of threads per task (default: 1).")
    parser.add_argument("--number_of_tasks_total", type=int, default=4, help="Total number of concurrent tasks (default: 4).")
    parser.add_argument("--trim_edges_off_reads", action="store_true", help="Flag to trim edges of reads.")
    parser.add_argument("--minimum_base_quality_trim_reads", type=float, default=13, help="Minimum base probability for trimming edges (default: 0.05).")
    parser.add_argument("--replace_low_quality_bases_with_N", action="store_true", help="Flag to replace low-quality bases with 'N'.")
    parser.add_argument("--minimum_base_quality_replace_with_N", type=int, default=20, help="Minimum base quality to replace with 'N' (default: 20).")
    parser.add_argument("--split_reads_by_Ns", action="store_true", help="Flag to split reads by 'N'.")
    parser.add_argument("--save_fastq_headers", action="store_true", help="Flag to save fastq headers.")
    parser.add_argument("--save_fastq_files", action="store_true", help="Flag to not delete fastq files after processing.")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries for downloading files (default: 5).")

    # Parse arguments
    args = parser.parse_args()

    # Run main processing with parsed arguments
    main(args)


# add trim_edges_off_reads (with minimum_base_quality_trim_reads), replace_low_quality_bases_with_N (with minimum_base_quality_replace_with_N argument), and/or split_reads_by_Ns
# swap --data_to_use rnaseq with wxs or wxs_with_corresponding_rnaseq_sample
# python3 download_and_align_ccle.py --mutation_index XXX --mutation_t2g XXX --standard_index XXX --standard_t2g XXX --json_path XXX --ccle_data_out_base XXX -k 55 --threads_per_task 1 --number_of_tasks_total 20 --data_to_use rnaseq