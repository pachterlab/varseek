import os
import re

from .utils import (
    set_up_logger,
    trim_edges_off_reads_fastq_list,
    run_fastqc_and_multiqc,
    replace_low_quality_bases_with_N_list,
    split_reads_by_N_list,
    concatenate_fastqs,
)

logger = set_up_logger()


def fastqpp(
    rnaseq_fastq_files_list,
    trim_edges_off_reads=False,
    run_fastqc=False,
    replace_low_quality_bases_with_N=False,
    split_reads_by_Ns=False,
    parity=None,
    fastqc_out_dir=".",
    minimum_base_quality_trim_reads=0,
    qualified_quality_phred=0,
    unqualified_percent_limit=100,
    n_base_limit=None,
    minimum_length=None,
    minimum_base_quality_replace_with_N=13,
    fastp="fastp",
    seqtk="seqtk",
    delete_intermediate_files=False,
):
    """
    Required input argument:
    - rnaseq_fastq_files_list     (list) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]

    Optional input arguments:
    - trim_edges_off_reads        (bool) If True, trim edges off reads
    - run_fastqc                  (bool) If True, run FastQC and MultiQC
    - replace_low_quality_bases_with_N (bool) If True, replace low quality bases with N
    - split_reads_by_Ns           (bool) If True, split reads by Ns
    - parity                      (str)  "single" or "paired"
    - fastqc_out_dir              (str)  Directory to save FastQC output
    - minimum_base_quality_trim_reads (int) Minimum base quality to trim reads
    - qualified_quality_phred     (int)  Phred score for qualified quality
    - unqualified_percent_limit   (int)  Percent of unqualified quality bases
    - n_base_limit                (int)  Maximum number of N bases allowed
    - minimum_length              (int)  Minimum length of reads
    - minimum_base_quality_replace_with_N (int) Minimum base quality to replace with N
    - fastp                       (str)  Path to fastp
    - seqtk                       (str)  Path to seqtk
    - delete_intermediate_files   (bool) If True, delete intermediate files
    """

    rnaseq_fastq_files_list_dict = {}
    rnaseq_fastq_files_list_dict["original"] = rnaseq_fastq_files_list

    rnaseq_fastq_files_list_sequence_only = []

    # don't include index files in any of the processing below
    for filename in rnaseq_fastq_files_list:
        match = re.search(r"_(I1|I2)_", filename)  # checks if index file (R1 and R2 are generally in scRNAseq files, not in bulk RNAseq, so I check specifically for the index notation)
        if not match:
            rnaseq_fastq_files_list_sequence_only.append(filename)

    if trim_edges_off_reads:
        print("Trimming edges off reads")
        rnaseq_fastq_files_list = trim_edges_off_reads_fastq_list(
            rnaseq_fastq_files=rnaseq_fastq_files_list,
            parity=parity,
            minimum_base_quality_trim_reads=minimum_base_quality_trim_reads,
            qualified_quality_phred=qualified_quality_phred,
            unqualified_percent_limit=unqualified_percent_limit,
            n_base_limit=n_base_limit,
            length_required=minimum_length,
            fastp=fastp,
        )
        rnaseq_fastq_files_list_dict["quality_controlled"] = rnaseq_fastq_files_list

    if run_fastqc:
        run_fastqc_and_multiqc(rnaseq_fastq_files_list, fastqc_out_dir)

    if replace_low_quality_bases_with_N:
        print("Replacing low quality bases with N")
        rnaseq_fastq_files_list = replace_low_quality_bases_with_N_list(
            rnaseq_fastq_files_quality_controlled=rnaseq_fastq_files_list,
            minimum_base_quality_replace_with_N=minimum_base_quality_replace_with_N,
            seqtk=seqtk,
        )
        rnaseq_fastq_files_list_dict["replaced_with_N"] = rnaseq_fastq_files_list

    if split_reads_by_Ns:
        print("Splitting reads by Ns")
        rnaseq_fastq_files_list = split_reads_by_N_list(
            rnaseq_fastq_files_list,
            minimum_sequence_length=minimum_length,
            delete_original_files=delete_intermediate_files,
        )
        rnaseq_fastq_files_list_dict["split_by_N"] = rnaseq_fastq_files_list

    if parity == "paired":
        print("Concatenating paired fastq files")
        rnaseq_fastq_files_list_copy = []
        for i in range(0, len(rnaseq_fastq_files_list), 2):
            file1 = rnaseq_fastq_files_list[i]
            file2 = rnaseq_fastq_files_list[i + 1]
            print(f"Concatenating {file1} and {file2}")
            file_concatenated = concatenate_fastqs(file1, file2, delete_original_files=delete_intermediate_files)
            rnaseq_fastq_files_list_copy.append(file_concatenated)
        rnaseq_fastq_files_list = rnaseq_fastq_files_list_copy
        rnaseq_fastq_files_list_dict["concatenated"] = rnaseq_fastq_files_list

    rnaseq_fastq_files_list_dict["final"] = rnaseq_fastq_files_list

    return rnaseq_fastq_files_list_dict


