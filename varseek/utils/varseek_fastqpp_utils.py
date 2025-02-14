import gzip
import os
import re
import subprocess

import pyfastx
from tqdm import tqdm

from varseek.constants import technology_barcode_and_umi_dict
from varseek.utils.logger_utils import get_printlog, is_program_installed

tqdm.pandas()


def concatenate_fastqs(*input_files, out_dir=".", delete_original_files=False, suffix="concatenatedPairs"):
    """
    Concatenate a variable number of FASTQ files (gzipped or not) into a single output file.

    Parameters:
    - output_file (str): Path to the output file.
    - *input_files (str): Paths to the input FASTQ files to concatenate.
    """
    # Detect if the files are gzipped based on file extension of the first input
    if not input_files:
        raise ValueError("No input files provided.")

    os.makedirs(out_dir, exist_ok=True)

    parts_filename = input_files[0].split(".", 1)
    output_file = os.path.join(out_dir, f"{parts_filename[0]}_{suffix}.{parts_filename[1]}")

    input_files_space_separated = " ".join(list(input_files))
    cat_command = f"cat {input_files_space_separated} > {output_file}"
    subprocess.run(cat_command, shell=True, check=True)

    if delete_original_files:
        for file in list(input_files):
            os.remove(file)

    # is_gzipped = input_files[0].endswith(".gz")
    # open_func = gzip.open if is_gzipped else open
    # with open_func(output_file, 'wt' if is_gzipped else 'w') as outfile:
    #     for file in input_files:
    #         with open_func(file, 'rt' if is_gzipped else 'r') as infile:
    #             shutil.copyfileobj(infile, outfile)

    return output_file


def split_qualities_based_on_sequence(nucleotide_sequence, quality_score_sequence):
    # Step 1: Split the original sequence by the delimiter and get the fragments
    fragments = nucleotide_sequence.split("N")

    # Step 2: Calculate the lengths of the fragments
    lengths = [len(fragment) for fragment in fragments]

    # Step 3: Use these lengths to split the associated sequence
    split_quality_score_sequence = []
    start = 0
    for length in lengths:
        split_quality_score_sequence.append(quality_score_sequence[start : (start + length)])
        start += length + 1

    return split_quality_score_sequence


def phred_to_error_rate(phred_score):
    return 10 ** (-phred_score / 10)


def trim_edges_and_adaptors_off_fastq_reads(filename, filename_r2=None, cut_mean_quality=13, cut_window_size=4, qualified_quality_phred=None, unqualified_percent_limit=None, n_base_limit=None, length_required=None, fastp="fastp", seqtk="seqtk", out_dir=".", threads=2, suffix="qc"):

    # output_dir = os.path.dirname(filename)

    # Define default output filenames if not provided
    os.makedirs(out_dir, exist_ok=True)
    parts_filename = filename.split(".", 1)
    filename_filtered = os.path.join(out_dir, f"{parts_filename[0]}_{suffix}.{parts_filename[1]}")

    try:
        fastp_command = [
            fastp,
            "-i",
            filename,
            "-o",
            filename_filtered,
            "--cut_front",
            "--cut_tail",
            "--cut_window_size",
            str(cut_window_size),
            "--cut_mean_quality",
            str(int(cut_mean_quality)),
            "-h",
            f"{out_dir}/fastp_report.html",
            "-j",
            f"{out_dir}/fastp_report.json",
            "--thread",
            str(threads),
        ]

        # Add optional parameters
        if qualified_quality_phred and unqualified_percent_limit:
            fastp_command += [
                "--qualified_quality_phred",
                str(int(qualified_quality_phred)),
                "--unqualified_percent_limit",
                str(int(unqualified_percent_limit)),
            ]
        else:
            fastp_command += [
                "--unqualified_percent_limit",
                str(100),
            ]  # * default is 40
        if n_base_limit and n_base_limit <= 50:
            fastp_command += ["--n_base_limit", str(int(n_base_limit))]
        else:
            fastp_command += ["--n_base_limit", str(50)]  # * default is 5; max is 50
        if length_required:
            fastp_command += ["--length_required", str(int(length_required))]
        else:
            fastp_command += ["--disable_length_filtering"]  # * default is 15

        # Paired-end handling
        if filename_r2:
            parts_filename_r2 = filename_r2.split(".", 1)
            filename_filtered_r2 = os.path.join(out_dir, f"{parts_filename_r2[0]}_{suffix}.{parts_filename_r2[1]}")

            fastp_command[3:3] = [
                "-I",
                filename_r2,
                "-O",
                filename_filtered_r2,
                "--detect_adapter_for_pe",
            ]

        # Run the command
        subprocess.run(fastp_command, check=True)
    except Exception as e1:
        try:
            print(f"Error: {e1}")
            print("fastp did not work. Trying seqtk")
            _ = trim_edges_of_fastq_reads_seqtk(filename, seqtk=seqtk, filename_filtered=filename_filtered, minimum_phred=cut_mean_quality, number_beginning=0, number_end=0, suffix=suffix)
            if filename_r2:
                _ = trim_edges_of_fastq_reads_seqtk(filename_r2, seqtk=seqtk, filename_filtered=filename_filtered_r2, minimum_phred=cut_mean_quality, number_beginning=0, number_end=0, suffix=suffix)
        except Exception as e2:
            print(f"Error: {e2}")
            print("seqtk did not work. Skipping QC")
            return filename, filename_r2

    return filename_filtered, filename_filtered_r2


def trim_edges_of_fastq_reads_seqtk(
    filename,
    seqtk="seqtk",
    filename_filtered=None,
    minimum_phred=13,
    number_beginning=0,
    number_end=0,
    suffix="qc",
):
    if filename_filtered is None:
        parts = filename.split(".", 1)
        filename_filtered = f"{parts[0]}_{suffix}.{parts[1]}"

    minimum_base_probability = phred_to_error_rate(minimum_phred)

    if number_beginning == 0 and number_end == 0:
        command = [seqtk, "trimfq", "-q", str(minimum_base_probability), filename]
    else:
        command = [
            seqtk,
            "trimfq",
            "-q",
            str(minimum_base_probability),
            "-b",
            str(number_beginning),
            "-e",
            str(number_end),
            filename,
        ]
    with open(filename_filtered, "w", encoding="utf-8") as output_file:
        subprocess.run(command, stdout=output_file, check=True)
    return filename_filtered


# def replace_low_quality_base_with_N_and_split_fastq_reads_by_N(input_fastq_file, output_fastq_file = None, minimum_sequence_length=31, seqtk = None, minimum_base_quality = 20):
#     parts = input_fastq_file.split(".")
#     output_replace_low_quality_with_N = f"{parts[0]}_with_Ns." + ".".join(parts[1:])
#     replace_low_quality_base_with_N(input_fastq_file, filename_filtered = output_replace_low_quality_with_N, seqtk = seqtk, minimum_base_quality = minimum_base_quality)
#     split_fastq_reads_by_N(input_fastq_file, output_fastq_file = output_fastq_file, minimum_sequence_length = minimum_sequence_length)


def replace_low_quality_base_with_N(filename, out_dir=".", seqtk="seqtk", minimum_base_quality=13, suffix="addedNs"):
    os.makedirs(out_dir, exist_ok=True)
    parts = filename.split(".", 1)
    filename_filtered = os.path.join(out_dir, f"{parts[0]}_{suffix}.{parts[1]}")
    command = [
        seqtk,
        "seq",
        "-q",
        str(minimum_base_quality),  # mask bases with quality lower than this value (<, NOT <=)
        "-n",
        "N",
        "-x",
        filename,
    ]  # to drop a read containing N, use -N
    command = " ".join(command)
    if ".gz" in parts[1]:
        command += f" | gzip > {filename_filtered}"
        # with open(filename_filtered, 'wb') as output_file:
        #     process = subprocess.Popen(command, stdout=subprocess.PIPE)
        #     subprocess.run(["gzip"], stdin=process.stdout, stdout=output_file, check=True)
        #     process.stdout.close()
        #     process.wait()
    else:
        command += f" > {filename_filtered}"
        # with open(filename_filtered, 'w', encoding="utf-8") as output_file:
        #     subprocess.run(command, stdout=output_file, check=True)
    subprocess.run(command, shell=True, check=True)
    return filename_filtered


# TODO: write this
def check_if_read_has_index_and_umi_smartseq3(sequence):
    pass
    # return True/False


def split_fastq_reads_by_N(input_fastq_file, out_dir=".", minimum_sequence_length=None, technology="bulk", contains_barcodes_or_umis=False, seqtk="seqtk", logger=None, verbose=True, suffix="splitNs"):  # set to False for bulk and for the paired file of any single-cell technology
    printlog = get_printlog(verbose, logger)
    os.makedirs(out_dir, exist_ok=True)
    parts = input_fastq_file.split(".", 1)
    output_fastq_file = os.path.join(out_dir, f"{parts[0]}_{suffix}.{parts[1]}")

    technology = technology.lower()

    if not is_program_installed(seqtk):
        logger.info("Seqtk is not installed. replace_low_quality_bases_with_N sees significant speedups for bulk technology with seqtk, so it is recommended to install seqtk for this step")
        seqtk_installed = False
    else:
        seqtk_installed = True

    if technology == "bulk" and seqtk_installed:  # use seqtk
        split_reads_by_N_command = f"{seqtk} cutN -n 1 -p 1 {input_fastq_file} | sed '/^$/d' > {output_fastq_file}"
        subprocess.run(split_reads_by_N_command, shell=True, check=True)
        if minimum_sequence_length:
            output_fastq_file_temp = f"{output_fastq_file}.tmp"
            seqtk_filter_short_read_command = f"{seqtk} seq -L {minimum_sequence_length} {output_fastq_file} > {output_fastq_file_temp}"
            try:
                subprocess.run(seqtk_filter_short_read_command, shell=True, check=True)
                # erase output_fastq_file, and rename output_fastq_file_temp to output_fastq_file
                if os.path.exists(output_fastq_file_temp):
                    os.remove(output_fastq_file)
                    os.rename(output_fastq_file_temp, output_fastq_file)
            except Exception as e:
                print(f"Error: {e}")
                printlog("seqtk seq did not work. Skipping minimum length filtering")
                if os.path.exists(output_fastq_file_temp):
                    os.remove(output_fastq_file_temp)
    else:  # must copy barcode/umi to each read, so seqtk will not work here
        if "smartseq" in technology:
            barcode_key = "spacer"
        else:
            barcode_key = "barcode"

        if technology != "bulk" and contains_barcodes_or_umis:
            if technology_barcode_and_umi_dict[technology][f"{barcode_key}_end"] is not None:
                barcode_length = technology_barcode_and_umi_dict[technology][f"{barcode_key}_end"] - technology_barcode_and_umi_dict[technology][f"{barcode_key}tart"]
            else:
                barcode_length = 0

            if technology_barcode_and_umi_dict[technology]["umi_start"] is not None:
                umi_length = technology_barcode_and_umi_dict[technology]["umi_end"] - technology_barcode_and_umi_dict[technology]["umi_start"]
            else:
                umi_length = 0

            prefix_len = barcode_length + umi_length

        prefix_len_original = prefix_len

        is_gzipped = ".gz" in parts[1]
        open_func = gzip.open if is_gzipped else open

        regex = re.compile(r"[^Nn]+")

        input_fastq_read_only = pyfastx.Fastx(input_fastq_file)
        plus_line = "+"

        with open_func(output_fastq_file, "wt") as out_file:
            for header, sequence, quality in input_fastq_read_only:
                header = header[1:]  # Remove '@' character
                if technology != "bulk" and contains_barcodes_or_umis:
                    if technology == "smartseq3":
                        sc_read_has_index_and_umi = check_if_read_has_index_and_umi_smartseq3(sequence)  # TODO: write this
                        if not sc_read_has_index_and_umi:
                            prefix_len = 0

                    barcode_and_umi_sequence = sequence[:prefix_len]
                    sequence_without_barcode_and_umi = sequence[prefix_len:]
                    barcode_and_umi_quality = quality[:prefix_len]
                    quality_without_barcode_and_umi = quality[prefix_len:]

                    prefix_len = prefix_len_original
                else:
                    sequence_without_barcode_and_umi = sequence
                    quality_without_barcode_and_umi = quality

                # Use regex to find all runs of non-"N" characters and their positions
                matches = list(regex.finditer(sequence_without_barcode_and_umi))
                if len(matches) == 1:
                    start = 1
                    end = matches[0].end()
                    new_header = f"@{header}:{start}-{end}"
                    out_file.write(f"{new_header}\n{sequence}\n{plus_line}\n{quality}\n")
                else:
                    # Extract sequence parts and their positions
                    split_sequence = [match.group() for match in matches]
                    positions = [(match.start(), match.end()) for match in matches]

                    # Use the positions to split the quality scores
                    split_qualities = [quality_without_barcode_and_umi[start:end] for start, end in positions]

                    if technology != "bulk" and contains_barcodes_or_umis:
                        split_sequence = [barcode_and_umi_sequence + sequence for sequence in split_sequence]
                        split_qualities = [barcode_and_umi_quality + quality for quality in split_qualities]

                    number_of_subsequences = len(split_sequence)
                    for i in range(number_of_subsequences):
                        if minimum_sequence_length and (len(split_sequence[i]) < minimum_sequence_length):
                            continue
                        start = matches[i].start()
                        end = matches[i].end()
                        new_header = f"@{header}:{start}-{end}"

                        out_file.write(f"{new_header}\n{split_sequence[i]}\n{plus_line}\n{split_qualities[i]}\n")

        # printlog(f"Split reads written to {output_fastq_file}")

    return output_fastq_file


def trim_edges_off_reads_fastq_list(rnaseq_fastq_files, parity, minimum_base_quality_trim_reads=0, cut_window_size=4, qualified_quality_phred=0, unqualified_percent_limit=100, n_base_limit=None, length_required=None, fastp="fastp", seqtk="seqtk", out_dir=".", threads=2, logger=None, verbose=True, suffix="qc"):
    printlog = get_printlog(verbose, logger)
    os.makedirs(out_dir, exist_ok=True)
    rnaseq_fastq_files_quality_controlled = []
    if parity == "single":
        for i in range(len(rnaseq_fastq_files)):
            printlog(f"Trimming {rnaseq_fastq_files[i]}")
            rnaseq_fastq_file, _ = trim_edges_and_adaptors_off_fastq_reads(filename=rnaseq_fastq_files[i], filename_r2=None, cut_mean_quality=minimum_base_quality_trim_reads, cut_window_size=cut_window_size, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, n_base_limit=n_base_limit, length_required=length_required, fastp=fastp, seqtk=seqtk, out_dir=out_dir, threads=threads, suffix=suffix)
            rnaseq_fastq_files_quality_controlled.append(rnaseq_fastq_file)
    elif parity == "paired":
        for i in range(0, len(rnaseq_fastq_files), 2):
            printlog(f"Trimming {rnaseq_fastq_files[i]} and {rnaseq_fastq_files[i + 1]}")
            rnaseq_fastq_file, rnaseq_fastq_file_2 = trim_edges_and_adaptors_off_fastq_reads(filename=rnaseq_fastq_files[i], filename_r2=rnaseq_fastq_files[i + 1], cut_mean_quality=minimum_base_quality_trim_reads, cut_window_size=cut_window_size, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, n_base_limit=n_base_limit, length_required=length_required, fastp=fastp, seqtk=seqtk, out_dir=out_dir, threads=threads, suffix=suffix)
            rnaseq_fastq_files_quality_controlled.extend([rnaseq_fastq_file, rnaseq_fastq_file_2])

    return rnaseq_fastq_files_quality_controlled


def run_fastqc_and_multiqc(rnaseq_fastq_files_quality_controlled, fastqc_out_dir, fastqc="fastqc", multiqc="multiqc"):
    os.makedirs(fastqc_out_dir, exist_ok=True)
    rnaseq_fastq_files_quality_controlled_string = " ".join(rnaseq_fastq_files_quality_controlled)

    try:
        fastqc_command = f"{fastqc} -o {fastqc_out_dir} {rnaseq_fastq_files_quality_controlled_string}"
        subprocess.run(fastqc_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running fastqc")
        print(e)

    try:
        multiqc_command = f"{multiqc} --filename multiqc --outdir {fastqc_out_dir} {fastqc_out_dir}/*fastqc*"
        subprocess.run(multiqc_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running multiqc")
        print(e)


def replace_low_quality_bases_with_N_list(rnaseq_fastq_files, minimum_base_quality, seqtk="seqtk", out_dir=".", delete_original_files=False, logger=None, verbose=True, suffix="addedNs"):
    printlog = get_printlog(verbose, logger)
    os.makedirs(out_dir, exist_ok=True)
    rnaseq_fastq_files_replace_low_quality_bases_with_N = []
    for i, rnaseq_fastq_file in enumerate(rnaseq_fastq_files):
        printlog(f"Replacing low quality bases with N in {rnaseq_fastq_file}")
        rnaseq_fastq_file = replace_low_quality_base_with_N(rnaseq_fastq_file, seqtk=seqtk, minimum_base_quality=minimum_base_quality, out_dir=out_dir, suffix=suffix)
        rnaseq_fastq_files_replace_low_quality_bases_with_N.append(rnaseq_fastq_file)
        # delete the file in rnaseq_fastq_files[i]
        if delete_original_files:
            os.remove(rnaseq_fastq_files[i])
    return rnaseq_fastq_files_replace_low_quality_bases_with_N


# TODO: enable single vs paired end mode (single end works as-is; paired end requires 2 files as input, and for every line it splits in file 1, I will add a line of all Ns in file 2); also get it working for scRNA-seq data (which is single end parity but still requires the paired-end treatment) - get Delaney's help to determine how to treat single cell files
def split_reads_by_N_list(rnaseq_fastq_files_replace_low_quality_bases_with_N, minimum_sequence_length=None, out_dir=".", delete_original_files=True, logger=None, verbose=True, suffix="splitNs", seqtk="seqtk"):
    printlog = get_printlog(verbose, logger)
    os.makedirs(out_dir, exist_ok=True)
    rnaseq_fastq_files_split_reads_by_N = []
    for i, rnaseq_fastq_file in enumerate(rnaseq_fastq_files_replace_low_quality_bases_with_N):
        printlog(f"Splitting reads by N in {rnaseq_fastq_file}")
        rnaseq_fastq_file = split_fastq_reads_by_N(rnaseq_fastq_file, minimum_sequence_length=minimum_sequence_length, out_dir=out_dir, logger=logger, verbose=verbose, suffix=suffix, seqtk=seqtk)  # TODO: would need a way of postprocessing to make sure I don't double-count fragmented reads - I would need to see where each fragmented read aligns - perhaps with kb extract or pseudobam
        # replace_low_quality_base_with_N_and_split_fastq_reads_by_N(input_fastq_file = rnaseq_fastq_file, output_fastq_file = None, minimum_sequence_length=k, seqtk = seqtk, minimum_base_quality = minimum_base_quality_replace_with_N)
        rnaseq_fastq_files_split_reads_by_N.append(rnaseq_fastq_file)
        # # delete the file in rnaseq_fastq_files_replace_low_quality_bases_with_N[i]
        if delete_original_files:
            os.remove(rnaseq_fastq_files_replace_low_quality_bases_with_N[i])
    return rnaseq_fastq_files_split_reads_by_N
