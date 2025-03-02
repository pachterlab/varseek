import gzip
import os
import re
import shutil
import subprocess

import pyfastx
from tqdm import tqdm

from varseek.constants import technology_barcode_and_umi_dict
from varseek.utils.logger_utils import get_printlog, is_program_installed

tqdm.pandas()


def concatenate_fastqs(*input_files, out_dir=".", delete_original_files=False):
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

    filename_0 = os.path.basename(input_files[0]).split(".", 1)[0]
    filename_1 = os.path.basename(input_files[1]).split(".", 1)[0]
    ext = os.path.basename(input_files[0]).split(".", 1)[1]

    output_file = os.path.join(out_dir, f"{filename_0}_{filename_1}.{ext}")

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


def replace_low_quality_base_with_N(filename, out_dir=".", seqtk="seqtk", minimum_base_quality=13):
    os.makedirs(out_dir, exist_ok=True)
    filename_filtered = os.path.join(out_dir, os.path.basename(filename))
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
    if ".gz" in filename:
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
    logger_info = get_printlog(verbose=verbose, logger=logger)
    os.makedirs(out_dir, exist_ok=True)
    parts = input_fastq_file.split(".", 1)
    output_fastq_file = os.path.join(out_dir, os.path.basename(input_fastq_file))

    technology = technology.lower()

    if not is_program_installed(seqtk):
        logger.info("Seqtk is not installed. split_reads_by_Ns_and_low_quality_bases sees significant speedups for bulk technology with seqtk, so it is recommended to install seqtk for this step")
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
                logger_info("seqtk seq did not work. Skipping minimum length filtering")
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

        # logger_info(f"Split reads written to {output_fastq_file}")

    return output_fastq_file


def trim_edges_off_reads_fastq_list(rnaseq_fastq_files, parity, minimum_base_quality_trim_reads=0, cut_window_size=4, qualified_quality_phred=0, unqualified_percent_limit=100, n_base_limit=None, length_required=None, fastp="fastp", seqtk="seqtk", out_dir=".", threads=2, logger=None, verbose=True, suffix="qc"):
    logger_info = get_printlog(verbose=verbose, logger=logger)
    os.makedirs(out_dir, exist_ok=True)
    rnaseq_fastq_files_quality_controlled = []
    if parity == "single":
        for i in range(len(rnaseq_fastq_files)):
            logger_info(f"Trimming {rnaseq_fastq_files[i]}")
            rnaseq_fastq_file, _ = trim_edges_and_adaptors_off_fastq_reads(filename=rnaseq_fastq_files[i], filename_r2=None, cut_mean_quality=minimum_base_quality_trim_reads, cut_window_size=cut_window_size, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, n_base_limit=n_base_limit, length_required=length_required, fastp=fastp, seqtk=seqtk, out_dir=out_dir, threads=threads, suffix=suffix)
            rnaseq_fastq_files_quality_controlled.append(rnaseq_fastq_file)
    elif parity == "paired":
        for i in range(0, len(rnaseq_fastq_files), 2):
            logger_info(f"Trimming {rnaseq_fastq_files[i]} and {rnaseq_fastq_files[i + 1]}")
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


def replace_low_quality_bases_with_N_list(rnaseq_fastq_files, minimum_base_quality, out_dir, seqtk="seqtk", delete_original_files=False, logger=None, verbose=True):
    logger_info = get_printlog(verbose=verbose, logger=logger)
    os.makedirs(out_dir, exist_ok=True)
    rnaseq_fastq_files_replace_low_quality_bases_with_N = []
    for i, rnaseq_fastq_file in enumerate(rnaseq_fastq_files):
        logger_info(f"Replacing low quality bases with N in {rnaseq_fastq_file}")
        rnaseq_fastq_file = replace_low_quality_base_with_N(rnaseq_fastq_file, seqtk=seqtk, minimum_base_quality=minimum_base_quality, out_dir=out_dir)
        rnaseq_fastq_files_replace_low_quality_bases_with_N.append(rnaseq_fastq_file)
        # delete the file in rnaseq_fastq_files[i]
        if delete_original_files:
            os.remove(rnaseq_fastq_files[i])
    return rnaseq_fastq_files_replace_low_quality_bases_with_N


# TODO: enable single vs paired end mode (single end works as-is; paired end requires 2 files as input, and for every line it splits in file 1, I will add a line of all Ns in file 2); also get it working for scRNA-seq data (which is single end parity but still requires the paired-end treatment) - get Delaney's help to determine how to treat single cell files
def split_reads_by_N_list(rnaseq_fastq_files_replace_low_quality_bases_with_N, minimum_sequence_length=None, out_dir=".", delete_original_files=True, logger=None, verbose=True, seqtk="seqtk"):
    logger_info = get_printlog(verbose=verbose, logger=logger)
    os.makedirs(out_dir, exist_ok=True)
    rnaseq_fastq_files_split_reads_by_N = []
    for i, rnaseq_fastq_file in enumerate(rnaseq_fastq_files_replace_low_quality_bases_with_N):
        logger_info(f"Splitting reads by N in {rnaseq_fastq_file}")
        rnaseq_fastq_file = split_fastq_reads_by_N(rnaseq_fastq_file, minimum_sequence_length=minimum_sequence_length, out_dir=out_dir, logger=logger, verbose=verbose, seqtk=seqtk)  # TODO: would need a way of postprocessing to make sure I don't double-count fragmented reads - I would need to see where each fragmented read aligns - perhaps with kb extract or pseudobam
        # replace_low_quality_base_with_N_and_split_fastq_reads_by_N(input_fastq_file = rnaseq_fastq_file, output_fastq_file = None, minimum_sequence_length=k, seqtk = seqtk, minimum_base_quality = minimum_base_quality_replace_with_N)
        rnaseq_fastq_files_split_reads_by_N.append(rnaseq_fastq_file)
        # # delete the file in rnaseq_fastq_files_replace_low_quality_bases_with_N[i]
        if delete_original_files:
            os.remove(rnaseq_fastq_files_replace_low_quality_bases_with_N[i])
    return rnaseq_fastq_files_split_reads_by_N



import pyfastx


def ensure_read_agreement(r1_unfiltered_fastq_path, r2_unfiltered_fastq_path, removed_reads_fastq_path, r1_fastq_out_path=None, indices_file_path=None, delete_indices_file=True):
    # assumes that I ran fastp on r2_unfiltered_fastq_path (transcripts) with some unintended filtering using --failed_out removed_reads_fastq_path, and I want to filter the same reads out of r1_unfiltered_fastq_path (barcodes/UMIs)
    
    if os.path.getsize(removed_reads_fastq_path) == 0:
        print("No reads were removed from the transcripts. No need to filter the barcodes/UMIs.")
        return

    if not r1_fastq_out_path:
        r1_fastq_out_path = r1_unfiltered_fastq_path  # overwrite the original file
    
    removed_indices_out_file_tmp = indices_file_path if indices_file_path else "removed_indices_tmp.txt"

    if not os.path.isfile(removed_indices_out_file_tmp):
        # Create a set of removed read headers
        removed_reads_fastq = pyfastx.Fastx(removed_reads_fastq_path)
        removed_read_headers = set()
        for header, _, _ in removed_reads_fastq:
            removed_read_headers.add(header)  # pyfastx takes the part of the header up until the first space

        # Create a list of the indices of the removed read headers from R2 and write to a txt file
        r2_unfiltered_fastq = pyfastx.Fastx(r2_unfiltered_fastq_path)
        indices = [i for i, (header, _, _) in enumerate(r2_unfiltered_fastq) if header in removed_read_headers]
        with open(removed_indices_out_file_tmp, "w") as f:
            f.writelines(f"{index}\n" for index in indices)

    try:
        # Remove the reads from the gathered indices from R1
        awk_command = [
            "awk",
            'NR==FNR {omit[$1]; next} '
            'FNR % 4 == 1 { i++; if ((i-1) in omit) { skip=4 } else { skip=0 } } '
            'skip > 0 { skip--; next } '
            '{ print }',
            removed_indices_out_file_tmp,
            r1_unfiltered_fastq_path
        ]

        with open(r1_fastq_out_path, "w") as output_file:
            subprocess.run(awk_command, stdout=output_file, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error occurred while running awk command: {e}") from e
    finally:
        if delete_indices_file:
            os.remove(removed_indices_out_file_tmp)



def run_fastp_bulk(r1_fastq_path, r2_fastq_path=None, out_dir="filtered", parity="single", cut_front=False, cut_tail=False, cut_window_size=4, cut_mean_quality=15, disable_adapter_trimming=False, qualified_quality_phred=15, unqualified_percent_limit=40, average_qual=15, n_base_limit=10, disable_quality_filtering=False, length_required=31, disable_length_filtering=False, dont_eval_duplication=True, disable_trim_poly_g=True, threads=2, failed_out=False, r1_fastq_out_path="r1_filtered.fq", r2_fastq_out_path="r2_filtered.fq"):
    fastp_cmd = ["fastp", "-i", r1_fastq_path, "-o", r1_fastq_out_path]
        
    if parity == "paired":
        fastp_cmd += ["-I", r2_fastq_path, "-O", r2_fastq_out_path]
            #? consider adding --merged (can merge overlapping reads from paired end data)
    if cut_front:
        fastp_cmd += ["--cut_front"]
    if cut_tail:
        fastp_cmd += ["--cut_tail"]
    if cut_front or cut_tail:
        fastp_cmd += ["--cut_window_size", str(cut_window_size), "--cut_mean_quality", str(cut_mean_quality)]
    if disable_adapter_trimming:
        fastp_cmd += ["--disable_adapter_trimming"]
    else:
        if parity == "paired":
            fastp_cmd += ["--detect_adapter_for_pe"]
        
    if disable_quality_filtering:
        fastp_cmd += ["--disable_quality_filtering"]
    else:
        fastp_cmd += ["--qualified_quality_phred", str(qualified_quality_phred), "--unqualified_percent_limit", str(unqualified_percent_limit), "--average_qual", str(average_qual), "--n_base_limit", str(n_base_limit)]

    if disable_length_filtering:
        fastp_cmd += ["--disable_length_filtering"]
    else:
        fastp_cmd += ["--length_required", str(length_required)]
        
    if dont_eval_duplication:
        fastp_cmd += ["--dont_eval_duplication"]
    if disable_trim_poly_g:
        fastp_cmd += ["--disable_trim_poly_g"]

    fastp_cmd += ["--thread", str(threads)]

    fastp_cmd += ["-h", os.path.join(out_dir, "fastp_report.html"), "-j", os.path.join(out_dir, "fastp_report.json")]

    if failed_out:
        fastp_cmd += ["--failed_out", failed_out]

    subprocess.run(fastp_cmd, check=True)

def run_fastp_single_cell_general(r1_fastq_path, r2_fastq_path, out_dir="filtered", cut_front=False, cut_tail=False, cut_window_size=4, cut_mean_quality=15, disable_adapter_trimming=False, qualified_quality_phred=15, unqualified_percent_limit=40, average_qual=15, n_base_limit=10, disable_quality_filtering=False, threads=2, failed_out=False, r1_fastq_out_path="r1_filtered.fq", r2_fastq_out_path="r2_filtered.fq", tmp_dir="tmp"):
    #* rather than doing fastp twice as I do below (once for quality filtering and once for edge trimming), I could do it all in one go, but that would require running ensure_read_agreement, which I haven't thoroughly debugged or benchmarked for runtime compared to another fastp command - if I debug ensure_read_agreement and either (1) find ensure_read_agreement is much faster than another fastp OR (2) find that I am running ensure_read_agreement with the current setup anyways, then replace the current setup with a single fastp call that combiens both read filtering and edge trimming, followed by ensure_read_agreement (this has the added benefit of being 100% sure that I don't factor barcode/UMI information in at all when filtering - plus, I could include length filtering again)
    if average_qual < cut_mean_quality:
        print("Warning: average_qual is less than cut_mean_quality. This means that ensure_read_agreement might need to run.")

    if not disable_quality_filtering:
        r1_fastq_out_path_tmp = os.path.join(tmp_dir, os.path.basename(r1_fastq_path))
        r2_fastq_out_path_tmp = os.path.join(tmp_dir, os.path.basename(r2_fastq_path))
        
        # low quality read removal
        fastp_cmd1 = ["fastp", "-i", r1_fastq_path, "-I", r2_fastq_path, "-o", r1_fastq_out_path_tmp, "-O", r2_fastq_out_path_tmp,
                          "--disable_adapter_trimming",
                          "--qualified_quality_phred", str(qualified_quality_phred),
                          "--unqualified_percent_limit", str(unqualified_percent_limit),
                          "--average_qual", str(average_qual),
                          "--n_base_limit", str(n_base_limit),
                          "--disable_length_filtering",
                          "--dont_eval_duplication",
                          "--disable_trim_poly_g",
                          "-h", os.path.join(out_dir, "fastp_report.html"),
                          "-j", os.path.join(out_dir, "fastp_report.json")
                          ]
        
        fastp_cmd1 += ["--thread", str(threads)]

        if failed_out:
            fastp_cmd1 += ["--failed_out", failed_out]

        subprocess.run(fastp_cmd1, check=True) 
    else:
        r1_fastq_out_path_tmp = r1_fastq_path
        r2_fastq_out_path_tmp = r2_fastq_path
        
    # edge trimming
    if cut_front or cut_tail or not disable_adapter_trimming:
        fastp_cmd2 = ["fastp", "-i", r2_fastq_out_path_tmp, "-o", r2_fastq_out_path]
            
        if cut_front:
            fastp_cmd2 += ["--cut_front"]
        if cut_tail:
            fastp_cmd2 += ["--cut_tail"]
        if cut_front or cut_tail:
            fastp_cmd2 += ["--cut_window_size", str(cut_window_size), "--cut_mean_quality", str(cut_mean_quality)]
        if disable_adapter_trimming:
            fastp_cmd2 += ["--disable_adapter_trimming"]
            
        fastp_cmd2 += ["--disable_quality_filtering", "--disable_length_filtering", "--dont_eval_duplication", "--disable_trim_poly_g"]

        fastp_cmd2 += ["--thread", str(threads)]
            
        if failed_out:
            failed_out2 = os.path.join(out_dir, "removed_reads2.fastq")
        else:
            failed_out2 = os.path.join(tmp_dir, "removed_reads2.fastq")  # tmp
            
        fastp_cmd2 += ["--failed_out", failed_out2]
            
        fastp_cmd2 += ["-h", os.path.join(out_dir, "fastp_report2.html"), "-j", os.path.join(out_dir, "fastp_report2.json")]

        subprocess.run(fastp_cmd2, check=True)

        if os.path.getsize(failed_out2) > 0:
            print(f"Removing reads from {r1_fastq_out_path} removed solely from {r2_fastq_out_path} during trimming")
            ensure_read_agreement(r1_fastq_out_path_tmp, r2_fastq_out_path_tmp, failed_out2, r1_fastq_out_path=r1_fastq_out_path)
    else:
        os.rename(r1_fastq_out_path_tmp, r1_fastq_out_path)
        os.rename(r2_fastq_out_path_tmp, r2_fastq_out_path)

def run_fastp_smartseq3(r1_fastq_path, r2_fastq_path, out_dir="filtered", cut_front=False, cut_tail=False, cut_window_size=4, cut_mean_quality=15, disable_adapter_trimming=False, qualified_quality_phred=15, unqualified_percent_limit=40, average_qual=15, n_base_limit=10, disable_quality_filtering=False, length_required=31, disable_length_filtering=False, threads=2, failed_out=False, r1_fastq_out_path="r1_filtered.fq", r2_fastq_out_path="r2_filtered.fq", tmp_dir="tmp"):
    if average_qual < cut_mean_quality:
        print("Warning: average_qual is less than cut_mean_quality. This means that ensure_read_agreement might need to run.")
    
    r1_fastq_out_path_tmp = os.path.join(tmp_dir, os.path.basename(r1_fastq_path))
    r2_fastq_out_path_tmp = os.path.join(tmp_dir, os.path.basename(r2_fastq_path))
    fastp_cmd = ["fastp", "-i", r1_fastq_path, "-I", r2_fastq_path, "-o", r1_fastq_out_path_tmp, "-O", r2_fastq_out_path_tmp]
        
    if cut_tail:
        fastp_cmd += ["--cut_tail", "--cut_window_size", str(cut_window_size), "--cut_mean_quality", str(cut_mean_quality)]
    if disable_adapter_trimming:
        fastp_cmd += ["--disable_adapter_trimming"]
    else:
        fastp_cmd += ["--detect_adapter_for_pe"]
        
    if disable_quality_filtering:
        fastp_cmd += ["--disable_quality_filtering"]
    else:
        fastp_cmd += ["--qualified_quality_phred", str(qualified_quality_phred), "--unqualified_percent_limit", str(unqualified_percent_limit), "--average_qual", str(average_qual), "--n_base_limit", str(n_base_limit)]

    if disable_length_filtering:
        fastp_cmd += ["--disable_length_filtering"]
    else:
        fastp_cmd += ["--length_required", str(length_required)]
        
    fastp_cmd += ["--dont_eval_duplication", "--disable_trim_poly_g", "--thread", str(threads)]

    fastp_cmd += ["-h", os.path.join(out_dir, "fastp_report.html"), "-j", os.path.join(out_dir, "fastp_report.json")]

    if failed_out:
        fastp_cmd += ["--failed_out", failed_out]

    subprocess.run(fastp_cmd, check=True)

    if cut_front:
        failed_out2 = os.path.join(tmp_dir, "removed_reads2.fastq")
        fastp_cmd2 = ["fastp", "-i", r2_fastq_out_path_tmp, "-o", r2_fastq_out_path,
                          "--cut_front", "--cut_window_size", str(cut_window_size), "--cut_mean_quality", str(cut_mean_quality),
                          "--disable_adapter_trimming", "--disable_quality_filtering", "--disable_length_filtering", "--dont_eval_duplication", "--disable_trim_poly_g", "--thread", str(threads),
                          "-h", os.path.join(out_dir, "fastp_report2.html"), "-j", os.path.join(out_dir, "fastp_report2.json"), "--failed_out", failed_out2]
            
        subprocess.run(fastp_cmd2, check=True)

        if os.path.getsize(failed_out2) > 0:
            print(f"Removing reads from {r1_fastq_out_path} removed solely from {r2_fastq_out_path} during trimming")
            ensure_read_agreement(r1_fastq_out_path_tmp, r2_fastq_out_path_tmp, failed_out2, r1_fastq_out_path=r1_fastq_out_path)
    else:
        os.rename(r1_fastq_out_path_tmp, r1_fastq_out_path)
        os.rename(r2_fastq_out_path_tmp, r2_fastq_out_path)
    
def run_fastp_10xv3_ultima(r1_fastq_path, out_dir, cut_tail=False, cut_window_size=4, cut_mean_quality=15, disable_adapter_trimming=False, qualified_quality_phred=15, unqualified_percent_limit=40, average_qual=15, n_base_limit=10, disable_quality_filtering=False, length_required=31, disable_length_filtering=False, threads=2, failed_out=False, r1_fastq_out_path="r1_filtered.fq"):
    fastp_cmd = ["fastp", "-i", r1_fastq_path, "-o", r1_fastq_out_path]
        
    if cut_tail:
        fastp_cmd += ["--cut_tail", "--cut_window_size", str(cut_window_size), "--cut_mean_quality", str(cut_mean_quality)]
    if disable_adapter_trimming:
        fastp_cmd += ["--disable_adapter_trimming"]
        
    if disable_quality_filtering:
        fastp_cmd += ["--disable_quality_filtering"]
    else:
        fastp_cmd += ["--qualified_quality_phred", str(qualified_quality_phred), "--unqualified_percent_limit", str(unqualified_percent_limit), "--average_qual", str(average_qual), "--n_base_limit", str(n_base_limit)]

    if disable_length_filtering:
        fastp_cmd += ["--disable_length_filtering"]
    else:
        fastp_cmd += ["--length_required", str(length_required)]

    fastp_cmd += ["--dont_eval_duplication", "--disable_trim_poly_g", "--thread", str(threads)]

    fastp_cmd += ["-h", os.path.join(out_dir, "fastp_report.html"), "-j", os.path.join(out_dir, "fastp_report.json")]

    if failed_out:
        fastp_cmd += ["--failed_out", failed_out]

    subprocess.run(fastp_cmd, check=True)

def run_fastp_10xv1(r1_fastq_path, r2_fastq_path, i1_fastq_path, out_dir, cut_front=False, cut_tail=False, cut_window_size=4, cut_mean_quality=15, disable_adapter_trimming=False, qualified_quality_phred=15, unqualified_percent_limit=40, average_qual=15, n_base_limit=10, disable_quality_filtering=False, length_required=31, disable_length_filtering=False, threads=2, failed_out=False, r1_fastq_out_path="r1_filtered.fq", r2_fastq_out_path="r2_filtered.fq", i1_fastq_out_path="i1_filtered.fq"):
    fastp_cmd = ["fastp", "-i", r2_fastq_path, "-o", r2_fastq_out_path]
        
    if cut_front:
        fastp_cmd += ["--cut_front"]
    if cut_tail:
        fastp_cmd += ["--cut_tail"]
    if cut_front or cut_tail:
        fastp_cmd += ["--cut_window_size", str(cut_window_size), "--cut_mean_quality", str(cut_mean_quality)]
    if disable_adapter_trimming:
        fastp_cmd += ["--disable_adapter_trimming"]
        
    if disable_quality_filtering:
        fastp_cmd += ["--disable_quality_filtering"]
    else:
        fastp_cmd += ["--qualified_quality_phred", str(qualified_quality_phred), "--unqualified_percent_limit", str(unqualified_percent_limit), "--average_qual", str(average_qual), "--n_base_limit", str(n_base_limit)]

    if disable_length_filtering:
        fastp_cmd += ["--disable_length_filtering"]
    else:
        fastp_cmd += ["--length_required", str(length_required)]

    fastp_cmd += ["--dont_eval_duplication", "--disable_trim_poly_g", "--thread", str(threads)]

    fastp_cmd += ["-h", os.path.join(out_dir, "fastp_report.html"), "-j", os.path.join(out_dir, "fastp_report.json"), "--failed_out", failed_out]

    subprocess.run(fastp_cmd, check=True)

    ensure_read_agreement(r1_fastq_path, r2_fastq_path, failed_out, delete_indices_file=False, r1_fastq_out_path=r1_fastq_out_path)
    ensure_read_agreement(i1_fastq_path, r2_fastq_path, failed_out, delete_indices_file=False, r1_fastq_out_path=i1_fastq_out_path)


def perform_fastp_trimming_and_filtering(technology, r1_fastq_path, r2_fastq_path=None, i1_fastq_path=None, i2_fastq_path=None, out_dir="filtered", parity="single", cut_front=False, cut_tail=False, cut_window_size=4, cut_mean_quality=15, disable_adapter_trimming=True, qualified_quality_phred=15, unqualified_percent_limit=40, average_qual=15, n_base_limit=10, disable_quality_filtering=True, length_required=31, disable_length_filtering=True, dont_eval_duplication=True, disable_trim_poly_g=True, threads=2, failed_out=False):
    """
    Perform trimming and filtering of FASTQ files using fastp.

    Parameters:
    - technology                    (str): The technology used for sequencing. From kb --list
    - r1_fastq_path                 (str): For single-cell, path to the R1 FASTQ file. Within the function, I use r1_fastq_path to be the fastq with the barcodes/UMIs (for technologies where R1 has transcripts and R2 has barcodes/UMIs, I swap internally early on). For bulk, path to the first FASTQ file. This
    - r2_fastq_path                 (str): For single-cell, path to the R2 FASTQ file. Within the function, I use r2_fastq_path to be the fastq with the transcripts. For bulk, path to the second FASTQ file if paired.
    - i1_fastq_path                 (str): For single-cell, path to the I1 FASTQ file. For bulk, path to the third FASTQ file if paired.
    ...
    """
    
    if all((not cut_front, not cut_tail, disable_adapter_trimming, disable_quality_filtering, disable_length_filtering, dont_eval_duplication, disable_trim_poly_g)):
        print("No trimming or filtering options selected. Exiting.")
        return {"R1": r1_fastq_path, "R2": r2_fastq_path, "I1": i1_fastq_path, "I2": i2_fastq_path}
    
    for input_file in [r1_fastq_path, r2_fastq_path, i1_fastq_path]:
        if input_file is not None and not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    os.makedirs(out_dir, exist_ok=True)
    r1_fastq_out_path = os.path.join(out_dir, os.path.basename(r1_fastq_path))
    r2_fastq_out_path = os.path.join(out_dir, os.path.basename(r2_fastq_path)) if r2_fastq_path else None
    i1_fastq_out_path = os.path.join(out_dir, os.path.basename(i1_fastq_path)) if i1_fastq_path else None

    tmp_dir = os.path.join(out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    if technology in {"10XV1", "INDROPSV3"}:
        failed_out = True

    if failed_out is True:
        failed_out = os.path.join(out_dir, "removed_reads.fastq")

    technology = technology.upper()
    if technology in {"INDROPSV2", "SPLIT-SEQ", "STORMSEQ"}:  # a technology where the latter file has the barcode/UMI-type information
        r1_fastq_path, r2_fastq_path = r2_fastq_path, r1_fastq_path  # swap R1 and R2

    if technology in {"BULK", "SMARTSEQ2"}:  # no barcodes or UMIs
        run_fastp_bulk(r1_fastq_path=r1_fastq_path, r2_fastq_path=r2_fastq_path, out_dir=out_dir, parity=parity, cut_front=cut_front, cut_tail=cut_tail, cut_window_size=cut_window_size, cut_mean_quality=cut_mean_quality, disable_adapter_trimming=disable_adapter_trimming, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, average_qual=average_qual, n_base_limit=n_base_limit, disable_quality_filtering=disable_quality_filtering, length_required=length_required, disable_length_filtering=disable_length_filtering, dont_eval_duplication=dont_eval_duplication, disable_trim_poly_g=disable_trim_poly_g, threads=threads, failed_out=failed_out, r1_fastq_out_path=r1_fastq_out_path, r2_fastq_out_path=r2_fastq_out_path)

    elif technology in {"10XV2, 10XV3", "BDWTA", "CELSEQ", "CELSEQ2", "INDROPSV1", "INDROPSV2", "SCRUBSEQ", "SPLIT-SEQ", "SURECELL", "VISIUM"}:
        run_fastp_single_cell_general(r1_fastq_path=r1_fastq_path, r2_fastq_path=r2_fastq_path, out_dir=out_dir, cut_front=cut_front, cut_tail=cut_tail, cut_window_size=cut_window_size, cut_mean_quality=cut_mean_quality, disable_adapter_trimming=disable_adapter_trimming, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, average_qual=average_qual, n_base_limit=n_base_limit, disable_quality_filtering=disable_quality_filtering, threads=threads, failed_out=failed_out, r1_fastq_out_path=r1_fastq_out_path, r2_fastq_out_path=r2_fastq_out_path, tmp_dir=tmp_dir)
        
    elif technology in {"SMARTSEQ3", "STORMSEQ"}:
        run_fastp_smartseq3(r1_fastq_path=r1_fastq_path, r2_fastq_path=r2_fastq_path, out_dir=out_dir, cut_front=cut_front, cut_tail=cut_tail, cut_window_size=cut_window_size, cut_mean_quality=cut_mean_quality, disable_adapter_trimming=disable_adapter_trimming, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, average_qual=average_qual, n_base_limit=n_base_limit, disable_quality_filtering=disable_quality_filtering, length_required=length_required, disable_length_filtering=disable_length_filtering, threads=threads, failed_out=failed_out, r1_fastq_out_path=r1_fastq_out_path, r2_fastq_out_path=r2_fastq_out_path, tmp_dir=tmp_dir)

    elif technology in {"10XV3_ULTIMA"}:
        run_fastp_10xv3_ultima(r1_fastq_path=r1_fastq_path, out_dir=out_dir, cut_tail=cut_tail, cut_window_size=cut_window_size, cut_mean_quality=cut_mean_quality, disable_adapter_trimming=disable_adapter_trimming, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, average_qual=average_qual, n_base_limit=n_base_limit, disable_quality_filtering=disable_quality_filtering, length_required=length_required, disable_length_filtering=disable_length_filtering, threads=threads, failed_out=failed_out, r1_fastq_out_path=r1_fastq_out_path)

    elif technology in {"10XV1", "INDROPSV3"}:
        run_fastp_10xv1(r1_fastq_path=r1_fastq_path, r2_fastq_path=r2_fastq_path, i1_fastq_path=i1_fastq_path, out_dir=out_dir, cut_front=cut_front, cut_tail=cut_tail, cut_window_size=cut_window_size, cut_mean_quality=cut_mean_quality, disable_adapter_trimming=disable_adapter_trimming, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, average_qual=average_qual, n_base_limit=n_base_limit, disable_quality_filtering=disable_quality_filtering, length_required=length_required, disable_length_filtering=disable_length_filtering, threads=threads, failed_out=failed_out, r1_fastq_out_path=r1_fastq_out_path, r2_fastq_out_path=r2_fastq_out_path, i1_fastq_out_path=i1_fastq_out_path)

    else:
        raise ValueError(f"Technology {technology} not recognized. See all valid values with `kb --list`.")
    
    # delete tmp_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    return {"R1": r1_fastq_out_path, "R2": r2_fastq_out_path, "I1": i1_fastq_out_path, "I2": i2_fastq_path}

