import gzip
import os

import anndata as ad
import numpy as np
import pandas as pd

# from varseek.utils.seq_utils import generate_noisy_quality_scores


# def read_fasta(file_path, semicolon_split=False):
#     with open(file_path, "r") as file:
#         header = None
#         sequence_lines = []
#         for line in file:
#             line = line.strip()
#             if line.startswith(">"):
#                 if header is not None:
#                     # Yield the previous entry
#                     sequence = "".join(sequence_lines)
#                     if semicolon_split:
#                         for sub_header in header.split(";"):
#                             yield sub_header, sequence
#                     else:
#                         yield header, sequence
#                 # Start a new record
#                 header = line[1:]  # Remove '>' character
#                 sequence_lines = []
#             else:
#                 sequence_lines.append(line)
#         # Yield the last entry after the loop ends
#         if header is not None:
#             sequence = "".join(sequence_lines)
#             if semicolon_split:
#                 for sub_header in header.split(";"):
#                     yield sub_header, sequence
#             else:
#                 yield header, sequence


# def get_header_set_from_fasta(synthetic_read_fa):
#     return {header for header, _ in read_fasta(synthetic_read_fa)}


# def create_mutant_t2g(mutation_reference_file_fasta, out="./cancer_mutant_reference_t2g.txt"):
#     if not os.path.exists(out):
#         with open(mutation_reference_file_fasta, "r") as fasta, open(out, "w") as t2g:
#             for line in fasta:
#                 if line.startswith(">"):
#                     header = line[1:].strip()
#                     t2g.write(f"{header}\t{header}\n")
#     else:
#         print(f"{out} already exists")


# def process_sam_file(sam_file):
#     with open(sam_file, "r") as sam:
#         for line in sam:
#             if line.startswith("@"):
#                 continue

#             fields = line.split("\t")
#             yield fields


# def fasta_to_fastq(
#     fasta_file,
#     fastq_file,
#     quality_score="I",
#     k=None,
#     add_noise=False,
#     average_quality_for_noisy_reads=30,
# ):
#     """
#     Convert a FASTA file to a FASTQ file with a default quality score

#     :param fasta_file: Path to the input FASTA file.
#     :param fastq_file: Path to the output FASTQ file.
#     :param quality_score: Default quality score to use for each base. Default is "I" (high quality).
#     """
#     with open(fastq_file, "w") as fastq:
#         for sequence_id, sequence in read_fasta(fasta_file):
#             if k is None or k >= len(sequence):
#                 if add_noise:
#                     quality_scores = generate_noisy_quality_scores(sequence, average_quality_for_noisy_reads)
#                 else:
#                     quality_scores = quality_score * len(sequence)
#                 fastq.write(f"@{sequence_id}\n")
#                 fastq.write(f"{sequence}\n")
#                 fastq.write("+\n")
#                 fastq.write(f"{quality_scores}\n")
#             else:
#                 for i in range(len(sequence) - k + 1):
#                     kmer = sequence[i : i + k]
#                     if add_noise:
#                         quality_scores = generate_noisy_quality_scores(kmer, average_quality_for_noisy_reads)
#                     else:
#                         quality_scores = quality_score * k

#                     fastq.write(f"@{sequence_id}_{i}\n")
#                     fastq.write(f"{kmer}\n")
#                     fastq.write("+\n")
#                     fastq.write(f"{quality_scores}\n")


# def read_fastq(fastq_file):
#     if fastq_file.endswith(".gz"):
#         file = gzip.open(fastq_file, "rt")  # 'rt' mode is for reading text
#     else:
#         file = open(fastq_file, "r")  # 'r' mode is for reading text

#     with file:
#         while True:
#             header = file.readline().strip()
#             sequence = file.readline().strip()
#             plus_line = file.readline().strip()
#             quality = file.readline().strip()

#             if not header:
#                 break

#             yield header, sequence, plus_line, quality


# def write_temp_fa(mcrs_fa):
#     mcrs_fa_original = mcrs_fa
#     mcrs_fa = mcrs_fa.replace(".fa", "_temp.fa")
#     with open(mcrs_fa, "w") as outfile:
#         for header, sequence in read_fasta(mcrs_fa_original):
#             new_header = header.replace(">", "x")
#             outfile.write(f">{new_header}\n{sequence}\n")

#     return mcrs_fa, mcrs_fa_original
