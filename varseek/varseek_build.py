"""varseek build and specific helper functions."""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal, Optional, Union

import gget
import numpy as np
import pandas as pd
import pyfastx
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator
from tqdm import tqdm

from .constants import (
    complement,
    downloadable_references,
    fasta_extensions,
    mutation_pattern,
    prebuilt_vk_ref_files,
    supported_databases_and_corresponding_reference_sequence_type,
    varseek_ref_only_allowable_kb_ref_arguments,
)
from .utils import (
    add_variant_type,
    convert_chromosome_value_to_int_when_possible,
    convert_mutation_cds_locations_to_cdna,
    create_identity_t2g,
    download_varseek_files,
    get_last_vcrs_number,
    add_variant_type_column_to_vcf_derived_df,
    add_variant_column_to_vcf_derived_df,
    generate_unique_ids,
    update_vcf_derived_df_with_multibase_duplication,
    make_function_parameter_to_value_dict,
    get_varseek_dry_run,
    report_time_elapsed,
    reverse_complement,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    translate_sequence,
    vcf_to_dataframe,
    wt_fragment_and_mutant_fragment_share_kmer,
    merge_fasta_file_headers,
    fasta_looks_like_cdna,
    run_pseudoalign_on_vcrs_df,
    longest_homopolymer,
    triplet_stats,
    download_cosmic_sequences,
    download_cosmic_mutations,
    merge_gtf_transcript_locations_into_cosmic_csv,
    set_varseek_logging_level_and_filehandler,
    count_chunks,
    determine_write_mode,
    validate_call,
    vk_config,
    FastaFile,
    CsvFile,
    IndexFile,
    T2GFile,
    TxtFile,
    ExistingFasta,
    PositiveInt,
    NonNegativeInt,
    OddInt3To63,
    Ratio,
    GtfArg,
    AlignmentReferenceType,
    LoggingLevel,
)
from .utils.varseek_build_utils import (
    create_vcrs_index,
    resolve_reference_dna_and_gtf,
    resolve_dlist,
    print_valid_values_for_variants_and_sequences_in_varseek_build,
    get_sequence_length,
    get_nucleotide_at_position,
    remove_gt_after_semicolon,
    extract_sequence,
    common_prefix_length,
    common_suffix_length,
    count_repeat_right_flank,
    count_repeat_left_flank,
    beginning_mut_nucleotides_with_right_flank,
    end_mut_nucleotides_with_left_flank,
    calculate_beginning_mutation_overlap_with_right_flank,
    calculate_end_mutation_overlap_with_left_flank,
    iterate_through_vcf_in_chunks,
    merge_subsequence_vcrss,
)

tqdm.pandas()
logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)

# Define global variables to count occurences of weird mutations
intronic_mutations = 0
posttranslational_region_mutations = 0
unknown_mutations = 0
uncertain_mutations = 0
ambiguous_position_mutations = 0
variants_incorrect_wt_base = 0
mut_idx_outside_seq = 0


accepted_build_file_types = (".csv", ".tsv", ".vcf", ".parquet")

# Advanced `build` parameters: still accepted on the command line (and in the Python signature,
# where @validate_call validates them), but hidden from `vk build --help` to keep it uncluttered.
# Consumed by main.py, which flips each matching argparse action's help to SUPPRESS. Matched by
# dest, so the --disable_* flags map to their underlying parameter name. To un-hide one, remove it.
vk_build_hidden_from_help = {
    "insertion_size_limit",
    "min_seq_len",
    "optimize_flanking_regions",
    "remove_seqs_with_wt_kmers",
    "required_insertion_overlap_length",
    "merge_identical",
    "merge_subsequences",
    "vcrs_strandedness",
    "use_IDs",
    "cosmic_version",
    "cosmic_grch",
    "cosmic_email",
    "cosmic_password",
}


class BuildParams(BaseModel):
    """Cross-field and kwargs-only validation for :func:`build`.

    Per-parameter type/extension checks live on the ``build`` signature (validated
    by ``@validate_call``). This model captures what those annotations cannot: the
    relationships between ``sequences``/``variants``, the ``w``/``k`` constraints,
    the polymorphic ``gtf`` argument, and the kwargs-only parameters that never
    appear in the typed signature. Instantiate it with the full params dict
    (extra keys are ignored)::

        BuildParams(**make_function_parameter_to_value_dict(1))
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # read for cross-field logic
    sequences: object = None
    variants: object = None
    w: object = None
    k: object = None

    # Every per-parameter check (including the value-semantics of gtf, dlist,
    # required_insertion_overlap_length, min_triplet_complexity, and
    # alignment_to_reference_type) is now expressed as a type annotation on the
    # `build` signature and validated by @validate_call. Only genuinely cross-field
    # constraints — ones a single-value annotation cannot see — remain below.
    # sequences/variants stay here because they validate *each other* (matching list
    # lengths, and the supported-genome ⇄ supported-database coupling).

    @model_validator(mode="after")
    def _validate(self):
        sequences = self.sequences
        mutations = self.variants  # apologies for the naming confusion

        # sequences
        if not isinstance(sequences, (list, tuple, str, Path)):
            raise ValueError(f"sequences must be a nucleotide string, a list of nucleotide strings, a path to a reference genome, or a string specifying a reference genome supported by varseek. Got {type(sequences)}\nTo see a list of internally supported variant databases and reference genomes, please use the 'list_internally_supported_indices' flag/argument.")
        if isinstance(sequences, (list, tuple)):
            if not all(isinstance(seq, str) for seq in sequences):
                raise ValueError("All elements in sequences must be nucleotide strings.")
            if not isinstance(mutations, (list, tuple)):
                raise ValueError("If sequences is a list, then variants must also be a list.")
            if len(sequences) != len(mutations):
                raise ValueError("If sequences is a list, then the number of elements in sequences must be equal to the number of elements in variants.")
        if isinstance(sequences, str):
            if all(c in "ACGTNU-.*" for c in sequences.upper()):  # a single reference sequence
                pass
            elif os.path.isfile(sequences) and sequences.endswith(fasta_extensions):  # a path to a reference genome with a valid extension
                pass
            elif isinstance(mutations, str) and supported_databases_and_corresponding_reference_sequence_type.get(mutations, {}).get("sequence_file_names", {}).get(sequences, None):  # a supported reference genome
                pass
            else:
                raise ValueError(f"sequences must be a nucleotide string, a list of nucleotide strings, a path to a reference genome, or a string specifying a reference genome supported by varseek. Got {sequences} of type {type(sequences)}.\nTo see a list of internally supported variant databases and reference genomes, please use the 'list_internally_supported_indices' flag/argument.")

        # mutations
        if not isinstance(mutations, (list, tuple, str, Path, pd.DataFrame)):
            raise ValueError(f"variants must be a string, a list of strings, a path to a variant database, or a string specifying a variant database supported by varseek. Got {mutations} of type {type(mutations)}\nTo see a list of internally supported variant databases and reference genomes, please use the 'list_internally_supported_indices' flag/argument.")
        if isinstance(mutations, list) and not all((isinstance(mut, str) and mut.startswith(("c.", "g."))) for mut in mutations):
            raise ValueError("All elements in variants must be strings that start with 'c.' or 'g.'.")
        if isinstance(mutations, str):
            if mutations.startswith("c.") or mutations.startswith("g."):  # a single mutation
                pass
            elif mutations in supported_databases_and_corresponding_reference_sequence_type:  # a supported mutation database
                if sequences not in supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"]:
                    raise ValueError(f"sequences {sequences} not internally supported.\nTo see a list of internally supported variant databases and reference genomes, please use the 'list_internally_supported_indices' flag/argument.")
            elif os.path.isfile(mutations) and any(x in os.path.basename(mutations) for x in accepted_build_file_types):  # a path to a mutation database with a valid extension (I avoid using 'endswith' becasue I want to check for compressed versions too - I can handle compressed versions, as pandas reads in CSVs/TSVs and pysam reads in VCFs)
                pass
            else:
                raise ValueError(f"variants must be a string, a list of strings, a path to a variant database, or a string specifying a variant database supported by varseek. Got {type(mutations)}.\nTo see a list of internally supported variant databases and reference genomes, please use the 'list_internally_supported_indices' flag/argument.")

        # w / k relationship (per-parameter validity — k odd in [3, 63] — is enforced on the signature by @validate_call)
        if self.w is not None and self.k is not None:
            w, k = int(self.w), int(self.k)
            if w >= k:
                raise ValueError(f"w should be less than k. Got w={w}, k={k}.")
            if k > 2 * w:
                raise ValueError("k must be less than or equal to 2*w")

        return self

@report_time_elapsed
@validate_call(config=vk_config)
def build(
    variants: object,
    sequences: object,
    w: PositiveInt = 47,
    k: OddInt3To63 = 51,
    max_ambiguous: NonNegativeInt = 0,
    var_column: str = "mutation",
    seq_id_column: str = "seq_ID",
    var_id_column: Optional[str] = None,
    gtf: GtfArg = None,
    gtf_transcript_id_column: Optional[str] = None,
    transcript_boundaries: bool = False,
    out: Union[str, Path] = ".",
    reference_out_dir: Optional[Union[str, Path]] = None,
    vcrs_unfiltered_fasta_out: Optional[FastaFile] = None,
    vcrs_fasta_out: Optional[FastaFile] = None,
    variants_updated_csv_out: Optional[CsvFile] = None,
    id_to_header_csv_out: Optional[CsvFile] = None,
    vcrs_t2g_out: Optional[T2GFile] = None,
    index_out: Optional[IndexFile] = None,
    removed_variants_text_out: Optional[TxtFile] = None,
    filtering_report_text_out: Optional[TxtFile] = None,
    return_variant_output: bool = False,
    save_variants_updated_csv: bool = False,
    store_full_sequences: bool = False,
    translate: bool = False,
    translate_start: Optional[Union[int, str]] = None,
    translate_end: Optional[Union[int, str]] = None,
    download: bool = False,
    chunksize: Optional[int] = None,
    dry_run: bool = False,
    list_internally_supported_indices: bool = False,
    list_downloadable_references: bool = False,
    dlist: Optional[Union[Literal["None", "intergenic_dna", "cdna"], ExistingFasta]] = None,
    overwrite: bool = False,
    threads: PositiveInt = 2,
    logging_level: Optional[Union[str, int]] = None,
    save_logs: bool = False,
    log_out_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    dont_create_index: bool = False,
    insertion_size_limit: Optional[PositiveInt] = None,
    min_seq_len: Optional[Union[PositiveInt, Literal["k"]]] = "k",  # "k" sentinel = "use the k value" (default); None = no length filter; int = that minimum
    optimize_flanking_regions: bool = True,
    remove_seqs_with_wt_kmers: bool = True,
    max_homopolymer_length: Optional[PositiveInt] = None,
    min_triplet_complexity: Optional[Ratio] = None,
    remove_alignment_to_reference: bool = True,
    alignment_to_reference_type: AlignmentReferenceType = "genome_or_transcriptome",
    species: Optional[str] = None,
    alignment_to_reference_dna: Optional[str] = None,
    alignment_to_reference_gtf: Optional[str] = None,
    required_insertion_overlap_length: Optional[Union[PositiveInt, Literal["all"]]] = None,
    merge_identical: bool = True,
    merge_subsequences: bool = True,
    vcrs_strandedness: bool = False,
    use_IDs: bool = True,
    original_order: bool = True,
    cdna_derived_vcf: bool = False,
    save_column_names_json_path: Optional[str] = None,
    cosmic_version: str = "101",  #! if I change this value, make sure to change vk clean's VCF df download accordingly
    cosmic_grch: Optional[str] = None,
    cosmic_email: Optional[str] = None,
    cosmic_password: Optional[str] = None,
    **kwargs,
):
    """
    Takes in nucleotide sequences and variants (in standard mutation/variant annotation - see below)
    and returns sequences containing the variants and the surrounding local context, dubbed variant-containing reference sequences (VCRSs),
    compatible with k-mer-based methods (i.e., kallisto | bustools) for variant detection.

    # Required input argument:
    - variants                          str or list[str] or DataFrame object) Variants to apply to the sequences. Input formats options include the following:
                                        1) Single variant (str), along with a single sequence for `sequences` (str). E.g., variants='c.2G>T' and sequences='AGCTAGCT'.
                                        2) List of variants (list[str]), along with a list of sequences for `sequences` (list[str]). E.g., variants=['c.2G>T', 'c.1A>C'] and sequences=['AGCTAGCT', 'AGCTAGCT'].
                                        NOTE: The number of variants must equal the number of input sequences.
                                        3) Path to CSV/TSV file (str) (e.g., 'variants.csv') or DataFrame (DataFrame object), along with a fasta file for `sequences`.
                                        NOTE: The `sequences` reference genome assembly (e.g., GRCh37 vs. GRCh38), source (e.g., genome vs. cDNA vs. CDS), and release (if source is cDNA or CDS, e.g., Ensembl release 111) must match the source used to annotate the variants.
                                        The CSV/TSV/DataFrame must be structured in the following way:

                                        | var_column         | seq_id_column | var_id_column |
                                        | c.2C>T             | seq1          | var1          | -> Apply varation 1 to sequence 1
                                        | c.9_13inv          | seq2          | var2          | -> Apply varation 2 to sequence 2
                                        | c.9_13inv          | seq3          | var2          | -> Apply varation 2 to sequence 3
                                        | c.9_13delinsAAT    | seq3          | var3          | -> Apply varation 3 to sequence 3
                                        | ...                | ...           | ...           |

                                        'var_column' = Column containing the variants to be performed written in standard mutation/variant annotation matching HGVS variant format (see below).
                                        'seq_id_column' = Column containing the identifiers of the sequences to be mutated (must correspond to the string following the > character in the 'sequences' fasta file; do NOT include spaces or dots).
                                        'var_id_column' = Column containing an identifier for each variant (optional).

                                        For more information on the standard mutation/variant annotation, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/.

                                        4) Path to VCF file (str) (e.g., 'variants.vcf'), along with a fasta file for `sequences`.
                                        NOTE: The `sequences` reference genome assembly (e.g., GRCh37 vs. GRCh38) and release (if source is cDNA or CDS, e.g., Ensembl release 111) must match the source used to annotate the variants.
                                        NOTE: For VCF input, the reference source is always the genome (i.e., never the cDNA or CDS). The arguments `var_column` and `seq_id_column` are not needed for VCF input (will be automatically set).
                                        The `var_id_column` ID column can be provided if wanting to use the value from the ID column in the VCF as the variant ID instead of the default HGVS ID.
                                        5) A value supported internally by vk ref (str), along with a value internally supported by vk ref corresponding to this variants value (str). See vk ref --list_internally_supported_indices for more information.

    - sequences                         (str) Sequences to which to apply the variants from `variants`. See the 'variants' argument for more information on the input formats for `sequences` and their corresponding `variants` formats.
                                        NOTE: Only the letters until the first space or dot will be used as sequence identifiers
                                        NOTE: When 'sequences' input is a genome, also see the arguments `gtf`, `gtf_transcript_id_column`, and `transcript_boundaries` in varseek build.

    # Parameters affecting VCRS creation
    - w                                  (int) Length of sequence windows flanking the variant. Default: 47.
                                         If w > total length of the sequence, the entire sequence will be kept.
    - k                                  (int) Length of the k-mers to be considered in remove_seqs_with_wt_kmers, and the default minimum value for the minimum sequence length (which can be changed with 'min_seq_len').
                                         If using kallisto in a later workflow, then this should correspond to kallisto k.
                                         Must be greater than the value passed in for w. Default: 51.
    - max_ambiguous                      (int) Maximum number of 'N' (or 'n') characters allowed in a VCRS. None means no 'N' filter will be applied. Default: 0.

    # Additional input files and associated parameters
    - var_column                         (str) Name of the column containing the variants to be introduced in 'variants'. Important for CSV/TSV/DataFrame input with pre-defined columns. Default: 'mutation'.
    - seq_id_column                      (str) Name of the column containing the IDs of the sequences to be mutated in 'variants'. Important for CSV/TSV/DataFrame input with pre-defined columns. Default: 'seq_ID'.
    - var_id_column                      (str) Name of the column containing the IDs of each variant in 'variants'. Optional. Default: use <seq_id_column>_<var_column> for each row.
    - gtf                                (str) Path to .gtf file. Only used in conjunction with the argument `transcript_boundaries`, as well as to add some information to the downloaded database when variants='cosmic_cmc'. If downloading sequence information, then setting gtf=True will automatically include it in the download. Default: None
    - gtf_transcript_id_column           (str) Column name in the input 'variants' file containing the transcript ID.
                                         In this case, column seq_id_column should contain the chromosome number.
                                         Required when 'gtf' is provided. Default: None
    - transcript_boundaries              (True/False) Whether to use the transcript boundaries in the input 'gtf' file to define the boundaries of the VCRSs. Only used when the `sequences` and `variants` information is in terms of the genome, and when `gtf` is specified. Default: False.

    # Output paths and associated parameters
    - out                                (str) Path to default output directory to containing created files. Any individual output file path can be overriden if the specific file path is provided
                                         as an argument. Default: "." (current directory).
    - reference_out_dir                  (str) Path to reference file directory to be downloaded if 'variants' is a supported database and the file corresponding to 'sequences' does not exist.
                                         Default: <out>/reference.
    - vcrs_unfiltered_fasta_out          (str) Path to output fasta file containing the unfiltered, unmerged VCRSs, written before the quality filters and before merging identical VCRSs.
                                         Headers are the per-variant `<seq_ID>:<variant>` (or var_id_column) values. Default: "<out>/vcrs_unfiltered.fa"
    - vcrs_fasta_out                     (str) Path to output fasta file containing the filtered VCRSs. This is the file the (kallisto) index and t2g are built from.
                                         If use_IDs=False, then the fasta headers will be the variant IDs (semicolon-joined if merge_identical=True).
                                         Otherwise, if use_IDs=True (default), then the fasta headers will be of the form 'vcrs_<int>' where <int> is a unique integer. Default: "<out>/vcrs.fa"
    - variants_updated_csv_out          (str) Path to output csv file containing the updated DataFrame. Only valid if save_variants_updated_csv=True. Default: "<out>/variants_updated.csv"
    - id_to_header_csv_out               (str) File name of csv file containing the mapping of unique IDs to the original sequence headers if use_IDs=True. Default: "<out>/id_to_header_mapping.csv"
    - vcrs_t2g_out                       (str) Path to output t2g file containing the transcript-to-gene mapping for the VCRSs. Used in kallisto | bustools workflow. Default: "<out>/vcrs_t2g.txt"
    - index_out                          (str) Path to output VCRS (kallisto) index file created via kb ref. Only used when the index is created (see `dont_create_index`). Default: "<out>/vcrs_index.idx".
    - fasta_out                          (str) Alias for `vcrs_fasta_out` (kept for backwards compatibility with vk ref; passed as a keyword argument). If provided, takes precedence over `vcrs_fasta_out`. Default: None.
    - t2g_out                            (str) Alias for `vcrs_t2g_out` (kept for backwards compatibility with vk ref; passed as a keyword argument). If provided, takes precedence over `vcrs_t2g_out`. Default: None.
    - removed_variants_text_out          (str) Path to output text file containing the removed variants. Default: "<out>/removed_variants.txt"
    - filtering_report_text_out          (str) Path to output text file containing the filtering report. Default: "<out>/filtering_report.txt"

    # Returning and saving of optional output
    - return_variant_output             (True/False) Whether to return the variant output saved in the fasta file. Default: False.
    - save_variants_updated_csv         (True/False) Whether to update the input 'variants' DataFrame to include additional columns with the variant type,
                                         wildtype nucleotide sequence, and mutant nucleotide sequence (only valid if 'variants' is a csv or tsv file). Default: False
    - store_full_sequences               (True/False) Whether to also include the complete wildtype and mutant sequences in the updated 'variants' DataFrame (not just the sub-sequence with
                                         w-length flanks). Only valid if save_variants_updated_csv=True. Default: False
    - translate                          (True/False) Add additional columns to the 'variants' DataFrame containing the wildtype and mutant amino acid sequences.
                                         Only valid if store_full_sequences=True. Default: False
    - translate_start                    (int | str | None) The position in the input nucleotide sequence to start translating. If a string is provided, it should correspond
                                         to a column name in 'variants' containing the open reading frame start positions for each sequence/variant.
                                         Only valid if translate=True. Default: None (translate from the beginning of the sequence)
    - translate_end                      (int | str | None) The position in the input nucleotide sequence to end translating. If a string is provided, it should correspond
                                         to a column name in 'variants' containing the open reading frame end positions for each sequence/variant.
                                         Only valid if translate=True. Default: None (translate from to the end of the sequence)

    # General arguments:
    - download                           (True/False) If True, download prebuilt reference files (index, t2g, vcrs fasta) into `out` (or the paths specified by index_out, t2g_out, and fasta_out, respectively) instead of building them. Default: False.
    - chunksize                          (int) Number of variants to process at a time. If None, then all variants will be processed at once. Default: None.
    - dry_run                            (True/False) Whether to simulate the function call without executing it. Default: False.
    - list_internally_supported_indices           (True/False) Whether to print the supported databases and sequences. Default: False.
    - list_downloadable_references       (True/False) If True, list the available downloadable references and exit. Default: False.
    - overwrite                          (True/False) Whether to overwrite existing output files. Will return if any output file already exists. Default: False.
    - threads                            (int) Number of threads to use when creating the index via kb ref. Default: 2.
    - logging_level                      (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                          (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                        (str) Directory to save logs. Default: `out`/logs
    - verbose                            (True/False) Whether to print additional information e.g., progress bars. Does not affect logging. Default: False.

    # # Advanced parameters (real `build` arguments, but hidden from the `vk build --help` CLI) - for niche use cases, specific databases, or debugging:
    # # niche use cases
    - dont_create_index                  (True/False) If True, skip the kb ref index-creation step and only produce the VCRS fasta/t2g (i.e., the classic vk build behavior). By default the index is created. Default: False.
    - insertion_size_limit               (int) Maximum number of nucleotides allowed in an insertion-type variant. Variants with insertions larger than this will be dropped.
                                         Default: None (no insertion size limit will be applied)
    - min_seq_len                        (int) Minimum length of the variant output sequence. Mutant sequences smaller than this will be dropped. None means no length filter will be applied. Default: k (from the "k" parameter)
    - optimize_flanking_regions          (True/False) Whether to remove nucleotides from either end of the mutant sequence to ensure (when possible)
                                         that the mutant sequence does not contain any (w+1)-mers (where a (w+1)-mer is a subsequence of length w+1, with w defined by the 'w' argument) also found in the wildtype/input sequence. Default: True
    - remove_seqs_with_wt_kmers          (True/False) Removes output sequences where at least one k-mer is also present in the wildtype/input sequence in the same region.
                                         If optimize_flanking_regions=True, only sequences for which a wildtype k-mer (with k defined by the 'k' argument) is still present after optimization will be removed.
                                         Default: True
    - max_homopolymer_length             (int) Drop any VCRS whose longest single-nucleotide homopolymer run is longer than this value. None disables the filter. Default: None.
    - min_triplet_complexity             (float) Drop any VCRS whose triplet complexity (distinct 3-mers / total 3-mers, in (0, 1]) is below this value. None disables the filter. Default: None.
    - remove_alignment_to_reference      (True/False) Remove VCRSs that pseudoalign to a normal reference (they would be false positives). Requires a reference DNA fasta (see `alignment_to_reference_dna`/`species`). Default: True.
    - alignment_to_reference_type        (str) Which normal reference to pseudoalign against when remove_alignment_to_reference=True: one of 'genome', 'cdna', 'transcriptome', 'genome_or_transcriptome'. Default: 'genome_or_transcriptome'.
    - species                            (str) If a supported species (e.g. 'human'), download a prebuilt reference index for the pseudoalignment filter instead of building one. Default: None.
    - alignment_to_reference_dna         (str) Path to the reference DNA (genome) fasta used to build the pseudoalignment index. Required when remove_alignment_to_reference=True and no downloadable `species` is given (falls back to `sequences` if it looks like a genome fasta). Default: None.
    - alignment_to_reference_gtf         (str) Path to the reference GTF used to build the pseudoalignment index for cdna/transcriptome/genome_or_transcriptome types. Falls back to the build `gtf` if not given. Default: None.
    - dlist                              (str) The d-list (distinguishing list) passed to `kb ref` when creating the VCRS index, used to mask k-mers shared with a normal reference. One of:
                                         None (no d-list; default), 'intergenic_dna' (build a d-list fasta of intergenic genomic regions from the reference DNA + GTF via make_intergenic_fasta),
                                         'cdna' (build a spliced-transcript/cDNA d-list fasta from the reference DNA + GTF via make_transcriptome_fasta), or a path to an existing fasta file to use directly.
                                         For 'intergenic_dna'/'cdna', the reference DNA fasta and GTF are taken from `alignment_to_reference_dna`/`alignment_to_reference_gtf`, falling back to the build `sequences`/`gtf`. Default: None.

    - required_insertion_overlap_length  (int | str | None) Enforces the minimum number of bases included in the inserted region for all (w+1)-mers (where a (w+1)-mer is a subsequence of length w+1, with w defined by the 'w' argument),
                                         or that all (w+1)-mers contain the entire inserted sequence (whatever is smaller). Only effective when optimize_flanking_regions is also True. None or 1 (minimum value) means that flank optimization occurs only until there is no shared k-mer in the VCRS and the reference sequence (i.e., as little as 1 base from the insertion could be required). If "all", then require the entire insertion and the following nucleotide (and filter out insertions of length >= 2*w). Experimental - does not work quite properly with values > 1 when there is overlap between the mutated regions and flanks. Default: None
    - merge_identical                    (True/False) Whether to merge sequence-identical VCRSs in the output (identical VCRSs will be merged by concatenating the sequence
                                         headers for all identical sequences with semicolons). Default: True
    - merge_subsequences                 (True/False) Whether to also merge VCRSs whose sequence is a subsequence (substring) of another VCRS's sequence (also considering
                                         reverse complements unless vcrs_strandedness=True). The shorter VCRS is folded into the longer (supersequence) VCRS: the longer
                                         sequence is kept and the shorter VCRS's header is concatenated with a semicolon. Only effective when merge_identical is also True. Default: True
    - vcrs_strandedness                  (True/False) Whether to consider the forward and reverse-complement mutant sequences as distinct if merging identical sequences. Only effective when merge_identical is also True. Default: False (ie consider forward and reverse-complement sequences to be equivalent).
    - use_IDs                            (True/False) Whether to keep the original sequence headers in the output fasta file, or to replace them with unique IDs of the form 'vcrs_<int>.
                                         If False, then an additional file at the path <id_to_header_csv_out> will be formed that maps sequence IDs from the fasta file to the <var_id_column>. Default: True.
    - original_order                     (True/False) Whether to keep the original order of the sequences in the output fasta file. Default: True.
    - cdna_derived_vcf                   (True/False) Whether the input VCF variants were derived from cDNA sequences.

    # # specific databases
    - cosmic_version                     (str) COSMIC release version to download. Default: "101".
    - cosmic_grch                        (str) COSMIC genome reference version to download. Default: None (choose the largest value from all internally supported values).
    - cosmic_email                       (str) Email address for COSMIC download. Default: None.
    - cosmic_password                    (str) Password for COSMIC download. Default: None.

    # other
    - save_column_names_json_path        (str) Whether to save the column names in their own json file. Utilized internally by vk ref. Default: None.


    Saves mutated sequences in fasta format (or returns a list containing the mutated sequences if out=None).
    """

    global intronic_mutations, posttranslational_region_mutations, unknown_mutations, uncertain_mutations, ambiguous_position_mutations, variants_incorrect_wt_base, mut_idx_outside_seq

    # * 0. Informational arguments that exit early
    if list_internally_supported_indices:
        print_valid_values_for_variants_and_sequences_in_varseek_build()
        return None

    if list_downloadable_references:
        print("All varseek build arguments are defaults unless otherwise specified.\n")
        for downloadable_reference in downloadable_references:
            print(f"Description: {downloadable_reference['description']}\nDownload command: {downloadable_reference['download_command']}\n")
        return None

    # * 0.5. fasta_out / t2g_out are backwards-compatible aliases (from vk ref) for vcrs_fasta_out / vcrs_t2g_out.
    # They live in **kwargs rather than the signature; validate them here the same way @validate_call would.
    fasta_out = kwargs.pop("fasta_out", None)
    t2g_out = kwargs.pop("t2g_out", None)
    if fasta_out:
        vcrs_fasta_out = TypeAdapter(FastaFile).validate_python(fasta_out)
    if t2g_out:
        vcrs_t2g_out = TypeAdapter(T2GFile).validate_python(t2g_out)

    # * 1. logger
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    if not kwargs.get("running_within_chunk_iteration", False):
        set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)

    # * 1.5 Chunk iteration
    if chunksize and return_variant_output:
        raise ValueError("return_variant_output cannot be True when chunksize is specified. Please set return_variant_output to False.")
    if chunksize and merge_identical and save_variants_updated_csv:
        raise ValueError("both merge_identical and save_variants_updated_csv cannot be True when chunksize is specified. Please set merge_identical to False and/or save_variants_updated_csv to False.")
    if chunksize is not None and isinstance(variants, (str, Path)) and os.path.exists(variants):
        variants = str(variants)  # convert Path to string
        params_dict = make_function_parameter_to_value_dict(1)
        for key in ["variants", "chunksize"]:
            params_dict.pop(key, None)
        merge_identical = params_dict.pop("merge_identical", True)
        total_chunks, total_rows = count_chunks(variants, chunksize, return_tuple_with_total_rows=True)
        if variants.endswith(".csv") or variants.endswith(".tsv"):
            sep = "\t" if variants.endswith(".tsv") else ","
            for i, chunk in enumerate(pd.read_csv(variants, sep=sep, chunksize=chunksize)):
                chunk_number = i + 1  # start at 1
                logger.info(f"Processing chunk {chunk_number}/{total_chunks}")
                build(variants=chunk, chunksize=None, chunk_number=chunk_number, total_rows=total_rows, running_within_chunk_iteration=True, merge_identical=False, **params_dict)  # running_within_chunk_iteration here for logger setup and report_time_elapsed decorator
                if chunk_number == total_chunks:
                    # the unfiltered vcrs was appended (unmerged) per-chunk; the merge + index happen on the filtered file
                    vcrs_fasta_out = os.path.join(out, "vcrs.fa") if not vcrs_fasta_out else vcrs_fasta_out  # copy-paste from below
                    if merge_identical:
                        id_to_header_csv_out = os.path.join(out, "id_to_header_mapping.csv") if not id_to_header_csv_out else id_to_header_csv_out  # copy-paste from below
                        vcrs_t2g_out = os.path.join(out, "vcrs_t2g.txt") if not vcrs_t2g_out else vcrs_t2g_out  # copy-paste from below
                        merge_fasta_file_headers(vcrs_fasta_out, use_IDs=use_IDs, id_to_header_csv_out=id_to_header_csv_out, merge_subsequences=merge_subsequences, merge_identical_rc=not vcrs_strandedness)
                        create_identity_t2g(vcrs_fasta_out, vcrs_t2g_out, mode="w")
                    # create the index once, after all chunks have been built and merged (unless disabled)
                    if not dont_create_index:
                        dlist_value = resolve_dlist(
                            dlist,
                            alignment_to_reference_dna,
                            alignment_to_reference_gtf,
                            sequences,
                            gtf,
                            out_dir=(reference_out_dir if reference_out_dir else os.path.join(out, "reference")),
                            k=k,
                            overwrite=overwrite,
                            dry_run=dry_run,
                        )
                        create_vcrs_index(
                            vcrs_fasta_out,
                            index_out if index_out else os.path.join(out, "vcrs_index.idx"),
                            k,
                            threads=threads,
                            overwrite=overwrite,
                            dry_run=dry_run,
                            kb_ref_kwargs=kwargs,
                            dlist=dlist_value,
                        )
                    return
        elif variants.endswith(".vcf") or variants.endswith(".vcf.gz"):
            iterate_through_vcf_in_chunks(variants, params_dict, chunksize, merge_identical=merge_identical)
        else:
            raise ValueError(f"Unsupported file type for chunk iteration: {variants}")
    
    chunk_number = kwargs.get("chunk_number", 1)
    first_chunk = (chunk_number == 1)
    total_rows = kwargs.get("total_rows", None)


    # * 1.75. For the nargs="+" arguments, convert any list of length 1 to a string
    if isinstance(sequences, (list, tuple)) and len(sequences) == 1:
        sequences = sequences[0]
    if isinstance(variants, (list, tuple)) and len(variants) == 1:
        variants = variants[0]

    # * 2. Type-checking (per-parameter types enforced by @validate_call; cross-field + kwargs by BuildParams)
    params_dict = make_function_parameter_to_value_dict(1)
    if not download:  # skip when downloading, as a downloadable variants/sequences pair (e.g. geuvadis) need not be a buildable one
        BuildParams(**params_dict)

    # * 3. Dry-run
    if dry_run:
        print(get_varseek_dry_run(params_dict, function_name="build"))
        if not download and not dont_create_index:
            dlist_value = resolve_dlist(
                dlist,
                alignment_to_reference_dna,
                alignment_to_reference_gtf,
                sequences,
                gtf,
                out_dir=(reference_out_dir if reference_out_dir else os.path.join(out, "reference")),
                k=k if k else w + 1,
                overwrite=overwrite,
                dry_run=True,
            )
            create_vcrs_index(
                vcrs_fasta_out if vcrs_fasta_out else os.path.join(out, "vcrs.fa"),
                index_out if index_out else os.path.join(out, "vcrs_index.idx"),
                k if k else w + 1,
                threads=threads,
                overwrite=overwrite,
                dry_run=True,
                kb_ref_kwargs=kwargs,
                dlist=dlist_value,
            )
        return None

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_build_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_build_run_info.txt")
    save_run_info(run_info_file, params_dict=params_dict, function_name="build")

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk build are required in the varseek workflow, so this is skipped

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    if not reference_out_dir:
        reference_out_dir = os.path.join(out, "reference")

    os.makedirs(out, exist_ok=True)
    os.makedirs(reference_out_dir, exist_ok=True)

    # if someone specifies an output path, then it should be saved - could technically incorporate this logic in else statements below, but this feels cleaner
    if variants_updated_csv_out:
        save_variants_updated_csv = True
    if not vcrs_unfiltered_fasta_out:
        vcrs_unfiltered_fasta_out = os.path.join(out, "vcrs_unfiltered.fa")
    if not vcrs_fasta_out:
        vcrs_fasta_out = os.path.join(out, "vcrs.fa")
    if not variants_updated_csv_out:
        variants_updated_csv_out = os.path.join(out, "variants_updated.csv")
    if not id_to_header_csv_out:
        id_to_header_csv_out = os.path.join(out, "id_to_header_mapping.csv")
    if not vcrs_t2g_out:
        vcrs_t2g_out = os.path.join(out, "vcrs_t2g.txt")
    if not removed_variants_text_out:
        removed_variants_text_out = os.path.join(out, "removed_variants.txt")
    if not filtering_report_text_out:
        filtering_report_text_out = os.path.join(out, "filtering_report.txt")
    if not index_out:
        index_out = os.path.join(out, "vcrs_index.idx")

    # * 6.5. Download prebuilt reference files instead of building them (absorbed from vk ref)
    if download:
        prebuilt_vk_ref_files_key = f"variants={variants},sequences={sequences}"  # matches constants.py (all) and server (COSMIC only)
        if prebuilt_vk_ref_files_key not in prebuilt_vk_ref_files:
            raise ValueError(f"Invalid combination of parameters for downloading prebuilt reference files. Supported combinations are: {list(prebuilt_vk_ref_files.keys())}")
        file_dict = prebuilt_vk_ref_files[prebuilt_vk_ref_files_key]
        if not file_dict:
            raise ValueError(f"No prebuilt files found for the given arguments:\nvariants: {variants}\nsequences: {sequences}")
        if file_dict["index"] == "COSMIC":
            raise NotImplementedError("Downloading COSMIC files is not currently supported. Please select another option or build a custom index.")

        logger.info(f"Downloading reference files with variants={variants}, sequences={sequences}")
        vk_ref_output_dict = download_varseek_files(file_dict, out=out, verbose=False)
        for key, destination in (("index", index_out), ("t2g", vcrs_t2g_out), ("fasta", vcrs_fasta_out)):
            if destination and vk_ref_output_dict.get(key) and vk_ref_output_dict[key] != destination:
                os.rename(vk_ref_output_dict[key], destination)
                vk_ref_output_dict[key] = destination
        logger.info(f"Downloaded files: {vk_ref_output_dict}")
        return vk_ref_output_dict

    # make sure directories of all output files exist
    output_files = [vcrs_unfiltered_fasta_out, vcrs_fasta_out, variants_updated_csv_out, id_to_header_csv_out, vcrs_t2g_out, removed_variants_text_out, filtering_report_text_out]
    for output_file in output_files:
        if os.path.isfile(output_file) and not overwrite and first_chunk:
            raise ValueError(f"Output file '{output_file}' already exists. Set 'overwrite=True' to overwrite it.")
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # * 7. Resolve derived defaults (advanced params are now real signature args validated by @validate_call)
    if not k:
        k = w + 1

    # min_seq_len: the "k" sentinel means "use k" (the default); an explicit None means "no length filter"
    if min_seq_len == "k":
        min_seq_len = k

    # get COSMIC info
    if cosmic_email:
        logger.info(f"Using COSMIC email from arguments: {cosmic_email}")
    elif os.getenv("COSMIC_EMAIL"):
        cosmic_email = os.getenv("COSMIC_EMAIL")
        logger.info(f"Using COSMIC email from COSMIC_EMAIL environment variable: {cosmic_email}")

    if cosmic_password:
        logger.info("Using COSMIC password from arguments")
    elif os.getenv("COSMIC_PASSWORD"):
        cosmic_password = os.getenv("COSMIC_PASSWORD")
        logger.info("Using COSMIC password from COSMIC_PASSWORD environment variable")

    mutations = variants
    del variants

    # * 7.5 make sure ints are ints
    w, k = int(w), int(k)
    if max_ambiguous is not None:
        max_ambiguous = int(max_ambiguous)
    if insertion_size_limit is not None:
        insertion_size_limit = int(insertion_size_limit)
    if min_seq_len is not None:
        min_seq_len = int(min_seq_len)
    if max_homopolymer_length is not None:
        max_homopolymer_length = int(max_homopolymer_length)
    if min_triplet_complexity is not None:
        min_triplet_complexity = float(min_triplet_complexity)

    # * 8. Start the actual function
    if isinstance(mutations, Path):
        mutations = str(mutations)
    if isinstance(sequences, Path):
        sequences = str(sequences)

    merge_identical_rc = not vcrs_strandedness
    
    column_name_dict = {}
    columns_to_keep = [
        "header",
        seq_id_column,
        var_column,
        "variant_type",
        "wt_sequence",
        "vcrs_sequence",
        "nucleotide_positions",
        "start_variant_position",
        "end_variant_position",
        "actual_variant",
    ]

    if isinstance(mutations, str):
        if mutations in supported_databases_and_corresponding_reference_sequence_type and "cosmic" in mutations:
            if cosmic_version not in supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_assembly_build"]:
                logger.warning(f"cosmic_version {cosmic_version} not explicitely supported internally. Using default value for reference genome build of Ensembl release 93")
            if not cosmic_grch:
                grch_supported_values_tuple = supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_assembly_build"][cosmic_version]
                grch_supported_values_tuple = [int(grch) for grch in grch_supported_values_tuple]
                grch = str(max(grch_supported_values_tuple))
            else:
                if cosmic_grch not in supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_assembly_build"][cosmic_version]:
                    raise ValueError(f"Invalid value for cosmic_grch: {cosmic_grch} for cosmic version {cosmic_version}. Supported values are {supported_databases_and_corresponding_reference_sequence_type[mutations]['database_version_to_reference_assembly_build'][cosmic_version]}.")
                grch = cosmic_grch

    # Load input sequences and their identifiers from fasta file
    if isinstance(sequences, str) and ("." in sequences or (mutations in supported_databases_and_corresponding_reference_sequence_type and sequences in supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"])):
        if isinstance(mutations, str) and mutations in supported_databases_and_corresponding_reference_sequence_type and sequences in supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"]:
            if "cosmic" in mutations:
                sequences, gtf, gtf_transcript_id_column, genome_file, cds_file, cdna_file = download_cosmic_sequences(sequences, seq_id_column, gtf, gtf_transcript_id_column, reference_out_dir, cosmic_version, mutations, grch, logger)

        titles, seqs = [], []
        for title, seq in pyfastx.Fastx(sequences):
            titles.append(title)
            seqs.append(seq)
        # titles, seqs = read_fasta(sequences)  # when using gget.utils.read_fasta()

    # Handle input sequences passed as a list
    elif isinstance(sequences, list):
        titles = [f"seq{i+1}" for i in range(len(sequences))]
        seqs = sequences

    # Handle a single sequence passed as a string
    elif isinstance(sequences, str) and "." not in sequences:
        titles = ["seq1"]
        seqs = [sequences]

    else:
        raise ValueError(
            """
            Format of the input to the 'sequences' argument not recognized.
            'sequences' must be one of the following:
            - Path to the fasta file containing the sequences to be mutated (e.g. 'seqs.fa')
            - A list of sequences to be mutated (e.g. ['ACTGCTAGCT', 'AGCTAGCT'])
            - A single sequence to be mutated passed as a string (e.g. 'AGCTAGCT')
            """
        )

    if isinstance(mutations, str) and os.path.isfile(mutations):
        mutations_path = mutations  # will account for mutations in supported_databases_and_corresponding_reference_sequence_type once the file is defined in the conditional
    else:
        mutations_path = ""

    if isinstance(mutations, str) and mutations in supported_databases_and_corresponding_reference_sequence_type:
        # TODO: expand beyond COSMIC (utilize the variant_file_name key in supported_databases_and_corresponding_reference_sequence_type)
        if "cosmic" in mutations:
            mutations, mutations_path, seq_id_column, var_column, var_id_column, columns_to_keep = download_cosmic_mutations(gtf, gtf_transcript_id_column, reference_out_dir, cosmic_version, cosmic_email, cosmic_password, columns_to_keep, grch, mutations, sequences, cds_file, cdna_file, var_id_column, verbose)

        if save_column_names_json_path:
            # save seq_id_column, var_column, var_id_column in temp json for vk ref
            column_name_dict["seq_id_column"] = seq_id_column
            column_name_dict["var_column"] = var_column
            column_name_dict["var_id_column"] = var_id_column

            column_name_dict["gtf"] = gtf if os.path.exists(gtf) else None
            column_name_dict["reference_genome_fasta"] = genome_file if os.path.exists(genome_file) else None
            column_name_dict["reference_cds_fasta"] = cds_file if os.path.exists(cds_file) else None
            column_name_dict["reference_cdna_fasta"] = cdna_file if os.path.exists(cdna_file) else None

            with open(save_column_names_json_path, "w") as f:
                json.dump(column_name_dict, f, indent=4)

    # Read in 'mutations' if passed as filepath to comma-separated csv
    if isinstance(mutations, str) and (mutations.endswith(".csv") or mutations.endswith(".tsv") or mutations.endswith(".parquet")):
        if mutations.endswith(".csv"):
            mutations = pd.read_csv(mutations)
        elif mutations.endswith(".tsv"):
            mutations = pd.read_csv(mutations, sep="\t")
        elif mutations.endswith(".parquet"):
            mutations = pd.read_parquet(mutations)
        
        for col in mutations.columns:
            if col not in columns_to_keep:
                columns_to_keep.append(col)  # append "mutation_aa", "gene_name", "mutation_id"

    elif isinstance(mutations, str) and (mutations.endswith(".vcf") or mutations.endswith(".vcf.gz")):
        mutations = vcf_to_dataframe(mutations, additional_columns=save_variants_updated_csv, explode_alt=True, filter_empty_alt=True, verbose=verbose)  # only load in additional columns if I plan to later save this updated csv
        mutations.rename(columns={"CHROM": seq_id_column}, inplace=True)
        if var_id_column:
            # mutations.rename(columns={"ID": var_id_column}, inplace=True)  # ID is not always guaranteed to be present - thus, this would complicate things for vk clean
            logger.warning("var_id_column not supported with varseek build for VCF input. Using default var_id_column as <seq_id_column>:<var_column> for each row.")
            var_id_column = None
        add_variant_type_column_to_vcf_derived_df(mutations)
        add_variant_column_to_vcf_derived_df(mutations, var_column=var_column, cdna_derived_vcf=cdna_derived_vcf)
        if any(s.startswith("chr") for s in mutations['seq_ID'].unique()) and all(not t.startswith("chr") for t in titles):
            logger.info("Chromosome numbers in the VCF file start with 'chr', but the input sequences do not. Removing 'chr' from the chromosome numbers in the variants dataframe.")
            mutations['seq_ID'] = mutations['seq_ID'].str.replace('^chr', '', regex=True)

    # Handle mutations passed as a list
    elif isinstance(mutations, list):
        if len(mutations) > 1:
            if len(mutations) != len(seqs):
                raise ValueError("If a list is passed, the number of mutations must equal the number of input sequences.")

            temp = pd.DataFrame()
            temp[var_column] = mutations
            temp[var_id_column] = [f"var{i+1}" for i in range(len(mutations))]
            temp[seq_id_column] = [f"seq{i+1}" for i in range(len(mutations))]
            mutations = temp
        else:
            temp = pd.DataFrame()
            temp[var_column] = [mutations[0]] * len(seqs)
            temp[var_id_column] = [f"var{i+1}" for i in range(len(seqs))]
            temp[seq_id_column] = [f"seq{i+1}" for i in range(len(seqs))]
            mutations = temp

    # Handle single mutation passed as a string
    elif isinstance(mutations, str) and mutations not in supported_databases_and_corresponding_reference_sequence_type:
        # This will work for one mutation for one sequence as well as one mutation for multiple sequences
        temp = pd.DataFrame()
        mutations = [mutations]
        temp[var_column] = [mutations[0]] * len(seqs)
        temp[var_id_column] = [f"var{i+1}" for i in range(len(seqs))]
        temp[seq_id_column] = [f"seq{i+1}" for i in range(len(seqs))]
        mutations = temp

    elif isinstance(mutations, pd.DataFrame):
        mutations = mutations.copy()
        for col in mutations.columns:
            if col not in columns_to_keep:
                columns_to_keep.append(col)  # append "mutation_aa", "gene_name", "mutation_id"

    else:
        raise ValueError(
            """
            Format of the input to the 'variants' argument not recognized.
            'variants' must be one of the following:
            - Path to comma-separated csv file (e.g. 'variants.csv')
            - A pandas DataFrame object
            - A single mutation to be applied to all input sequences (e.g. 'c.2C>T')
            - A list of variants (the number of variants must equal the number of input sequences) (e.g. ['c.2C>T', 'c.1A>C'])
            """
        )

    if "c." in mutations[var_column].values[0]:
        reference_source = "transcriptome"
    elif "g." in mutations[var_column].values[0]:
        reference_source = "genome"
    else:
        reference_source = "unknown"
    
    # Set of possible nucleotides (- and . are gap annotations)
    nucleotides = set("ATGCUNatgcun.-")

    seq_dict = {}
    non_nuc_seqs = 0
    for title, seq in zip(titles, seqs):
        # Check that sequences are nucleotide sequences
        if not set(seq) <= nucleotides:
            non_nuc_seqs += 1

        # seq = seq.strip("N")  # cds position sometimes assumes no leading Ns (eg with COSMIC) - keep this off by default, but consider adding as a setting

        # Keep text following the > until the first space/dot as the sequence identifier
        # Dots are removed so Ensembl version numbers are removed
        seq_dict[title.split(" ")[0].split(".")[0]] = seq

    del titles
    del seqs

    if non_nuc_seqs > 0:
        logger.warning("Non-nucleotide characters detected in %s input sequences. vk build is only optimized for mutating nucleotide sequences.", non_nuc_seqs)

    if original_order:
        mutations["original_order"] = range(len(mutations))  # ensure that original order can be restored at the end
        columns_to_keep.append("original_order")  # just so it doesn't get removed automatically (but I remove it manually later)

    total_mutations = mutations.shape[0]

    # Drop inputs for sequences or variants that were not found
    mutations = mutations.dropna(subset=[seq_id_column, var_column])
    missing_seq_id_or_var = total_mutations - mutations.shape[0]
    total_mutations_updated = mutations.shape[0]
    if len(mutations) < 1:
        raise ValueError(
            """
            None of the input sequences match the sequence IDs provided in 'mutations'.
            Ensure that the sequence IDs correspond to the string following the > character in the 'sequences' fasta file (do NOT include spaces or dots).
            """
        )

    # remove duplicate entries from the mutations dataframe, keeping the ones with the most information
    mutations["non_na_count"] = mutations.notna().sum(axis=1)
    mutations = mutations.sort_values(by="non_na_count", ascending=False)
    mutations = mutations.drop_duplicates(subset=[seq_id_column, var_column], keep="first")
    mutations = mutations.drop(columns=["non_na_count"])

    duplicate_count = total_mutations_updated - mutations.shape[0]
    total_mutations_updated = mutations.shape[0]

    # ensure seq_ID column is string type, and chromosome numbers don't have decimals
    mutations[seq_id_column] = mutations[seq_id_column].apply(convert_chromosome_value_to_int_when_possible)

    if "variant_type" not in mutations.columns:
        add_variant_type(mutations, var_column)
    variant_types = ["substitution", "deletion", "duplication", "insertion", "inversion", "delins"]
    mutations['variant_type'] = mutations['variant_type'].astype(pd.CategoricalDtype(categories=variant_types))  # new as of 3/2025

    # Link sequences to their mutations using the sequence identifiers
    if store_full_sequences or ".vcf" in mutations_path:
        mutations["wt_sequence_full"] = mutations[seq_id_column].map(seq_dict)
        if ".vcf" in mutations_path:  # look for long duplications - needed seq_dict
            update_vcf_derived_df_with_multibase_duplication(mutations, seq_dict, seq_id_column=seq_id_column, var_column=var_column, cdna_derived_vcf=cdna_derived_vcf)
            if not store_full_sequences:
                mutations.drop(columns=["wt_sequence_full"], inplace=True)

    # Handle sequences that were not found based on their sequence IDs
    mutations[seq_id_column] = mutations[seq_id_column].str.split(".").str[0]  #$ new 2026
    seqs_not_found_count = len(mutations[~mutations[seq_id_column].isin(seq_dict.keys())])
    if seqs_not_found_count > 0:
        logger.warning(
            """
            The sequences corresponding to %d sequence IDs were not found.
            These sequences and their corresponding mutations will not be included in the output.
            Ensure that the sequence IDs correspond to the string following the > character in the 'sequences' FASTA file (do NOT include spaces or dots).
            """,
            seqs_not_found_count,
        )

        mutations = mutations[mutations[seq_id_column].isin(seq_dict.keys())]

    mutations["vcrs_sequence"] = ""

    if var_id_column is not None:
        mutations["header"] = mutations[var_id_column]
        mutations["hgvs"] = mutations[seq_id_column].astype(str) + ":" + mutations[var_column]
        logger.info("Using var_id_column '%s' as the variant header column.", var_id_column)
    else:
        mutations["header"] = mutations[seq_id_column].astype(str) + ":" + mutations[var_column]
        logger.info("Using the seq_id_column:var_column '%s' columns as the variant header column.", f"{seq_id_column}:{var_column}")

    # make a set of all initial mutation IDs
    initial_mutation_id_set = set(mutations["header"].dropna())

    # Calculate number of bad mutations
    uncertain_mutations = mutations[var_column].str.contains(r"\?").sum()  # I originally tried doing a += thing here to account for the cDNA to CDS thing, but then it is hard to track with the double counting and becomes not worth it

    ambiguous_position_mutations = mutations[var_column].str.contains(r"\(|\)").sum()

    intronic_mutations = mutations[var_column].str.contains(r"\+|\-").sum()

    posttranslational_region_mutations = mutations[var_column].str.contains(r"\*").sum()

    # Filter out bad mutations
    combined_pattern = re.compile(r"(\?|\(|\)|\+|\-|\*)")
    bad_mutations_mask = mutations[var_column].str.contains(combined_pattern)
    mutations = mutations[~bad_mutations_mask]
    del bad_mutations_mask

    # Extract nucleotide positions and mutation info from Mutation CDS
    mutations[["nucleotide_positions", "actual_variant"]] = mutations[var_column].str.extract(mutation_pattern)

    # Filter out mutations that did not match the re
    unknown_mutations = mutations["nucleotide_positions"].isna().sum()
    mutations = mutations.dropna(subset=["nucleotide_positions", "actual_variant"])

    if mutations.empty:
        logger.warning("No valid variants found in the input.")
        return [] if return_variant_output else None

    # Split nucleotide positions into start and end positions
    split_positions = mutations["nucleotide_positions"].str.split("_", expand=True)

    mutations["start_variant_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        mutations["end_variant_position"] = split_positions[1].fillna(split_positions[0])
    else:
        mutations["end_variant_position"] = mutations["start_variant_position"]

    mutations.loc[mutations["end_variant_position"].isna(), "end_variant_position"] = mutations["start_variant_position"]

    mutations[["start_variant_position", "end_variant_position"]] = mutations[["start_variant_position", "end_variant_position"]].astype(int)

    # Adjust positions to 0-based indexing
    mutations["start_variant_position"] -= 1
    mutations["end_variant_position"] -= 1  # don't forget to increment by 1 later

    # Calculate sequence length
    mutations["sequence_length"] = mutations[seq_id_column].apply(lambda x: get_sequence_length(x, seq_dict)).astype(int)  # noqa: F821

    # Filter out mutations with positions outside the sequence
    index_error_mask = (mutations["start_variant_position"] > mutations["sequence_length"]) | (mutations["end_variant_position"] > mutations["sequence_length"])

    mut_idx_outside_seq = index_error_mask.sum()

    mutations = mutations[~index_error_mask]

    if mutations.empty:
        logger.warning("No valid variants found in the input.")
        return [] if return_variant_output else None

    # Create masks for each type of mutation
    mutations["wt_nucleotides_ensembl"] = None
    substitution_mask = mutations["variant_type"] == "substitution"
    deletion_mask = mutations["variant_type"] == "deletion"
    delins_mask = mutations["variant_type"] == "delins"
    insertion_mask = mutations["variant_type"] == "insertion"
    duplication_mask = mutations["variant_type"] == "duplication"
    inversion_mask = mutations["variant_type"] == "inversion"

    if remove_seqs_with_wt_kmers:
        long_duplications = ((duplication_mask) & ((mutations["end_variant_position"] - mutations["start_variant_position"]) >= k)).sum()
        logger.info("Removing %d duplications > k", long_duplications)
        mutations = mutations[~((duplication_mask) & ((mutations["end_variant_position"] - mutations["start_variant_position"]) >= k))]
    else:
        long_duplications = 0

    # Create a mask for all non-substitution mutations
    non_substitution_mask = deletion_mask | delins_mask | insertion_mask | duplication_mask | inversion_mask
    insertion_and_delins_and_dup_and_inversion_mask = insertion_mask | delins_mask | duplication_mask | inversion_mask

    # Extract the WT nucleotides for the substitution rows from reference fasta (i.e., Ensembl)
    start_positions = mutations.loc[substitution_mask, "start_variant_position"].values

    # Get the nucleotides at the start positions
    wt_nucleotides_substitution = np.array([get_nucleotide_at_position(seq_id, pos, seq_dict) for seq_id, pos in zip(mutations.loc[substitution_mask, seq_id_column], start_positions)])

    mutations.loc[substitution_mask, "wt_nucleotides_ensembl"] = wt_nucleotides_substitution

    # Extract the WT nucleotides for the substitution rows from the Mutation CDS (i.e., COSMIC)
    mutations["wt_nucleotides_cosmic"] = None
    mutations.loc[substitution_mask, "wt_nucleotides_cosmic"] = mutations["actual_variant"].str[0]

    congruent_wt_bases_mask = (mutations["wt_nucleotides_cosmic"] == mutations["wt_nucleotides_ensembl"]) | mutations[["wt_nucleotides_cosmic", "wt_nucleotides_ensembl"]].isna().any(axis=1)

    variants_incorrect_wt_base = (~congruent_wt_bases_mask).sum()

    mutations = mutations[congruent_wt_bases_mask]
    del congruent_wt_bases_mask

    if mutations.empty:
        logger.warning("No valid variants found in the input.")
        return [] if return_variant_output else None

    # Adjust the start and end positions for insertions
    mutations.loc[insertion_mask, "start_variant_position"] += 1  # in other cases, we want left flank to exclude the start of mutation site; but with insertion, the start of mutation site as it is denoted still belongs in the flank region
    mutations.loc[insertion_mask, "end_variant_position"] -= 1  # in this notation, the end position is one before the start position

    # Extract the WT nucleotides for the non-substitution rows from the Mutation CDS (i.e., COSMIC)
    mutations.loc[non_substitution_mask, "wt_nucleotides_ensembl"] = mutations.loc[non_substitution_mask].apply(lambda row: extract_sequence(row, seq_dict, seq_id_column), axis=1)  # noqa: F821

    # Apply mutations to the sequences
    mutations["mut_nucleotides"] = None
    mutations.loc[substitution_mask, "mut_nucleotides"] = mutations.loc[substitution_mask, "actual_variant"].str[-1]
    mutations.loc[deletion_mask, "mut_nucleotides"] = ""
    mutations.loc[delins_mask, "mut_nucleotides"] = mutations.loc[delins_mask, "actual_variant"].str.extract(r"delins([A-Z]+)")[0]
    mutations.loc[insertion_mask, "mut_nucleotides"] = mutations.loc[insertion_mask, "actual_variant"].str.extract(r"ins([A-Z]+)")[0]
    mutations.loc[duplication_mask, "mut_nucleotides"] = mutations.loc[duplication_mask].apply(lambda row: row["wt_nucleotides_ensembl"], axis=1)
    if inversion_mask.any():
        mutations.loc[inversion_mask, "mut_nucleotides"] = mutations.loc[inversion_mask].apply(
            lambda row: "".join(complement.get(nucleotide, "N") for nucleotide in row["wt_nucleotides_ensembl"][::-1]),
            axis=1,
        )

    # Adjust the nucleotide positions of duplication mutations to mimic that of insertions (since duplications are essentially just insertions)
    mutations.loc[duplication_mask, "start_variant_position"] = mutations.loc[duplication_mask, "end_variant_position"] + 1  # in the case of duplication, the "mutant" site is still in the left flank as well

    mutations.loc[duplication_mask, "wt_nucleotides_ensembl"] = ""

    # Calculate the kmer bounds
    mutations["start_kmer_position_min"] = mutations["start_variant_position"] - w
    mutations["start_kmer_position"] = mutations["start_kmer_position_min"].combine(0, max)

    mutations["end_kmer_position_max"] = mutations["end_variant_position"] + w
    mutations["end_kmer_position"] = mutations[["end_kmer_position_max", "sequence_length"]].min(axis=1)  # don't forget to increment by 1 later on

    if gtf is not None and transcript_boundaries:
        if "start_transcript_position" not in mutations.columns and "end_transcript_position" not in mutations.columns:  # * currently hard-coded column names, but optionally can be changed to arguments later
            mutations = merge_gtf_transcript_locations_into_cosmic_csv(mutations, gtf, gtf_transcript_id_column=gtf_transcript_id_column, output_mutations_path=mutations_path)

            columns_to_keep.extend(["start_transcript_position", "end_transcript_position", "strand"])
        else:
            logger.warning("Transcript positions already present in the input variants file. Skipping GTF file merging.")

        # adjust start_transcript_position to be 0-index
        mutations["start_transcript_position"] -= 1

        mutations["start_kmer_position"] = mutations[["start_kmer_position", "start_transcript_position"]].max(axis=1)
        mutations["end_kmer_position"] = mutations[["end_kmer_position", "end_transcript_position"]].min(axis=1)

    mut_apply = (lambda *args, **kwargs: mutations.progress_apply(*args, **kwargs)) if verbose else mutations.apply

    if save_variants_updated_csv and store_full_sequences:
        # Extract flank sequences
        if verbose:
            tqdm.pandas(desc="Extracting full left flank sequences")

        mutations["left_flank_region_full"] = mut_apply(
            lambda row: seq_dict[row[seq_id_column]][0 : row["start_variant_position"]],  # noqa: F821
            axis=1,
        )  # ? vectorize

        if verbose:
            tqdm.pandas(desc="Extracting full right flank sequences")

        mutations["right_flank_region_full"] = mut_apply(
            lambda row: seq_dict[row[seq_id_column]][row["end_variant_position"] + 1 : row["sequence_length"]],  # noqa: F821
            axis=1,
        )  # ? vectorize

    if verbose:
        tqdm.pandas(desc="Extracting VCRS left flank sequences")

    mutations["left_flank_region"] = mut_apply(
        lambda row: seq_dict[row[seq_id_column]][row["start_kmer_position"] : row["start_variant_position"]],  # noqa: F821
        axis=1,
    )  # ? vectorize

    if verbose:
        tqdm.pandas(desc="Extracting VCRS right flank sequences")

    mutations["right_flank_region"] = mut_apply(
        lambda row: seq_dict[row[seq_id_column]][row["end_variant_position"] + 1 : row["end_kmer_position"] + 1],  # noqa: F821
        axis=1,
    )  # ? vectorize

    del seq_dict

    mutations["inserted_nucleotide_length"] = None

    number_of_mutations_greater_than_insertion_size_limit = 0
    if insertion_and_delins_and_dup_and_inversion_mask.any():
        mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "inserted_nucleotide_length"] = mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "mut_nucleotides"].str.len()

        mutations_len = len(mutations)
        if insertion_size_limit is not None:
            mutations = mutations[(mutations["inserted_nucleotide_length"].isna()) | (mutations["inserted_nucleotide_length"] <= insertion_size_limit)]  # # Keep rows where it's <= insertion_size_limit
        number_of_mutations_greater_than_insertion_size_limit = mutations_len - len(mutations)

    mutations["beginning_mutation_overlap_with_right_flank"] = 0
    mutations["end_mutation_overlap_with_left_flank"] = 0

    # Rules for shaving off kmer ends - r1 = left flank, r2 = right flank, d = deleted portion, i = inserted portion
    # Substitution: N/A
    # Deletion:
    # To what extend the beginning of d overlaps with the beginning of r2 --> shave up to that many nucleotides off the beginning of r1 until w - len(r1) ≥ extent of overlap
    # To what extend the end of d overlaps with the beginning of r1 --> shave up to that many nucleotides off the end of r2 until w - len(r2) ≥ extent of overlap
    # Insertion, Duplication:
    # To what extend the beginning of i overlaps with the beginning of r2 --> shave up to that many nucleotides off the beginning of r1 until w - len(r1) ≥ extent of overlap
    # To what extend the end of i overlaps with the beginning of r1 --> shave up to that many nucleotides off the end of r2 until w - len(r2) ≥ extent of overlap
    # Delins, inversion:
    # To what extend the beginning of i overlaps with the beginning of d --> shave up to that many nucleotides off the beginning of r1 until w - len(r1) ≥ extent of overlap
    # To what extend the end of i overlaps with the beginning of d --> shave up to that many nucleotides off the end of r2 until w - len(r2) ≥ extent of overlap
    if optimize_flanking_regions and non_substitution_mask.any():
        # Apply the function for beginning of mut_nucleotides with right_flank_region
        mutations.loc[non_substitution_mask, "beginning_mutation_overlap_with_right_flank"] = mutations.loc[non_substitution_mask].apply(calculate_beginning_mutation_overlap_with_right_flank, axis=1)

        # Apply the function for end of mut_nucleotides with left_flank_region
        mutations.loc[non_substitution_mask, "end_mutation_overlap_with_left_flank"] = mutations.loc[non_substitution_mask].apply(calculate_end_mutation_overlap_with_left_flank, axis=1)

        # for insertions and delins, make sure I see at bare minimum the full insertion context and the subseqeuent nucleotide - eg if I have c.2_3insA to become ACGTT to ACAGTT, if I only check for ACAG, then I can't distinguosh between ACAGTT, ACAGGTT, ACAGGGTT, etc. (and there are more complex examples)
        # TODO: for duplications, required_insertion_overlap_length=None works fine; but required_insertion_overlap_length="all" or some number >1 causes issues (ruins symmetry)
        if required_insertion_overlap_length and required_insertion_overlap_length != 1 and insertion_and_delins_and_dup_and_inversion_mask.any():  # * new as of 11/20/24
            if required_insertion_overlap_length == "all":
                required_insertion_overlap_length = np.inf

            if required_insertion_overlap_length >= 2 * w:
                mutations = mutations[(mutations["inserted_nucleotide_length"].isna()) | (mutations["inserted_nucleotide_length"] < 2 * w)]  # Keep rows where it is None/NaN  # Keep rows where it's < 2*w

            mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "beginning_mutation_overlap_with_right_flank"] = np.maximum(
                mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "beginning_mutation_overlap_with_right_flank"],
                np.minimum(mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "inserted_nucleotide_length"], required_insertion_overlap_length - 1),  # Feb 2025: the -1 was added empirically
            )

            mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "end_mutation_overlap_with_left_flank"] = np.maximum(
                mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "end_mutation_overlap_with_left_flank"],
                np.minimum(mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "inserted_nucleotide_length"], required_insertion_overlap_length - 1),
            )

        # Calculate w-len(flank) (see above instructions)
        mutations.loc[non_substitution_mask, "k_minus_left_flank_length"] = w - mutations.loc[non_substitution_mask, "left_flank_region"].apply(len)
        mutations.loc[non_substitution_mask, "k_minus_right_flank_length"] = w - mutations.loc[non_substitution_mask, "right_flank_region"].apply(len)

        mutations.loc[non_substitution_mask, "updated_left_flank_start"] = np.maximum(
            mutations.loc[non_substitution_mask, "beginning_mutation_overlap_with_right_flank"] - mutations.loc[non_substitution_mask, "k_minus_left_flank_length"],
            0,
        )
        mutations.loc[non_substitution_mask, "updated_right_flank_end"] = np.maximum(
            mutations.loc[non_substitution_mask, "end_mutation_overlap_with_left_flank"] - mutations.loc[non_substitution_mask, "k_minus_right_flank_length"],
            0,
        )

        mutations["updated_left_flank_start"] = mutations["updated_left_flank_start"].fillna(0).astype(int)
        mutations["updated_right_flank_end"] = mutations["updated_right_flank_end"].fillna(0).astype(int)

    else:
        mutations["updated_left_flank_start"] = 0
        mutations["updated_right_flank_end"] = 0

    # Create WT substitution w-mer sequences
    if substitution_mask.any():
        mutations.loc[substitution_mask, "wt_sequence"] = mutations.loc[substitution_mask, "left_flank_region"] + mutations.loc[substitution_mask, "wt_nucleotides_ensembl"] + mutations.loc[substitution_mask, "right_flank_region"]

    # Create WT non-substitution w-mer sequences
    if non_substitution_mask.any():
        mutations.loc[non_substitution_mask, "wt_sequence"] = mutations.loc[non_substitution_mask].apply(
            lambda row: row["left_flank_region"][row["updated_left_flank_start"] :] + row["wt_nucleotides_ensembl"] + row["right_flank_region"][: len(row["right_flank_region"]) - row["updated_right_flank_end"]],
            axis=1,
        )

    # Create mutant substitution w-mer sequences
    if substitution_mask.any():
        mutations.loc[substitution_mask, "vcrs_sequence"] = mutations.loc[substitution_mask, "left_flank_region"] + mutations.loc[substitution_mask, "mut_nucleotides"] + mutations.loc[substitution_mask, "right_flank_region"]

    # Create mutant non-substitution w-mer sequences
    if non_substitution_mask.any():
        mutations.loc[non_substitution_mask, "vcrs_sequence"] = mutations.loc[non_substitution_mask].apply(
            lambda row: row["left_flank_region"][row["updated_left_flank_start"] :] + row["mut_nucleotides"] + row["right_flank_region"][: len(row["right_flank_region"]) - row["updated_right_flank_end"]],
            axis=1,
        )

    # * Save the unfiltered VCRS fasta (before the quality filters and before merging identical VCRSs).
    # One record per surviving variant, headers = the per-variant `header` column, no t2g and no merge.
    raw_mask = mutations["vcrs_sequence"].astype(bool)
    if raw_mask.any():
        raw_fasta = ">" + mutations.loc[raw_mask, "header"].astype(str) + "\n" + mutations.loc[raw_mask, "vcrs_sequence"].astype(str) + "\n"
        with open(vcrs_unfiltered_fasta_out, determine_write_mode(vcrs_unfiltered_fasta_out, overwrite=overwrite, first_chunk=first_chunk), encoding="utf-8") as raw_fasta_file:
            raw_fasta_file.write("".join(raw_fasta.values))
        logger.info("Unfiltered (pre-filter, pre-merge) VCRS fasta written to %s.", vcrs_unfiltered_fasta_out)

    if remove_seqs_with_wt_kmers:
        if verbose:
            tqdm.pandas(desc="Removing VCRSs that share a k-mer with their respective non-variant sequence")

        mutations["wt_fragment_and_mutant_fragment_share_kmer"] = mut_apply(
            lambda row: wt_fragment_and_mutant_fragment_share_kmer(
                mutated_fragment=row["vcrs_sequence"],
                wildtype_fragment=row["wt_sequence"],
                k=k,
            ),
            axis=1,
        )

        mutations_overlapping_with_wt = mutations["wt_fragment_and_mutant_fragment_share_kmer"].sum()

        mutations = mutations[~mutations["wt_fragment_and_mutant_fragment_share_kmer"]]
    else:
        mutations_overlapping_with_wt = 0

    if save_variants_updated_csv and store_full_sequences:
        columns_to_keep.extend(["wt_sequence_full", "vcrs_sequence_full"])

        # Create full sequences (substitution and non-substitution)
        mutations["vcrs_sequence_full"] = mutations["left_flank_region_full"] + mutations["mut_nucleotides"] + mutations["right_flank_region_full"]

    if min_seq_len:
        # Calculate k-mer lengths (where k=w) and report the distribution
        mutations["vcrs_sequence_kmer_length"] = mutations["vcrs_sequence"].apply(lambda x: len(x) if pd.notna(x) else 0)

        rows_less_than_minimum = (mutations["vcrs_sequence_kmer_length"] < min_seq_len).sum()

        mutations = mutations[mutations["vcrs_sequence_kmer_length"] >= min_seq_len]

        logger.info("Removed %d variant-containing reference sequences with length less than %d...", rows_less_than_minimum, min_seq_len)
    else:
        rows_less_than_minimum = 0

    if max_ambiguous is not None:
        # Get number of 'N' or 'n' occuring in the sequence
        mutations["num_N"] = mutations["vcrs_sequence"].str.lower().str.count("n")
        num_rows_with_N = (mutations["num_N"] > max_ambiguous).sum()
        mutations = mutations[mutations["num_N"] <= max_ambiguous]
        mutations = mutations.drop(columns=["num_N"])

        logger.info("Removed %d variant-containing reference sequences containing more than %d 'N's...", num_rows_with_N, max_ambiguous)
    else:
        num_rows_with_N = 0

    # * New quality filters (homopolymer, triplet complexity, pseudoalignment-to-reference).
    # These operate on `vcrs_sequence`/`header` and persist no new columns, so they run before
    # the columns_to_keep subsetting below and require no changes to columns_to_keep.
    num_rows_homopolymer = num_rows_triplet = num_rows_pseudoaligned = 0

    if max_homopolymer_length is not None:
        homopolymer_lengths = mutations["vcrs_sequence"].apply(lambda s: longest_homopolymer(s)[0] if pd.notna(s) else 0)
        num_rows_homopolymer = int((homopolymer_lengths > max_homopolymer_length).sum())
        mutations = mutations[homopolymer_lengths <= max_homopolymer_length]
        logger.info("Removed %d variant-containing reference sequences with a homopolymer run longer than %d...", num_rows_homopolymer, max_homopolymer_length)

    if min_triplet_complexity is not None:
        triplet_complexities = mutations["vcrs_sequence"].apply(lambda s: triplet_stats(s)[2] if pd.notna(s) else 1.0)
        num_rows_triplet = int((triplet_complexities < min_triplet_complexity).sum())
        mutations = mutations[triplet_complexities >= min_triplet_complexity]
        logger.info("Removed %d variant-containing reference sequences with triplet complexity below %s...", num_rows_triplet, min_triplet_complexity)

    if remove_alignment_to_reference and not mutations.empty:
        dna_ref, gtf_ref = alignment_to_reference_dna, alignment_to_reference_gtf
        if dna_ref is None and species is None:
            if isinstance(sequences, str) and os.path.isfile(sequences) and sequences.endswith(fasta_extensions):
                # Best gtf available for the cDNA-vs-genome cross-check: the explicit
                # alignment gtf if given, else the build gtf (if it is a path on disk).
                detection_gtf = gtf_ref if gtf_ref is not None else (gtf if isinstance(gtf, str) and os.path.isfile(gtf) else None)
                if fasta_looks_like_cdna(sequences, gtf=detection_gtf):
                    if alignment_to_reference_type == "genome":
                        # `genome` uses kb ref's `custom` workflow, which indexes the provided fasta
                        # directly, so a cDNA fasta still yields a usable (if unintended) index. Warn only.
                        logger.warning(
                            "alignment_to_reference_dna not provided; falling back to `sequences` (%s), which looks like a cDNA/transcriptome fasta rather than genomic DNA. "
                            "Proceeding because alignment_to_reference_type='genome' indexes the provided fasta directly, but ensure this is the reference you intend to pseudoalign against.",
                            sequences,
                        )
                    else:
                        # cdna/transcriptome/genome_or_transcriptome build their index from a genome
                        # (DNA) fasta plus a gtf (kb ref `standard`/`nac`), so a cDNA fasta will not work.
                        raise ValueError(
                            f"alignment_to_reference_dna not provided, and `sequences` ({sequences}) looks like a cDNA/transcriptome fasta rather than genomic DNA. "
                            f"alignment_to_reference_type={alignment_to_reference_type} builds its reference index from a genome (DNA) fasta plus a gtf, so a cDNA fasta will not work. "
                            "Provide a genome fasta via alignment_to_reference_dna (or a downloadable species), or set alignment_to_reference_type='genome'."
                        )
                else:
                    logger.warning("alignment_to_reference_dna not provided; falling back to `sequences` as the reference DNA fasta. Ensure `sequences` is a genome (DNA) fasta, not cDNA.")
                dna_ref = sequences
            else:
                raise ValueError("remove_alignment_to_reference=True requires alignment_to_reference_dna, a downloadable species, or a genome fasta in `sequences`.")
        if alignment_to_reference_type != "genome" and gtf_ref is None and species is None:
            if isinstance(gtf, str) and os.path.isfile(gtf):
                logger.warning("alignment_to_reference_gtf not provided; falling back to the build `gtf`.")
                gtf_ref = gtf
            else:
                raise ValueError(f"alignment_to_reference_type={alignment_to_reference_type} requires a gtf (alignment_to_reference_gtf, build gtf, or a downloadable species).")
        len_before_pseudoalign = len(mutations)
        mutations = run_pseudoalign_on_vcrs_df(
            mutations,
            reference_type=alignment_to_reference_type,
            index_dir=reference_out_dir,
            out_dir=os.path.join(out, "pseudoalignment_tmp"),
            dna_fasta=dna_ref,
            gtf=gtf_ref,
            k=k,
            threads=threads,
            seq_col="vcrs_sequence",
            species=species,
        )
        num_rows_pseudoaligned = len_before_pseudoalign - len(mutations)
        logger.info("Removed %d variant-containing reference sequences that pseudoaligned to the %s reference...", num_rows_pseudoaligned, alignment_to_reference_type)

    # Report status of mutations back to user
    good_mutations = mutations.shape[0]
    total_removed_mutations = total_mutations - good_mutations

    report = f"""
        {good_mutations} variants correctly recorded ({good_mutations/total_mutations*100:.2f}%)
        {total_removed_mutations} variants removed ({total_removed_mutations/total_mutations*100:.2f}%)
          {missing_seq_id_or_var} variants missing seq_id or var_column ({missing_seq_id_or_var/total_mutations*100:.3f}%)
          {duplicate_count} entries removed due to having a duplicate entry ({duplicate_count/total_mutations*100:.3f}%)
          {seqs_not_found_count} variants with seq_ID not found in sequences ({seqs_not_found_count/total_mutations*100:.3f}%)
          {intronic_mutations} intronic variants found ({intronic_mutations/total_mutations*100:.3f}%)
          {posttranslational_region_mutations} posttranslational region variants found ({posttranslational_region_mutations/total_mutations*100:.3f}%)
          {unknown_mutations} unknown variants found ({unknown_mutations/total_mutations*100:.3f}%)
          {uncertain_mutations} variants with uncertain mutation found ({uncertain_mutations/total_mutations*100:.3f}%)
          {ambiguous_position_mutations} variants with ambiguous position found ({ambiguous_position_mutations/total_mutations*100:.3f}%)
          {variants_incorrect_wt_base} variants with incorrect wildtype base found ({variants_incorrect_wt_base/total_mutations*100:.3f}%)
          {mut_idx_outside_seq} variants with indices outside of the sequence length found ({mut_idx_outside_seq/total_mutations*100:.3f}%)
        """

    if remove_seqs_with_wt_kmers:
        report += f"""  {long_duplications} duplications longer than k found ({long_duplications/total_mutations*100:.3f}%)
          {mutations_overlapping_with_wt} variants with overlapping kmers found ({mutations_overlapping_with_wt/total_mutations*100:.3f}%)
        """

    if min_seq_len:
        report += f"""  {rows_less_than_minimum} variants with fragment length < min_seq_len removed ({rows_less_than_minimum/total_mutations*100:.3f}%)
        """

    if max_ambiguous is not None:
        report += f"""  {num_rows_with_N} variants with more than {max_ambiguous} Ns found ({num_rows_with_N/total_mutations*100:.3f}%)
        """

    if number_of_mutations_greater_than_insertion_size_limit > 0:
        report += f"""  {number_of_mutations_greater_than_insertion_size_limit} variants with inserted nucleotide length > insertion_size_limit removed ({number_of_mutations_greater_than_insertion_size_limit/total_mutations*100:.3f}%)
        """

    if max_homopolymer_length is not None:
        report += f"""  {num_rows_homopolymer} variants with a homopolymer run > {max_homopolymer_length} removed ({num_rows_homopolymer/total_mutations*100:.3f}%)
        """

    if min_triplet_complexity is not None:
        report += f"""  {num_rows_triplet} variants with triplet complexity < {min_triplet_complexity} removed ({num_rows_triplet/total_mutations*100:.3f}%)
        """

    if remove_alignment_to_reference:
        report += f"""  {num_rows_pseudoaligned} variants removed for pseudoaligning to the reference ({num_rows_pseudoaligned/total_mutations*100:.3f}%)
        """

    if good_mutations != total_mutations:
        logger.warning(report)
    else:
        logger.info("All variants correctly recorded")

    # Save the report string to the specified path
    with open(filtering_report_text_out, determine_write_mode(filtering_report_text_out, overwrite=overwrite, first_chunk=first_chunk), encoding="utf-8") as file:
        file.write(report)

    if translate and save_variants_updated_csv and store_full_sequences:
        columns_to_keep.extend(["wt_sequence_aa_full", "vcrs_sequence_aa_full"])

        if not translate_start:
            translate_start = "translate_start"
        if not translate_end:
            translate_end = "translate_end"

        if translate_start not in mutations.columns:
            mutations["translate_start"] = 0
        if translate_end not in mutations.columns:
            mutations["translate_end"] = None

        if verbose:
            tqdm.pandas(desc="Translating WT amino acid sequences")

        mutations["wt_sequence_aa_full"] = mutations.apply(
            lambda row: translate_sequence(row["wt_sequence_full"], row["translate_start"], row["translate_end"]),
            axis=1,
        )

        if verbose:
            tqdm.pandas(desc="Translating mutant amino acid sequences")

        mutations["vcrs_sequence_aa_full"] = mutations.apply(
            lambda row: translate_sequence(
                row["vcrs_sequence_full"],
                row[translate_start],
                row[translate_end],
            ),
            axis=1,
        )

    mutations = mutations[columns_to_keep]

    # save text files of mutations filtered out
    final_mutation_id_set = set(mutations["header"].dropna())

    removed_mutation_set = initial_mutation_id_set - final_mutation_id_set
    del initial_mutation_id_set, final_mutation_id_set

    # Save as a newline-separated text file
    with open(removed_variants_text_out, determine_write_mode(removed_variants_text_out, overwrite=overwrite, first_chunk=first_chunk), encoding="utf-8") as file:
        for mutation in removed_mutation_set:
            file.write(f"{mutation}\n")

    if save_variants_updated_csv:
        # recalculate start_variant_position and end_variant_position due to messing with it above
        mutations.drop(
            columns=["start_variant_position", "end_variant_position"],
            inplace=True,
            errors="ignore",
        )
        mutations["start_variant_position"] = split_positions[0]
        if split_positions.shape[1] > 1:
            mutations["end_variant_position"] = split_positions[1].fillna(split_positions[0])
        else:
            mutations["end_variant_position"] = mutations["start_variant_position"]

        mutations[["start_variant_position", "end_variant_position"]] = mutations[["start_variant_position", "end_variant_position"]].astype(int)

    if merge_identical:
        logger.info("Merging rows of identical VCRSs")

        mutations = mutations.sort_values(by="header", ascending=True)  # so that the headers are merged in alphabetical order

        # total mutations
        number_of_mutations_total = len(mutations)

        if merge_identical_rc:
            mutations["vcrs_sequence_rc"] = mutations["vcrs_sequence"].apply(reverse_complement)

            # Create a column that stores a sorted tuple of (vcrs_sequence, vcrs_sequence_rc)
            mutations["vcrs_sequence_and_rc_tuple"] = mutations.apply(
                lambda row: tuple(sorted([row["vcrs_sequence"], row["vcrs_sequence_rc"]])),
                axis=1,
            )

            # mutations = mutations.drop(columns=['vcrs_sequence_rc'])

            group_key = "vcrs_sequence_and_rc_tuple"
            columns_not_to_semicolon_join = [
                "vcrs_sequence",
                "vcrs_sequence_rc",
                "vcrs_sequence_and_rc_tuple",
            ]
            agg_columns = mutations.columns

        else:
            group_key = "vcrs_sequence"
            columns_not_to_semicolon_join = []
            agg_columns = [col for col in mutations.columns if col != "vcrs_sequence"]

        if save_variants_updated_csv:
            logger.warning("Merging rows of identical VCRSs can take a while if save_variants_updated_csv=True since it will concatenate all VCRSs too")
            mutations = mutations.groupby(group_key, sort=False).agg({col: ("first" if col in columns_not_to_semicolon_join else (";".join if col == "header" else lambda x: list(x.fillna(np.nan)))) for col in agg_columns}).reset_index(drop=merge_identical_rc)  # lambda x: list(x) will make simple list, but lengths will be inconsistent with NaN values  # concatenate values with semicolons: lambda x: `";".join(x.astype(str))`   # drop if merging by vcrs_sequence_and_rc_tuple, but not if merging by vcrs_sequence
            if original_order:
                mutations["original_order"] = mutations["original_order"].apply(min)  # get the minimum original order for each group
        else:
            if original_order:
                mutations_temp = mutations.groupby(group_key, sort=False, group_keys=False).agg({"header": ";".join, "original_order": lambda x: min(x)}).reset_index()  # Take the minimum order value
            else:
                mutations_temp = mutations.groupby(group_key, sort=False, group_keys=False)["header"].apply(";".join).reset_index()  # ignores original_order

            if merge_identical_rc:
                mutations_temp = mutations_temp.merge(mutations[["vcrs_sequence", group_key]], on=group_key, how="left")
                mutations_temp = mutations_temp.drop_duplicates(subset="header")
                mutations_temp.drop(columns=[group_key], inplace=True)

            mutations = mutations_temp
            del mutations_temp

        if "vcrs_sequence_and_rc_tuple" in mutations.columns:
            mutations = mutations.drop(columns=["vcrs_sequence_and_rc_tuple"])

        if merge_subsequences:
            logger.info("Merging VCRSs that are subsequences of other VCRSs")
            mutations, number_of_subsequence_merges = merge_subsequence_vcrss(mutations, merge_identical_rc=merge_identical_rc)
            if number_of_subsequence_merges > 0:
                logger.info(f"Merged {number_of_subsequence_merges} subsequence VCRS(s) into a supersequence VCRS")
            else:
                logger.info("No subsequence VCRSs found to merge")

        # Calculate the number of semicolons in each entry
        mutations["semicolon_count"] = mutations["header"].str.count(";")

        # number of VCRSs
        number_of_vcrss = len(mutations)

        # number_of_unique_mutations
        number_of_unique_mutations = (mutations["semicolon_count"] == 0).sum()

        number_of_merged_mutations = number_of_mutations_total - number_of_unique_mutations

        # equivalent code to calculate number_of_merged_mutations
        # mutations["semicolon_count"] += 1

        # # Convert all 1 values to NaN
        # mutations["semicolon_count"] = mutations["semicolon_count"].replace(1, np.nan)

        # # Take the sum across all rows of the new column
        # number_of_merged_mutations = int(mutations["semicolon_count"].sum())

        mutations = mutations.drop(columns=["semicolon_count"])

        merging_report = f"""
        Number of variants total: {number_of_mutations_total}
        Number of variants merged: {number_of_merged_mutations}
        Number of unique variants: {number_of_unique_mutations}
        Number of VCRSs: {number_of_vcrss}
        """

        # Save the report string to the specified path
        with open(filtering_report_text_out, determine_write_mode(filtering_report_text_out, overwrite=overwrite, first_chunk=first_chunk), encoding="utf-8") as file:
            file.write(merging_report)

        logger.info(merging_report)
        logger.info("Merged headers were combined and separated using a semicolon (;). Occurences of identical VCRSs may be reduced by increasing w.")

    empty_kmer_count = (mutations["vcrs_sequence"] == "").sum()

    if empty_kmer_count > 0:
        logger.warning(f"{empty_kmer_count} VCRSs were empty and were not included in the output.")

    mutations = mutations[mutations["vcrs_sequence"] != ""]

    # Restore the original order (minus any dropped rows)
    if original_order:
        mutations = mutations.sort_values(by="original_order").drop(columns="original_order")

    mutations.rename(columns={"header": "vcrs_header"}, inplace=True)
    if use_IDs:  # or (var_id_column in mutations.columns and not merge_identical):
        vcrs_id_start = get_last_vcrs_number(id_to_header_csv_out) + 1 if not first_chunk else 1
        mutations["vcrs_id"] = generate_unique_ids(len(mutations), start=vcrs_id_start, total_rows=total_rows)
        mutations[["vcrs_id", "vcrs_header"]].to_csv(id_to_header_csv_out, index=False, header=first_chunk, mode=determine_write_mode(id_to_header_csv_out, overwrite=overwrite, first_chunk=first_chunk))  # make the mapping csv
    else:
        mutations["vcrs_id"] = mutations["vcrs_header"]
    columns_to_keep.extend(["vcrs_id", "vcrs_header"])

    if save_variants_updated_csv:  # use variants_updated_csv_out if present,
        logger.info("Saving dataframe with updated variant info...")
        logger.warning("File size can be very large if the number of variants is large.")
        mutations.to_csv(variants_updated_csv_out, index=False, header=first_chunk, mode=determine_write_mode(variants_updated_csv_out, overwrite=overwrite, first_chunk=first_chunk))
        logger.info(f"Updated variant info has been saved to {variants_updated_csv_out}")

    if len(mutations) > 0:
        mutations["fasta_format"] = ">" + mutations["vcrs_id"] + "\n" + mutations["vcrs_sequence"] + "\n"

    # Save the filtered VCRSs in the filtered fasta file (this is what the index/t2g are built from)
    if not mutations.empty:
        with open(vcrs_fasta_out, determine_write_mode(vcrs_fasta_out, overwrite=overwrite, first_chunk=first_chunk), encoding="utf-8") as fasta_file:
            fasta_file.write("".join(mutations["fasta_format"].values))

        create_identity_t2g(vcrs_fasta_out, vcrs_t2g_out, mode=determine_write_mode(vcrs_t2g_out, overwrite=overwrite, first_chunk=first_chunk))

    logger.info("Filtered FASTA file containing VCRSs created at %s.", vcrs_fasta_out)
    logger.info("t2g file containing VCRSs created at %s.", vcrs_t2g_out)

    # When stream_output is True, return list of mutated seqs (no index is created in this streaming mode)
    if return_variant_output:
        all_mut_seqs = []
        all_mut_seqs.extend(mutations["vcrs_sequence"].values)

        # Remove empty strings from final list of mutated sequences (these are introduced when unknown mutations are encountered)
        while "" in all_mut_seqs:
            all_mut_seqs.remove("")

        return all_mut_seqs if len(all_mut_seqs) > 0 else None

    # * Create the VCRS (kallisto) index via kb ref (on by default; disable with the hidden dont_create_index flag).
    # Skipped for per-chunk recursive calls; the top-level chunked call creates the index once after merging (see chunk-iteration block above).
    if not kwargs.get("running_within_chunk_iteration", False):
        if not dont_create_index:
            dlist_value = resolve_dlist(
                dlist,
                alignment_to_reference_dna,
                alignment_to_reference_gtf,
                sequences,
                gtf,
                out_dir=(reference_out_dir if reference_out_dir else os.path.join(out, "reference")),
                k=k,
                overwrite=overwrite,
                dry_run=False,
            )
            create_vcrs_index(
                vcrs_fasta_out,
                index_out,
                k,
                threads=threads,
                overwrite=overwrite,
                dry_run=False,
                kb_ref_kwargs=kwargs,
                dlist=dlist_value,
            )

        # return the produced reference file paths (for backwards compatibility with vk ref, which returned this dict)
        vk_ref_output_dict = {
            "index": os.path.abspath(index_out) if (isinstance(index_out, str) and os.path.isfile(index_out)) else None,
            "t2g": os.path.abspath(vcrs_t2g_out) if (isinstance(vcrs_t2g_out, str) and os.path.isfile(vcrs_t2g_out)) else None,
            "fasta": os.path.abspath(vcrs_fasta_out) if (isinstance(vcrs_fasta_out, str) and os.path.isfile(vcrs_fasta_out)) else None,
        }
        logger.info(f"Produced files: {vk_ref_output_dict}")
        return vk_ref_output_dict
