"""varseek constant values."""

from collections import defaultdict

fasta_extensions = (".fa", ".fasta", ".fa.gz", ".fasta.gz", ".fna", ".fna.gz", ".ffn", ".ffn.gz")
fastq_extensions = (".fq", ".fastq", ".fq.gz", ".fastq.gz")

# Canonical per-technology metadata. One nested entry per sequencing technology is the
# single source of truth for the technology vocabulary and every per-technology lookup.
# Look these up with the canonical (upper-cased) technology name, e.g.
# technology_info[technology.upper()]["num_files"]. Fields:
#   strand_bias           -> tuple of possible strand-bias ends, or None
#   transcript_file_index -> index of the FASTQ file holding the transcript/cDNA read
#   barcode_position      -> (file_index, start, end) of the barcode (nested tuple if the
#                            barcode is split across positions); None if no barcode. Same
#                            format as "barcode" in `kb --list`.
#   num_files             -> number of FASTQ files (int, or {"single": .., "paired": ..})
#   barcode_umi           -> barcode/UMI/spacer offsets used by fastq preprocessing, or
#                            None if the technology has no barcode/UMI entry.
technology_info = {
    "10XV1": {
        "strand_bias": ("5p", "3p"),
        "transcript_file_index": 2,
        "barcode_position": (0, 0, 14),
        "num_files": 3,
        "barcode_umi": None,
    },
    "10XV2": {
        "strand_bias": ("5p", "3p"),
        "transcript_file_index": 1,
        "barcode_position": (0, 0, 16),
        "num_files": 2,
        "barcode_umi": {"barcode_start": 0, "barcode_end": 16, "umi_start": 16, "umi_end": 26, "spacer_start": None, "spacer_end": None},
    },
    "10XV3": {
        "strand_bias": ("5p", "3p"),
        "transcript_file_index": 1,
        "barcode_position": (0, 0, 16),
        "num_files": 2,
        "barcode_umi": {"barcode_start": 0, "barcode_end": 16, "umi_start": 16, "umi_end": 28, "spacer_start": None, "spacer_end": None},
    },
    "10XV3_ULTIMA": {
        "strand_bias": ("5p", "3p"),
        "transcript_file_index": 0,
        "barcode_position": (0, 22, 38),
        "num_files": 1,
        "barcode_umi": None,
    },
    "10XV4": {
        "strand_bias": ("5p", "3p"),
        "transcript_file_index": 1,
        "barcode_position": (0, 0, 16),
        "num_files": 2,
        "barcode_umi": {"barcode_start": 0, "barcode_end": 16, "umi_start": 16, "umi_end": 26, "spacer_start": None, "spacer_end": None},
    },
    "BDWTA": {
        "strand_bias": ("3p",),
        "transcript_file_index": 1,
        "barcode_position": ((0, 0, 9), (0, 21, 30), (0, 43, 52)),
        "num_files": 2,
        "barcode_umi": None,
    },
    "BULK": {
        "strand_bias": None,
        "transcript_file_index": 0,
        "barcode_position": None,
        "num_files": {"single": 1, "paired": 2},
        "barcode_umi": {"barcode_start": None, "barcode_end": None, "umi_start": None, "umi_end": None, "spacer_start": None, "spacer_end": None},
    },
    "CELSEQ": {
        "strand_bias": ("3p",),
        "transcript_file_index": 1,
        "barcode_position": (0, 0, 8),
        "num_files": 2,
        "barcode_umi": None,
    },
    "CELSEQ2": {
        "strand_bias": ("3p",),
        "transcript_file_index": 1,
        "barcode_position": (0, 6, 12),
        "num_files": 2,
        "barcode_umi": None,
    },
    "DROPSEQ": {
        "strand_bias": ("3p",),
        "transcript_file_index": 1,
        "barcode_position": (0, 0, 12),
        "num_files": 2,
        "barcode_umi": None,
    },
    "INDROPSV1": {
        "strand_bias": ("3p",),
        "transcript_file_index": 1,
        "barcode_position": ((0, 0, 11), (0, 30, 38)),
        "num_files": 2,
        "barcode_umi": None,
    },
    "INDROPSV2": {
        "strand_bias": ("3p",),
        "transcript_file_index": 0,
        "barcode_position": ((1, 0, 11), (1, 30, 38)),
        "num_files": 1,
        "barcode_umi": None,
    },
    "INDROPSV3": {
        "strand_bias": ("3p",),
        "transcript_file_index": 2,
        "barcode_position": (0, 0, 8),
        "num_files": 3,
        "barcode_umi": None,
    },
    "SCRUBSEQ": {
        "strand_bias": ("3p",),
        "transcript_file_index": 1,
        "barcode_position": (0, 0, 6),
        "num_files": 2,
        "barcode_umi": None,
    },
    "SMARTSEQ2": {
        "strand_bias": None,
        "transcript_file_index": 0,
        "barcode_position": None,
        "num_files": {"single": 1, "paired": 2},
        "barcode_umi": {"barcode_start": None, "barcode_end": None, "umi_start": None, "umi_end": None, "spacer_start": None, "spacer_end": None},
    },
    "SMARTSEQ3": {
        "strand_bias": None,
        "transcript_file_index": 0,
        "barcode_position": None,
        "num_files": 2,
        "barcode_umi": {"barcode_start": None, "barcode_end": None, "umi_start": 11, "umi_end": 19, "spacer_start": 0, "spacer_end": 11},
    },
    "SPLIT-SEQ": {
        "strand_bias": ("3p",),
        "transcript_file_index": 0,
        "barcode_position": ((1, 10, 18), (1, 48, 56), (1, 78, 86)),
        "num_files": 2,
        "barcode_umi": None,
    },
    "STORMSEQ": {
        "strand_bias": ("3p",),
        "transcript_file_index": 0,
        "barcode_position": None,
        "num_files": 2,
        "barcode_umi": None,
    },
    "SURECELL": {
        "strand_bias": ("3p",),
        "transcript_file_index": 1,
        "barcode_position": ((0, 0, 6), (0, 21, 27), (0, 42, 48)),
        "num_files": 2,
        "barcode_umi": None,
    },
    "VISIUM": {
        "strand_bias": ("3p",),
        "transcript_file_index": 1,
        "barcode_position": (0, 0, 16),
        "num_files": 2,
        "barcode_umi": {"barcode_start": 0, "barcode_end": 16, "umi_start": 16, "umi_end": 28, "spacer_start": None, "spacer_end": None},
    },
}
technology_valid_values = set(technology_info)
non_single_cell_technologies = {"BULK", "VISIUM"}
supported_downloadable_normal_reference_genomes_with_kb_ref = {"human", "mouse", "dog", "monkey", "zebrafish"}  # see full list at https://github.com/pachterlab/kallisto-transcriptome-indices/


complement_trans = str.maketrans("ACGTNacgtn.", "TGCANtgcan.")

# Get complement
complement = {
    "A": "T",
    "T": "A",
    "U": "A",
    "C": "G",
    "G": "C",
    "N": "N",
    "a": "t",
    "t": "a",
    "u": "a",
    "c": "g",
    "g": "c",
    "n": "n",
    "*": "*",
    ".": ".",  # annotation for gaps
    "-": "-",  # annotation for gaps
    ">": ">",  # in case mutation section has a '>' character indicating substitution
}


codon_to_amino_acid = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# this should be a dict of database:reference_sequence
# reference_sequence should be a dict of reference_sequence_type:download_info
# download_info should be a string of the command to download the reference sequence - use OUT_DIR as the output directory, and replace in the script


# a dictionary that maps from dict[variants][sequences] to a dict of files {"index": index_url, "t2g": t2g_url}

default_filename_dict = {"index": "vcrs_index.idx", "t2g": "vcrs_t2g.txt", "fasta": "vcrs_fasta.fa"}

# * variants, sequences, w, k - single string, comma-separated
# * matches varseek ref and server
# * for cosmic, leave the value "COSMIC" in place of a link (used for authentication), and keep the links in varseek_server/validate_cosmic.py; for others, replace with a link
prebuilt_vk_ref_files = {
    "variants=cosmic_cmc,sequences=cdna": {"index": "COSMIC", "t2g": "COSMIC", "fasta": "COSMIC"},
    "variants=geuvadis,sequences=cdna": {"index": "https://caltech.box.com/shared/static/gk3bxwsm2huvs5xm6v0lfv2oleatlerc.idx", "t2g": "https://caltech.box.com/shared/static/3zc01t3h9kt2nurbikbagqrmo0xsxcmz.txt", "fasta": "https://caltech.box.com/shared/static/foimmij3hz8etf444jpb4h3gk5p5282f.fa"},
}


supported_databases_and_corresponding_reference_sequence_type = {
    "cosmic_cmc": {
        "sequence_download_commands": {
            "genome": "gget ref -w FILES_TO_DOWNLOAD -r ENSEMBL_VERSION --out_dir OUT_DIR -d SPECIES",  # dna,gtf
            "cdna": "gget ref -w FILES_TO_DOWNLOAD -r ENSEMBL_VERSION --out_dir OUT_DIR -d SPECIES",  # cdna,cds
            "cds": "gget ref -w FILES_TO_DOWNLOAD -r ENSEMBL_VERSION --out_dir OUT_DIR -d SPECIES",  # cds
        },
        "sequence_file_names": {
            "genome": "Homo_sapiens.GRChGRCH_NUMBER.dna.primary_assembly.fa",
            "gtf": "Homo_sapiens.GRChGRCH_NUMBER.87.gtf",
            "cdna": "Homo_sapiens.GRChGRCH_NUMBER.cdna.all.fa",
            "cds": "Homo_sapiens.GRChGRCH_NUMBER.cds.all.fa",
        },
        "database_version_to_reference_release": defaultdict(lambda: "93", {"100": "93", "101": "93"}),  # sets default to 93
        "database_version_to_reference_assembly_build": defaultdict(lambda: ("37",), {"100": ("37",), "101": ("37",)}),  # sets default to ("37",)
        "variant_file_name": "CancerMutationCensus_AllData_Tsv_vCOSMIC_RELEASE_GRChGRCH_NUMBER/CancerMutationCensus_AllData_vCOSMIC_RELEASE_GRChGRCH_NUMBER_mutation_workflow.csv",
        "column_names": {
            "seq_id_genome_column": "chromosome",
            "var_genome_column": "mutation_genome",
            "seq_id_cdna_column": "seq_ID",
            "var_cdna_column": "mutation_cdna",
            "seq_id_cds_column": "seq_ID",
            "var_cds_column": "mutation",
            "var_id_column": "mutation_id",
            "gene_name_column": "gene_name",
        },
    }
}

# def recursive_defaultdict():
#     return defaultdict(recursive_defaultdict)

# supported_databases_and_corresponding_reference_sequence_type = defaultdict(recursive_defaultdict, supported_databases_and_corresponding_reference_sequence_type)  # can unexpectedly add keys when indexing


seqID_pattern = r"(ENST\d+|(?:[1-9]|1[0-9]|2[0-3]|X|Y|MT)\d+)"
mutation_pattern = r"(?:c|g)\.([0-9_\-\+\*\(\)\?]+)([a-zA-Z>]+)"  # more complex: r'c\.([0-9_\-\+\*\(\)\?]+)([a-zA-Z>\(\)0-9]+)'
HGVS_pattern = rf"^{seqID_pattern}:{mutation_pattern}$"

seqID_pattern_general = r"[A-Za-z0-9_-]+"
HGVS_pattern_general = rf"^{seqID_pattern_general}:{mutation_pattern}$"

varseek_ref_only_allowable_kb_ref_arguments = {"zero_arguments": {"--keep-tmp", "--verbose", "--aa"}, "one_argument": {"--tmp", "--kallisto", "--bustools"}, "multiple_arguments": set()}  # don't include d-list, t, i, k, workflow, overwrite here because I do it myself later

varseek_count_only_allowable_kb_count_arguments = {
    "zero_arguments": {"--keep-tmp", "--verbose", "--tcc", "--cellranger", "--gene-names", "--report", "--long", "--opt-off", "--matrix-to-files", "--matrix-to-directories"},
    "one_argument": {"--tmp", "--kallisto", "--bustools", "-w", "-r", "-m", "--inleaved", "--filter", "--filter-threshold", "-N", "--threshold", "--platform"},
    "multiple_arguments": set(),
}  # don't include t, i, workflow here because I do it myself later; cannot take in a custom value for k (because this would get confusing with k for fastqpp/clean)

# species -> reference_type -> k -> prebuilt-index URL (placeholder URLs for now).
# Source of truth for the Species vocabulary (type_utils.Species derives from its keys).
species_to_url = {
    "human": {
        "genome": {"41": "https://example.com/human_index.idx"},
        "cdna": {"41": "https://example.com/human_cdna_index.idx"},
        "transcriptome": {"41": "https://example.com/human_transcriptome_index.idx"},
        "genome_or_transcriptome": {"41": "https://example.com/human_genome_or_transcriptome_index.idx"},
    },
}

# a list of dictionaries with keys "variants", "sequences", and "description" (used by the download / list_downloadable_references features that vk build absorbed from vk ref)
downloadable_references = [
    {"description": "COSMIC Cancer Mutation Census version 101 - Ensembl GRCh37 release 93 cDNA reference annotations. w=47, k=51. Header format (showing the column(s) from the original database used): 'seq_ID':'mutation_cdna'.", "download_command": "vk build -v cosmic_cmc -s cdna -d"},
    {"description": "Geuvadis dataset - Ensembl GRCh37 release 113 cDNA reference annotations. w=37, k=41. Header format (showing the column(s) from the original database used): 'seq_ID':'mutation_cdna'.", "download_command": "vk build -v geuvadis -s cdna -d"},
]