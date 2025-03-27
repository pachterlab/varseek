"""varseek constant values."""

from collections import defaultdict

# allowable_kwargs = {
#     "varseek_build": {"insertion_size_limit", "min_seq_len", "optimize_flanking_regions", "remove_seqs_with_wt_kmers", "required_insertion_overlap_length", "merge_identical", "merge_identical_strandedness", "use_IDs", "cosmic_version", "cosmic_grch", "cosmic_email", "cosmic_password", "save_files"},
#     "varseek_info": {"bowtie_path"},
#     "varseek_filter": {"filter_all_dlists", "dlist_genome_fasta", "dlist_cdna_fasta", "dlist_genome_filtered_fasta_out", "dlist_cdna_filtered_fasta_out"},
#     "kb_ref": set(),
#     "kb_count": {"union"},
#     "varseek_fastqpp": {"seqtk"},
#     "varseek_clean": set(),
#     "varseek_summarize": set(),
#     "varseek_ref": set(),
#     "varseek_count": set()
# }

fasta_extensions = (".fa", ".fasta", ".fa.gz", ".fasta.gz", ".fna", ".fna.gz", ".ffn", ".ffn.gz")
fastq_extensions = (".fq", ".fastq", ".fq.gz", ".fastq.gz")

technology_valid_values = {"10XV1", "10XV2", "10XV3", "10XV3_ULTIMA", "BDWTA", "BULK", "CELSEQ", "CELSEQ2", "DROPSEQ", "INDROPSV1", "INDROPSV2", "INDROPSV3", "SCRUBSEQ", "SMARTSEQ2", "SMARTSEQ3", "SPLIT-SEQ", "STORMSEQ", "SURECELL", "VISIUM"}
non_single_cell_technologies = {"BULK", "VISIUM"}
supported_downloadable_normal_reference_genomes_with_kb_ref = {"human", "mouse", "dog", "monkey", "zebrafish"}  # see full list at https://github.com/pachterlab/kallisto-transcriptome-indices/
technology_to_strand_bias_mapping = {
    "10XV1": ("5p", "3p"),
    "10XV2": ("5p", "3p"),
    "10XV3": ("5p", "3p"),
    "10XV3_ULTIMA": ("5p", "3p"),
    "BDWTA": ("3p",),
    "BULK": None,
    "CELSEQ": ("3p",),
    "CELSEQ2": ("3p",),
    "DROPSEQ": ("3p",),
    "INDROPSV1": ("3p",),
    "INDROPSV2": ("3p",),
    "INDROPSV3": ("3p",),
    "SCRUBSEQ": ("3p",),
    "SMARTSEQ2": None,
    "SMARTSEQ3": None,
    "SPLIT-SEQ": ("3p",),
    "STORMSEQ": ("3p",),
    "SURECELL": ("3p",),
    "VISIUM": ("3p",),
}
technology_to_file_index_with_transcripts_mapping = {
    "10XV1": 2,
    "10XV2": 1,
    "10XV3": 1,
    "10XV3_ULTIMA": 0,
    "BDWTA": 1,
    "BULK": 0,
    "CELSEQ": 1,
    "CELSEQ2": 1,
    "DROPSEQ": 1,
    "INDROPSV1": 1,
    "INDROPSV2": 0,
    "INDROPSV3": 2,
    "SCRUBSEQ": 1,
    "SMARTSEQ2": 0,
    "SMARTSEQ3": 0,
    "SPLIT-SEQ": 0,
    "STORMSEQ": 0,
    "SURECELL": 1,
    "VISIUM": 1,
}
# None means no barcode/umi
technology_barcode_and_umi_dict = {
    "bulk": {"barcode_start": None, "barcode_end": None, "umi_start": None, "umi_end": None, "spacer_start": None, "spacer_end": None},
    "10xv2": {"barcode_start": 0, "barcode_end": 16, "umi_start": 16, "umi_end": 26, "spacer_start": None, "spacer_end": None},
    "10xv3": {"barcode_start": 0, "barcode_end": 16, "umi_start": 16, "umi_end": 28, "spacer_start": None, "spacer_end": None},
    "Visium": {"barcode_start": 0, "barcode_end": 16, "umi_start": 16, "umi_end": 28, "spacer_start": None, "spacer_end": None},
    "SMARTSEQ2": {"barcode_start": None, "barcode_end": None, "umi_start": None, "umi_end": None, "spacer_start": None, "spacer_end": None},
    "SMARTSEQ3": {"barcode_start": None, "barcode_end": None, "umi_start": 11, "umi_end": 19, "spacer_start": 0, "spacer_end": 11},
}


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

# * variants, sequences, w, k, dlist_reference_source - single string, comma-separated
# * matches varseek ref and server
# * for cosmic, leave the value "COSMIC" in place of a link (used for authentication), and keep the links in varseek_server/validate_cosmic.py; for others, replace with a link
prebuilt_vk_ref_files = {
    "variants=cosmic_cmc,sequences=cdna,w=47,k=51,dlist_reference_source=grch37": {"index": "COSMIC", "t2g": "COSMIC", "fasta": "COSMIC"},
    "variants=cosmic_cmc,sequences=cdna,w=47,k=51,dlist_reference_source=t2t": {"index": "COSMIC", "t2g": "COSMIC", "fasta": "COSMIC"},
    #  ("variants=cosmic_cmc", "sequences=genome", "w=47", "k=51", "dlist_reference_source=grch37"): {"index": "COSMIC", "t2g": "COSMIC", "fasta": "COSMIC"},
    #  ("variants=cosmic_cmc", "sequences=genome", "w=47", "k=51", "dlist_reference_source=t2t"): {"index": "COSMIC", "t2g": "COSMIC", "fasta": "COSMIC"},
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

varseek_ref_only_allowable_kb_ref_arguments = {"zero_arguments": {"--keep-tmp", "--verbose", "--aa"}, "one_argument": {"--tmp", "--kallisto", "--bustools"}, "multiple_arguments": set()}  # don't include d-list, t, i, k, workflow, overwrite here because I do it myself later

varseek_count_only_allowable_kb_count_arguments = {
    "zero_arguments": {"--keep-tmp", "--verbose", "--tcc", "--cellranger", "--gene-names", "--report", "--long", "--opt-off", "--matrix-to-files", "--matrix-to-directories"},
    "one_argument": {"--tmp", "--kallisto", "--bustools", "-w", "-r", "-m", "--inleaved", "--filter", "--filter-threshold", "-N", "--threshold", "--platform"},
    "multiple_arguments": set(),
}  # don't include d-list, t, i, k, workflow here because I do it myself later
