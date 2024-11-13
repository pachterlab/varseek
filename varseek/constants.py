import re
from collections import defaultdict

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
    ".": ".",  # annotation for gaps
    "-": "-",  # annotation for gaps
    ">": ">",  # in case mutation section has a '>' character indicating substitution
    "x": "x",  # in case one uses 'x' in place of '>'
    "X": "X",  # in case one uses 'X' in place of '>'
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


supported_databases_and_corresponding_reference_sequence_type = {
    "cosmic_cmc": {
        "sequence_download_commands": {
            "genome": "gget ref -w dna,gtf -r ENSEMBL_VERSION --out_dir OUT_DIR -d GRCH_NUMBER",
            "cdna": "gget ref -w cdna,cds -r ENSEMBL_VERSION --out_dir OUT_DIR -d GRCH_NUMBER",
            "cds": "gget ref -w cds -r ENSEMBL_VERSION --out_dir OUT_DIR -d GRCH_NUMBER",
        },
        "sequence_file_names": {
            "genome": "Homo_sapiens.GRChGRCH_NUMBER.dna.primary_assembly.fa",
            "gtf": "Homo_sapiens.GRChGRCH_NUMBER.87.gtf",
            "cdna": "Homo_sapiens.GRChGRCH_NUMBER.cdna.all.fa",
            "cds": "Homo_sapiens.GRChGRCH_NUMBER.cds.all.fa",
        },
        "database_version_to_reference_release": {"100": "93"},
        "database_version_to_reference_assembly_build": {"100": "37"},
        "mutation_file_name": "CancerMutationCensus_AllData_Tsv_vCOSMIC_RELEASE_GRChGRCH_NUMBER/CancerMutationCensus_AllData_vCOSMIC_RELEASE_GRChGRCH_NUMBER_mutation_workflow.csv",
    }
}

# def recursive_defaultdict():
#     return defaultdict(recursive_defaultdict)

# supported_databases_and_corresponding_reference_sequence_type = defaultdict(recursive_defaultdict, supported_databases_and_corresponding_reference_sequence_type)  # can unexpectedly add keys when indexing


seqID_pattern = r"(ENST\d+|(?:[1-9]|1[0-9]|2[0-3]|X|Y|MT)\d+)"
mutation_pattern = r"(?:c|g)\.([0-9_\-\+\*\(\)\?]+)([a-zA-Z>]+)"  # more complex: r'c\.([0-9_\-\+\*\(\)\?]+)([a-zA-Z>\(\)0-9]+)'
