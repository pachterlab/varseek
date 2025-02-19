import os
import shutil
from pathlib import Path
import sys

import pandas as pd
import pytest

import varseek as vk
from varseek.utils import add_mutation_type

from .conftest import (
    compare_two_dataframes_without_regard_for_order_of_rows_or_columns,
    compare_two_fastas_without_regard_for_order_of_entries,
    compare_two_id_to_header_mappings,
    compare_two_t2gs,
)

test_directory = os.path.dirname(os.path.abspath(__file__))
ground_truth_folder = os.path.join(test_directory, "pytest_ground_truth")
reference_folder_parent = os.path.join(os.path.dirname(test_directory), "data", "reference")
ensembl_grch37_release93_folder = os.path.join(reference_folder_parent, "ensembl_grch37_release93")
cosmic_csv_path_starting = os.path.join(reference_folder_parent, "cosmic", "CancerMutationCensus_AllData_Tsv_v100_GRCh37_v2", "CancerMutationCensus_AllData_v100_GRCh37_mutation_workflow.csv")

sample_size=12_000  # 2,000 each for each of the 6 mutation types
columns_to_drop_info_filter = None  # drops columns for info and filter df - will not throw an error if the column does not exist in the df   # ["nearby_variants", "number_of_kmers_with_overlap_to_other_VCRSs", "number_of_other_VCRSs_with_overlapping_kmers", "overlapping_kmers", "VCRSs_with_overlapping_kmers", "kmer_overlap_with_other_VCRSs"]
make_new_gt = False


# If "tests/test_ref.py" is not explicitly in the command line arguments, skip this module.
if not any("tests/test_ref.py" in arg for arg in sys.argv):
    pytest.skip("Skipping test_ref.py due to its slow nature; run this file by explicity including the file i.e., 'pytest tests/test_ref.py'", allow_module_level=True)


@pytest.fixture
def cosmic_csv_path():
    global cosmic_csv_path_starting, sample_size, make_new_gt, ground_truth_folder

    if (not os.path.exists(ground_truth_folder) or not os.listdir(ground_truth_folder)):
        if not make_new_gt:
            pytest.skip("Ground truth folder is missing or empty, and make_new_gt is False. Skipping test.")

    if not os.path.exists(cosmic_csv_path_starting):
        pytest.skip("cosmic_csv_path_starting not found. Please download it to continue")

    subsampled_cosmic_csv_path = Path(str(cosmic_csv_path_starting).replace(".csv", "_subsampled_pytest.csv"))
    if not os.path.exists(subsampled_cosmic_csv_path):

        mutations = pd.read_csv(cosmic_csv_path_starting)
        mutations = add_mutation_type(mutations, var_column = "mutation_cdna")

        mutation_types = ["substitution", "deletion", "insertion", "delins", "duplication", "inversion"]
        sample_size_per_mutation_type = sample_size // len(mutation_types)

        final_df = pd.DataFrame()

        for mutation_type in mutation_types:
            # Filter DataFrame for the current mutation type
            mutation_subset = mutations[mutations["mutation_type"] == mutation_type]

            sample_size_round = min(sample_size_per_mutation_type, len(mutation_subset))
            
            # Randomly sample `sample_size_per_mutation_type` rows from the subset (adjusts if less than sample_size_per_mutation_type)
            sample = mutation_subset.sample(n=min(sample_size_round, len(mutation_subset)), random_state=42)
            
            # Append the sample to the final DataFrame
            final_df = pd.concat([final_df, sample], ignore_index=True)

        
        final_df.to_csv(subsampled_cosmic_csv_path, index=False)

    return subsampled_cosmic_csv_path


def apply_file_comparison(test_path, ground_truth_path, file_type, columns_to_drop_info_filter = None):
    if file_type == "fasta":
        compare_two_fastas_without_regard_for_order_of_entries(test_path, ground_truth_path)
    elif file_type == "df":
        compare_two_dataframes_without_regard_for_order_of_rows_or_columns(test_path, ground_truth_path, columns_to_drop = columns_to_drop_info_filter)
    elif file_type == "id_to_header_mapping":
        compare_two_id_to_header_mappings(test_path, ground_truth_path)
    elif file_type == "t2g":
        compare_two_t2gs(test_path, ground_truth_path)
    else:
        raise ValueError(f"File type {file_type} is not supported.")

#* note: temp files will be deleted upon completion of the test or running into an error - to debug with a temp file, place a breakpoint before the error occurs
def test_vk_ref(cosmic_csv_path, tmp_path):
    # global ground_truth_folder, reference_folder_parent, make_new_gt, ensembl_grch37_release93_folder

    cosmic_cdna_path = os.path.join(ensembl_grch37_release93_folder, "Homo_sapiens.GRCh37.cds.all.fa")
    cosmic_genome_path = os.path.join(ensembl_grch37_release93_folder, "Homo_sapiens.GRCh37.dna.primary_assembly.fa")
    cosmic_gtf_path = os.path.join(ensembl_grch37_release93_folder, "Homo_sapiens.GRCh37.87.gtf")

    # skip this run if you don't have the ground truth and are not making it
    if not os.path.exists(ground_truth_folder) or not os.listdir(ground_truth_folder):
        if not make_new_gt:
            pytest.skip("Ground truth folder is missing or empty, and make_new_gt is False. Skipping test.")

    for path in [cosmic_csv_path, cosmic_cdna_path, cosmic_genome_path, cosmic_gtf_path]:
        if not os.path.isfile(path):
            pytest.skip(f"{path} not found. Please download it to continue")

    var_column = "mutation_cdna"
    w = 54
    k = 55
    dlist_reference_source = "T2T"
    columns_to_include = "all"
    threads = 2

    if make_new_gt:
        os.makedirs(ground_truth_folder, exist_ok=True)

    out_dir = tmp_path

    vk.ref(
        variants = cosmic_csv_path,  # build args
        sequences = cosmic_cdna_path,
        out = out_dir,
        var_column = var_column,
        w = w,
        k = k,
        save_variants_updated_csv = True,
        columns_to_include = columns_to_include,  # info args
        dlist_reference_source = dlist_reference_source,
        reference_folder = reference_folder_parent,
        dlist_reference_genome_fasta = cosmic_genome_path,  # for d-listing
        dlist_reference_cdna_fasta = cosmic_cdna_path,  # for d-listing
        dlist_reference_gtf = cosmic_gtf_path,  # for d-listing
        reference_genome_fasta = cosmic_genome_path,  # for compare_cdna_and_genome
        reference_cdna_fasta = cosmic_cdna_path,  # for compare_cdna_and_genome
        gtf = cosmic_gtf_path,  # for distance to nearest splice junction
        save_variants_updated_exploded_vk_info_csv = True,
        threads = threads,
        filters = filters,  # filter args
    )

    # file name, file type, columns to drop for comparison
    files_to_compare_and_file_type = [
        ("vcrs.fa", "fasta", None),
        ("CancerMutationCensus_AllData_v100_GRCh37_mutation_workflow_with_cdna_subsampled_pytest_updated.csv", "df", None),
        ("id_to_header_mapping.csv", "id_to_header_mapping", None),
        ("vcrs_t2g.txt", "t2g", None),
        ("dlist.fa", "fasta", None),
        ("variants_updated_vk_info.csv", "df", columns_to_drop_info_filter),
        ("variants_updated_exploded_vk_info.csv", "df", columns_to_drop_info_filter),
        ("vcrs_filtered.fa", "fasta", None),
        ("variants_updated_filtered.csv", "df", columns_to_drop_info_filter),
        ("variants_updated_exploded_filtered.csv", "df", columns_to_drop_info_filter),
        ("dlist_filtered.fa", "fasta", None),
        ("vcrs_t2g_filtered.txt", "t2g", None),
        ("id_to_header_mapping_filtered.csv", "id_to_header_mapping", None)
    ]

    for file, file_type, columns_to_drop_info_filter in files_to_compare_and_file_type:
        test_path = os.path.join(out_dir, file)
        ground_truth_path = os.path.join(ground_truth_folder, file)
        if make_new_gt:
            shutil.copy(test_path, ground_truth_path)
        apply_file_comparison(test_path, ground_truth_path, file_type, columns_to_drop_info_filter)
