import os
import shutil
from pathlib import Path
import sys
from datetime import datetime
import importlib

import pandas as pd
import pytest

from pdb import set_trace as st

import varseek as vk
from varseek.utils import add_variant_type

from .conftest import (
    compare_two_dataframes_without_regard_for_order_of_rows_or_columns,
    compare_two_fastas_without_regard_for_order_of_entries,
    compare_two_id_to_header_mappings,
    compare_two_t2gs,
)

sample_size=12_000  # 2,000 each for each of the 6 mutation types
columns_to_drop_info_filter = None  # drops columns for info and filter df - will not throw an error if the column does not exist in the df   # ["nearby_variants", "number_of_kmers_with_overlap_to_other_VCRSs", "number_of_other_VCRSs_with_overlapping_kmers", "overlapping_kmers", "VCRSs_with_overlapping_kmers", "kmer_overlap_with_other_VCRSs"]
make_new_gt = True
store_out_in_permanent_paths = True

test_directory = Path(__file__).resolve().parent
ground_truth_folder = os.path.join(test_directory, "pytest_ground_truth")
reference_folder_parent = os.path.join(os.path.dirname(test_directory), "data", "reference")
ensembl_grch37_release93_folder = os.path.join(reference_folder_parent, "ensembl_grch37_release93")
cosmic_csv_path_starting = os.path.join(reference_folder_parent, "cosmic", "CancerMutationCensus_AllData_Tsv_v100_GRCh37_v2", "CancerMutationCensus_AllData_v100_GRCh37_mutation_workflow.csv")
pytest_permanent_out_dir_base = test_directory / "pytest_output" / Path(__file__).stem
current_datetime = datetime.now().strftime("date_%Y_%m_%d_time_%H%M_%S")

#$ TOGGLE THIS SECTION TO HAVE THIS FILE RECOGNIZED BY PYTEST (commented out means it will be recognized, uncommented means it will be hidden)
# If "tests/test_ref.py" is not explicitly in the command line arguments, skip this module. - notice that uncommenting this will hide it from vscode such that I can't press the debug button
if not any("test_varseek_ref.py" in arg for arg in sys.argv):
    pytest.skip("Skipping test_varseek_ref.py due to its slow nature; run this file by explicity including the file i.e., 'pytest tests/test_varseek_ref.py'", allow_module_level=True)

@pytest.fixture
def out_dir(tmp_path, request):
    """Fixture that returns the appropriate output directory for each test."""
    if store_out_in_permanent_paths:
        current_test_function_name = request.node.name
        out = Path(f"{pytest_permanent_out_dir_base}/{current_datetime}/{current_test_function_name}")
    else:
        out = tmp_path / "out_vk_ref"

    out.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return out

@pytest.fixture
def cosmic_csv_path(out_dir):
    global cosmic_csv_path_starting, sample_size, make_new_gt, ground_truth_folder

    if (not os.path.exists(ground_truth_folder) or not os.listdir(ground_truth_folder)):
        if not make_new_gt:
            pytest.skip("Ground truth folder is missing or empty, and make_new_gt is False. Skipping test.")

    if not os.path.exists(cosmic_csv_path_starting):
        pytest.skip("cosmic_csv_path_starting not found. Please download it to continue")

    cosmic_csv_path_starting = Path(cosmic_csv_path_starting)
    # subsampled_cosmic_csv_path = out_dir / f"{cosmic_csv_path_starting.stem}_subsampled_pytest.csv"  # if I want to have it save in out_dir instead of the reference directory
    subsampled_cosmic_csv_path = Path(str(cosmic_csv_path_starting).replace(".csv", "_subsampled_pytest.csv"))
    if not os.path.exists(subsampled_cosmic_csv_path):

        mutations = pd.read_csv(cosmic_csv_path_starting)
        mutations = add_variant_type(mutations, var_column = "mutation_cdna")

        variant_types = ["substitution", "deletion", "insertion", "delins", "duplication", "inversion"]
        sample_size_per_variant_type = sample_size // len(variant_types)

        final_df = pd.DataFrame()

        for variant_type in variant_types:
            # Filter DataFrame for the current mutation type
            mutation_subset = mutations[mutations["variant_type"] == variant_type]

            sample_size_round = min(sample_size_per_variant_type, len(mutation_subset))
            
            # Randomly sample `sample_size_per_variant_type` rows from the subset (adjusts if less than sample_size_per_variant_type)
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
def test_vk_ref(cosmic_csv_path, out_dir):
    # global ground_truth_folder, reference_folder_parent, make_new_gt, ensembl_grch37_release93_folder

    cosmic_cdna_path = os.path.join(ensembl_grch37_release93_folder, "Homo_sapiens.GRCh37.cdna.all.fa")
    cosmic_genome_path = os.path.join(ensembl_grch37_release93_folder, "Homo_sapiens.GRCh37.dna.primary_assembly.fa")
    cosmic_gtf_path = os.path.join(ensembl_grch37_release93_folder, "Homo_sapiens.GRCh37.87.gtf")

    bowtie2_reference_genome_folder = os.path.join(ensembl_grch37_release93_folder, "bowtie_index_genome")
    bowtie2_reference_transcriptome_folder = os.path.join(ensembl_grch37_release93_folder, "bowtie_index_transcriptome")

    # skip this run if you don't have the ground truth and are not making it
    if not os.path.exists(ground_truth_folder) or not os.listdir(ground_truth_folder):
        if not make_new_gt:
            pytest.skip("Ground truth folder is missing or empty, and make_new_gt is False. Skipping test.")

    for path in [cosmic_csv_path, cosmic_cdna_path, cosmic_genome_path, cosmic_gtf_path]:
        if not os.path.isfile(path):
            pytest.skip(f"{path} not found. Please download it to continue")

    for directory in [bowtie2_reference_genome_folder, bowtie2_reference_transcriptome_folder]:
        if not os.path.isdir(directory) or len(os.listdir(directory)) == 0:
            pytest.skip(f"{directory} not found. Please make this bowtie2 index to continue")

    w = 47
    k = 51
    columns_to_include = "all"
    threads = 2
    filters=(
        "alignment_to_reference:is_not_true",
        # "substring_alignment_to_reference:is_not_true",  # filter out variants that are a substring of the reference genome  #* uncomment this and erase the line above when implementing d-list
        "pseudoaligned_to_reference_despite_not_truly_aligning:is_not_true",  # filter out variants that pseudoaligned to human genome despite not truly aligning
        "num_distinct_triplets:greater_than=2",  # filters out VCRSs with <= 2 unique triplets
    )

    if make_new_gt:
        os.makedirs(ground_truth_folder, exist_ok=True)

    vk.ref(
        variants = cosmic_csv_path,  # build args
        sequences = cosmic_cdna_path,
        out = out_dir,
        seq_id_column="seq_ID",
        var_column = "mutation_cdna",
        seq_id_cdna_column="seq_ID",
        var_cdna_column="mutation_cdna",
        seq_id_genome_column="chromosome",
        var_genome_column="mutation_genome",
        gene_name_column="gene_name",
        w = w,
        k = k,
        save_variants_updated_csv = True,
        columns_to_include = columns_to_include,  # info args
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


def test_parameter_values(toy_sequences_fasta_for_vk_ref, toy_variants_csv_for_vk_ref, out_dir):
    good_parameter_values_list_of_dicts = [
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "dlist_reference_source": "t2t"},
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "w": 27, "k": "31", "dlist_reference_source": "grch37", "minimum_info_columns": True},
    ]
    
    bad_parameter_values_list_of_dicts = [
        {"sequences": "fake_path.fa", "variants": toy_variants_csv_for_vk_ref, "out": out_dir},  # invalid sequences path
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": "fake_variants.fa", "out": out_dir},  # invalid variants path
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": 123},  # invalid out path
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "w": 54.1},  # float w
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "k": 55.1},  # float k
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "k": 56},  # even k
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "w": 59, "k": 55},  # w > k
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "filters": ["alignment_to_reference:is_not_true", "num_distinct_triplets:greater_than"]},  # bad filter rule (greater_than needs a VALUE)
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "dlist_reference_source": "invalid"},  # invalid dlist_reference_source
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "index_out": "index.fasta"},  # bad ext for index_out (expects .idx)
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "t2g_out": "t2g.fasta"},  # bad ext for t2g_out (expects .txt)
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "download": "yes"},  # download should be a boolean
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "threads": 0},  # threads should be a positive integer

    ]
    
    for parameter_dict in good_parameter_values_list_of_dicts:
        vk.ref(**parameter_dict, overwrite=True)

    for parameter_dict in bad_parameter_values_list_of_dicts:
        with pytest.raises(ValueError):
            vk.ref(**parameter_dict, overwrite=True)

def test_varseek_ref_on_command_line(monkeypatch):
    import varseek.main as main_module

    test_args = [
        "vk", "ref", 
        "-v", "cosmic_cmc",
        "-s", "cdna",
        "-k", "11",
        "--make_kat_histogram",
        "--dlist_cdna_fasta_out", "pretend_dlist_cdna_fasta_out_path",
        "--columns_to_include", "col1", "col2", "col3",
        "--save_logs",
        "--cosmic_version", "100",
        "--logging_level", "20",
        "--disable_optimize_flanking_regions",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setenv("TESTING", "true")  # so that main returns params_dict (there is a conditional in main to look for this environment variable)
    
    importlib.reload(main_module)
    params_dict = main_module.main()  # now returns params_dict because TESTING is set

    expected_dict = {
        'sequences': 'cdna',
        'variants': 'cosmic_cmc',
        'k': 11,
        'make_kat_histogram': True,
        'dlist_cdna_fasta_out': 'pretend_dlist_cdna_fasta_out_path',
        'columns_to_include': ['col1', 'col2', 'col3'],
        'save_logs': True,
        'cosmic_version': '100',
        'logging_level': '20',
        'optimize_flanking_regions': False
    }
    
    assert params_dict == expected_dict