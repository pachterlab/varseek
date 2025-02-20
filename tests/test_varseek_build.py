import os
import tempfile
import inspect
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import varseek as vk

store_out_in_permanent_paths = False
tests_dir = Path(__file__).resolve().parent
pytest_permanent_out_dir_base = tests_dir / "pytest_output" / Path(__file__).stem
current_datetime = datetime.now().strftime("date_%Y_%m_%d_time_%H%M_%S")

@pytest.fixture
def out_dir(tmp_path, request):
    """Fixture that returns the appropriate output directory for each test."""
    if store_out_in_permanent_paths:
        current_test_function_name = request.node.name
        out = Path(f"{pytest_permanent_out_dir_base}/{current_datetime}/{current_test_function_name}")
    else:
        out = tmp_path / "out_vk_build"

    out.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return out

@pytest.fixture
def long_sequence():
    return 'CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCG'

@pytest.fixture
def extra_long_sequence():
    return 'CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCG'

@pytest.fixture
def long_sequence_with_N():
    return 'CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCNCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCG'


@pytest.fixture
def create_temp_files(long_sequence):
    # Create a temporary CSV file
    temp_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    
    # Data to write to CSV
    mutation_list = ["c.35G>A", "c.65G>A", "c.35del", "c.4_5insT"]
    protein_mutation_list = ['A12T', 'A22T', 'A12del', 'A4_5insT']
    mut_ID_list = ['GENE1_MUT1A_MUT1B', 'GENE1_MUT2A_MUT2B', 'GENE2_MUT1A_MUT1B', 'GENE3_MUT1A_MUT1B']
    seq_ID_list = ['ENST1', 'ENST2', 'ENST3', 'ENST4']
    
    data = {
        'mutation': mutation_list,
        'mutation_aa': protein_mutation_list,
        'mut_ID': mut_ID_list,
        'seq_ID': seq_ID_list
    }

    df = pd.DataFrame(data)
    df.to_csv(temp_csv_file.name, index=False)

    # Create a temporary FASTA file
    sequence_list = [long_sequence for _ in range(len(mutation_list))]
    temp_fasta_file = tempfile.NamedTemporaryFile(delete=False, suffix='.fasta')
    
    with open(temp_fasta_file.name, 'w', encoding="utf-8") as fasta_file:
        for seq_id, sequence in zip(seq_ID_list, sequence_list):
            fasta_file.write(f">{seq_id}\n")
            fasta_file.write(f"{sequence}\n")
    
    yield temp_csv_file.name, temp_fasta_file.name
    
    # Cleanup
    os.remove(temp_csv_file.name)
    os.remove(temp_fasta_file.name)

def assert_global_variables_zero(number_intronic_position_mutations = 0, number_posttranslational_region_mutations = 0, number_uncertain_mutations = 0, number_ambiguous_position_mutations = 0, number_index_errors = 0):
    assert vk.varseek_build.intronic_mutations == number_intronic_position_mutations
    assert vk.varseek_build.posttranslational_region_mutations == number_posttranslational_region_mutations
    assert vk.varseek_build.uncertain_mutations == number_uncertain_mutations
    assert vk.varseek_build.ambiguous_position_mutations == number_ambiguous_position_mutations
    assert vk.varseek_build.mut_idx_outside_seq == number_index_errors


def test_single_substitution(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35G>A",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()

def test_single_substitution_near_right_end(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.65G>A",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG"

    assert_global_variables_zero()


def test_single_substitution_near_left_end(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.5G>A",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCACCCCACCCCGCCCCTCCCCGCCCCACCCCG"

    assert_global_variables_zero()


def test_single_deletion(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35del",  # del the G
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()


def test_multi_deletion(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35_40del",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCGCCCCACCCCGCCCCTCCCCGCCCCA"

    assert_global_variables_zero()

def test_single_deletion_with_right_repeats(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.31del",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CGCCCCACCCCGCCCCTCCCCGCCCCACCCGCCCCTCCCCGCCCCACCCCGCCCCTC"

    assert_global_variables_zero()

def test_single_deletion_with_left_repeats(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.34del",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CGCCCCACCCCGCCCCTCCCCGCCCCACCCGCCCCTCCCCGCCCCACCCCGCCCCTC"

    assert_global_variables_zero()

def test_multi_deletion_with_right_repeats(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.31_32del",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCGCCCCACCCCGCCCCTCCCCGCCCCACCGCCCCTCCCCGCCCCACCCCGCCCCTCC"

    assert_global_variables_zero()

def test_single_insertion(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.4_5insT",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCCC"

    assert_global_variables_zero()

def test_single_insertion_mid_sequence_small_w(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20_21insA", # --> 19_20 (index 0) --> start at 15, end at 24 (0-index positions, inclusive, from original sequence)
        w=5,
        return_variant_output=True,
        required_insertion_overlap_length=None,
        k=7,
        out=out_dir
    )

    # CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCG

    assert result[0] == "CCCCTACCCCG"

    assert_global_variables_zero()


def test_multi_insertion(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.65_66insTTTTT",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCTCCCCGCCCCACCCCGCCCCTCCCCGTTTTTCCCCACCCCG"

    assert_global_variables_zero()


def test_multi_insertion_with_left_repeats(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20_21insCCAAA",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCAAACCCCGCCCCACCCCGCCCCTCCCCGCCCCA"

    assert_global_variables_zero()


def test_single_delins(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.38delinsAAA",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGCCAAACTCCCCGCCCCACCCCGCCCCTCCCCGCCC"

    assert_global_variables_zero()


def test_multi_delins(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.38_40delinsAAA",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGCCAAACCCCGCCCCACCCCGCCCCTCCCCGCCCCA"

    assert_global_variables_zero()


def test_multi_delins_with_psuedo_left_repeats(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.36_37delinsAG",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCACCCCGCCCCTCCCCGCCCCACCCCGAGCCTCCCCGCCCCACCCCGCCCCTCCCCGCC"

    assert_global_variables_zero()

def test_multi_delins_with_true_left_repeats(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.36_37delinsAC",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCACCCCGCCCCTCCCCGCCCCACCCCGACCCTCCCCGCCCCACCCCGCCCCTCCCCGC"

    assert_global_variables_zero()


def test_multi_delins_with_true_right_repeats(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.36_37delinsCA",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCACCCCGCCCCTCCCCGCCCCACCCCGCACCTCCCCGCCCCACCCCGCCCCTCCCCGCC"

    assert_global_variables_zero()

def test_single_dup(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35dup",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCACCCCGCCCCTCCCCGCCCCACCCCGGCCCCTCCCCGCCCCACCCCGCCCCTCCCC"

    assert_global_variables_zero()

def test_multi_dup(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35_37dup",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGCCGCCCCTCCCCGCCCCACCCCGCCCCTCC"

    assert_global_variables_zero()

def test_inversion_with_overlaps(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35_38inv",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCACCCCGCCCCTCCCCGCCCCACCCCGGGCCTCCCCGCCCCACCCCGCCCCTCCCCGCC"

    assert_global_variables_zero()




def test_list_of_mutations(long_sequence, out_dir):
    mutation_list = ["c.35G>A", "c.65G>A", "c.35del", "c.4_5insT"]
    sequence_list = [long_sequence for _ in range(len(mutation_list))]
    
    result = vk.build(
        sequences=sequence_list,
        variants=mutation_list,
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG", "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCCC"]

    assert_global_variables_zero()


def test_csv_of_mutations(create_temp_files, out_dir):
    mutation_temp_csv_file, sequence_temp_fasta_path = create_temp_files

    result = vk.build(
        sequences=sequence_temp_fasta_path,
        variants=mutation_temp_csv_file,
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG", "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCCC"]

    assert_global_variables_zero()



def test_intron_mutation_plus(long_sequence, out_dir):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20+3T>A",
        out=out_dir
    )
    
    assert_global_variables_zero(number_intronic_position_mutations=1)

def test_intron_mutation_minus(long_sequence, out_dir):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20-3T>A",
        out=out_dir
    )

    assert_global_variables_zero(number_intronic_position_mutations=1)


def test_posttranslational_mutation(long_sequence, out_dir):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20*5T>A",
        out=out_dir)

    assert_global_variables_zero(number_posttranslational_region_mutations=1)


def test_uncertain_mutation(long_sequence, out_dir):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.?",
        out=out_dir)

    assert_global_variables_zero(number_uncertain_mutations=1)


def test_ambiguous_mutation(long_sequence, out_dir):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.(20_28)del",
        out=out_dir)

    assert_global_variables_zero(number_ambiguous_position_mutations=1)


def test_index_error(long_sequence, out_dir):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.99999999C>A",
        out=out_dir)

    assert_global_variables_zero(number_index_errors=1)


def test_mismatch_error(long_sequence, out_dir):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.2G>A",
        out=out_dir)
    
    assert vk.varseek_build.variants_incorrect_wt_base == 1   

    assert_global_variables_zero()


def test_large_w(extra_long_sequence, out_dir):
    result = vk.build(
        sequences=extra_long_sequence,
        optimize_flanking_regions = True,
        variants="c.40T>G",
        w=54,
        return_variant_output=True,
        required_insertion_overlap_length=None,
        k=55,
        out=out_dir
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCGCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCC"

    assert_global_variables_zero()


def test_large_min_seq_length(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35G>A",
        min_seq_len=100,
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result is None


def test_single_deletion_with_right_repeats_and_unoptimized_flanks(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = False,
        remove_seqs_with_wt_kmers = False,
        variants="c.31del",
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCGCCCCTCCCCGCCCCACCCCGCCCCTC"


def test_single_deletion_with_right_repeats_and_removing_seqs_with_wt_kmers(long_sequence, out_dir):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = False,
        variants="c.31del",
        remove_seqs_with_wt_kmers = True,
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result is None


def test_sequence_with_N(long_sequence_with_N, out_dir):
    result = vk.build(
        sequences=long_sequence_with_N,
        optimize_flanking_regions = True,
        variants="c.35G>A",
        max_ambiguous = 0,
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result is None



def test_semicolon_merging(long_sequence, out_dir):
    mutation_list = ["c.35G>A", "c.35G>A"]
    sequence_list = [long_sequence, f"{long_sequence}AAAAAAA"]
    
    result = vk.build(
        sequences=sequence_list,
        variants=mutation_list,
        merge_identical=True,
        return_variant_output=True,
        required_insertion_overlap_length=None,
        w=30,
        k=31,
        out=out_dir
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG"]

    assert_global_variables_zero()


# def test_translation(long_sequence_with_N, out_dir):
#     result = vk.build(
#         sequences=long_sequence_with_N,
#         optimize_flanking_regions = True,
#         variants="c.35G>A",
#         translate = True,
#         save_variants_updated_csv = True,
#         store_full_sequences=True,
#         return_variant_output=True,
#         required_insertion_overlap_length=None,
#         w=30,
#         k=31,
#         out=out_dir
#     )

#     assert result[0] == "APPRPSPPHPTPPRPTPPLP"  #* translate is not returned by vk build; only stored in update_df

#     assert_global_variables_zero()

def test_parameter_values(toy_sequences_fasta_for_vk_ref, toy_variants_csv_for_vk_ref, out_dir):
    good_parameter_values_list_of_dicts = [
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir},
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "w": 27, "k": "31", "dlist_reference_source": "grch37", "minimum_info_columns": True},
    ]
    
    bad_parameter_values_list_of_dicts = [
        {"sequences": "fake_path.fa", "variants": toy_variants_csv_for_vk_ref, "out": out_dir},  # invalid sequences path
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": "fake_variants.fa", "out": out_dir},  # invalid variants path
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": 123},  # invalid out path
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "w": 54.1},  # float w
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "k": 55.1},  # float k
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "w": 59, "k": 55},  # w > k

    ]
    
    for parameter_dict in good_parameter_values_list_of_dicts:
        vk.build(**parameter_dict, overwrite=True)

    for parameter_dict in bad_parameter_values_list_of_dicts:
        with pytest.raises(ValueError):
            vk.build(**parameter_dict, overwrite=True)