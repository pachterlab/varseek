import pytest
import varseek as vk
import pandas as pd
import os
import tempfile

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
    
    with open(temp_fasta_file.name, 'w') as fasta_file:
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


def test_single_substitution(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.35G>A",
        id_to_header_csv_out=None
    )

    assert result[0] == "GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()

def test_single_substitution_near_right_end(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.65G>A",
        id_to_header_csv_out=None
    )

    assert result[0] == "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG"

    assert_global_variables_zero()


def test_single_substitution_near_left_end(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.5G>A",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCACCCCACCCCGCCCCTCCCCGCCCCACCCCG"

    assert_global_variables_zero()


def test_single_deletion(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.35del",  # del the G
        id_to_header_csv_out=None
    )

    assert result[0] == "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()


def test_multi_deletion(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.35_40del",
        id_to_header_csv_out=None
    )

    assert result[0] == "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCGCCCCACCCCGCCCCTCCCCGCCCCA"

    assert_global_variables_zero()

def test_single_deletion_with_right_repeats(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.31del",
        id_to_header_csv_out=None
    )

    assert result[0] == "CGCCCCACCCCGCCCCTCCCCGCCCCACCCGCCCCTCCCCGCCCCACCCCGCCCCTC"

    assert_global_variables_zero()

def test_single_deletion_with_left_repeats(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.34del",
        id_to_header_csv_out=None
    )

    assert result[0] == "CGCCCCACCCCGCCCCTCCCCGCCCCACCCGCCCCTCCCCGCCCCACCCCGCCCCTC"

    assert_global_variables_zero()

def test_multi_deletion_with_right_repeats(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.31_32del",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCGCCCCACCCCGCCCCTCCCCGCCCCACCGCCCCTCCCCGCCCCACCCCGCCCCTCC"

    assert_global_variables_zero()

def test_single_insertion(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.4_5insT",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCC"

    assert_global_variables_zero()

def test_single_insertion_mid_sequence_small_w(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.20_21insA", # --> 19_20 (index 0) --> start at 15, end at 24 (0-index positions, inclusive, from original sequence)
        w=5,
        id_to_header_csv_out=None
    )

    # CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCG

    assert result[0] == "CCCTACCCC"

    assert_global_variables_zero()


def test_multi_insertion(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.65_66insTTTTT",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCCCGTTTTTCCCCACCCCG"

    assert_global_variables_zero()


def test_multi_insertion_with_left_repeats(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.20_21insCCAAA",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCAAACCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()


def test_single_delins(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.38delinsAAA",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCGCCCCTCCCCGCCCCACCCCGCCAAACTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()


def test_multi_delins(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.38_40delinsAAA",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCGCCCCTCCCCGCCCCACCCCGCCAAACCCCGCCCCACCCCGCCCCTCCCCGCC"

    assert_global_variables_zero()


def test_multi_delins_with_psuedo_left_repeats(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.36_37delinsAG",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGAGCCTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()

def test_multi_delins_with_true_left_repeats(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.36_37delinsAC",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGACCCTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()


def test_multi_delins_with_true_right_repeats(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.36_37delinsCA",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGCACCTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()

def test_single_dup(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.35dup",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCACCCCGCCCCTCCCCGCCCCACCCCGGCCCCTCCCCGCCCCACCCCGCCCCTCCCC"

    assert_global_variables_zero()

def test_multi_dup(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.35_37dup",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGCCGCCCCTCCCCGCCCCACCCCGCCCCTCC"

    assert_global_variables_zero()

def test_inversion_with_overlaps(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.35_38inv",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCACCCCGCCCCTCCCCGCCCCACCCCGGGCCTCCCCGCCCCACCCCGCCCCTCCCCGCC"

    assert_global_variables_zero()




def test_list_of_mutations(long_sequence):
    mutation_list = ["c.35G>A", "c.65G>A", "c.35del", "c.4_5insT"]
    sequence_list = [long_sequence for _ in range(len(mutation_list))]
    
    result = vk.build(
        sequences=sequence_list,
        mutations=mutation_list,
        id_to_header_csv_out=None
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG", "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCCC"]

    assert_global_variables_zero()


def test_csv_of_mutations(create_temp_files):
    mutation_temp_csv_file, sequence_temp_fasta_path = create_temp_files

    result = vk.build(
        sequences=sequence_temp_fasta_path,
        mutations=mutation_temp_csv_file,
        id_to_header_csv_out=None
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG", "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCCC"]

    assert_global_variables_zero()



def test_intron_mutation_plus(long_sequence):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.20+3T>A",
        id_to_header_csv_out=None)
    
    assert_global_variables_zero(number_intronic_position_mutations=1)

def test_intron_mutation_minus(long_sequence):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.20-3T>A",
        id_to_header_csv_out=None)

    assert_global_variables_zero(number_intronic_position_mutations=1)


def test_posttranslational_mutation(long_sequence):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.20*5T>A",
        id_to_header_csv_out=None)

    assert_global_variables_zero(number_posttranslational_region_mutations=1)


def test_uncertain_mutation(long_sequence):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.?",
        id_to_header_csv_out=None)

    assert_global_variables_zero(number_uncertain_mutations=1)


def test_ambiguous_mutation(long_sequence):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.(20_28)del",
        id_to_header_csv_out=None)

    assert_global_variables_zero(number_ambiguous_position_mutations=1)


def test_index_error(long_sequence):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.99999999C>A",
        id_to_header_csv_out=None)

    assert_global_variables_zero(number_index_errors=1)


def test_mismatch_error(long_sequence):
    vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.2G>A",
        id_to_header_csv_out=None)
    
    assert vk.varseek_build.cosmic_incorrect_wt_base == 1   

    assert_global_variables_zero()


def test_large_w(extra_long_sequence):
    result = vk.build(
        sequences=extra_long_sequence,
        optimize_flanking_regions = True,
        mutations="c.40T>G",
        w=54,
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCGCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCC"

    assert_global_variables_zero()


def test_large_min_seq_length(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = True,
        mutations="c.35G>A",
        min_seq_len=100,
        id_to_header_csv_out=None
    )

    assert result is None


def test_single_deletion_with_right_repeats_and_unoptimized_flanks(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = False,
        mutations="c.31del",
        id_to_header_csv_out=None
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCGCCCCTCCCCGCCCCACCCCGCCCCTC"


def test_single_deletion_with_right_repeats_and_removing_seqs_with_wt_kmers(long_sequence):
    result = vk.build(
        sequences=long_sequence,
        optimize_flanking_regions = False,
        mutations="c.31del",
        remove_seqs_with_wt_kmers = True,
        id_to_header_csv_out=None
    )

    assert result is None


def test_sequence_with_N(long_sequence_with_N):
    result = vk.build(
        sequences=long_sequence_with_N,
        optimize_flanking_regions = True,
        mutations="c.35G>A",
        max_ambiguous = 0,
        id_to_header_csv_out=None
    )

    assert result is None



def test_semicolon_merging(long_sequence):
    mutation_list = ["c.35G>A", "c.35G>A"]
    sequence_list = [long_sequence, f"{long_sequence}AAAAAAA"]
    
    result = vk.build(
        sequences=sequence_list,
        mutations=mutation_list,
        merge_identical=True,
        id_to_header_csv_out=None
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG"]

    assert_global_variables_zero()


# def test_translation(long_sequence_with_N):
#     result = vk.build(
#         sequences=long_sequence_with_N,
#         optimize_flanking_regions = True,
#         mutations="c.35G>A",
#         translate = True,
#         update_df = True,
#         store_full_sequences=True,
#         id_to_header_csv_out=None
#     )

#     assert result[0] == "APPRPSPPHPTPPRPTPPLP"  #* translate is not returned by gget mutate; only stored in update_df

#     assert_global_variables_zero()