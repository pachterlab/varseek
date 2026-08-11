import os
import tempfile
import inspect
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import varseek as vk


@pytest.fixture(autouse=True)
def _disable_alignment_to_reference_filter(monkeypatch):
    """Opt every vk.build call in this file out of two on-by-default behaviors.

    remove_alignment_to_reference (default True) pseudoaligns each VCRS against a normal genome
    reference to drop false positives, which requires a genome fasta / downloadable species that
    these toy tests do not provide. It is exercised in test_varseek_ref.py.

    shorten_repetitive_regions (default True) shaves flanks wherever the variant abuts a tandem
    repeat, and the toy fixture sequences here are near-pure repeat ('CCCCGCCCCACCCC...'), so it
    fires on nearly every case and obscures the flank arithmetic these tests are pinning down. It
    has its own dedicated tests below, including test_shorten_repetitive_regions_is_on_by_default.

    Both are only defaulted off here, so any test that names either one explicitly still wins."""
    original_build = vk.build

    def build_without_alignment_filter(*args, **kwargs):
        kwargs.setdefault("remove_alignment_to_reference", False)
        kwargs.setdefault("shorten_repetitive_regions", False)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(vk, "build", build_without_alignment_filter)


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
def create_temp_files(long_sequence):  # do not generalize this for temp files - use tmp_path instead
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
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35G>A",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()

def test_single_substitution_near_right_end(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.65G>A",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG"

    assert_global_variables_zero()


def test_single_substitution_near_left_end(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.5G>A",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCACCCCACCCCGCCCCTCCCCGCCCCACCCCG"

    assert_global_variables_zero()


def test_single_deletion(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35del",  # del the G
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG"

    assert_global_variables_zero()


def test_multi_deletion(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35_40del",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCGCCCCACCCCGCCCCTCCCCGCCCCA"

    assert_global_variables_zero()

def test_single_deletion_with_right_repeats(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.31del",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CGCCCCACCCCGCCCCTCCCCGCCCCACCCGCCCCTCCCCGCCCCACCCCGCCCCTC"

    assert_global_variables_zero()

def test_single_deletion_with_left_repeats(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.34del",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CGCCCCACCCCGCCCCTCCCCGCCCCACCCGCCCCTCCCCGCCCCACCCCGCCCCTC"

    assert_global_variables_zero()

def test_multi_deletion_with_right_repeats(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.31_32del",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCGCCCCACCCCGCCCCTCCCCGCCCCACCGCCCCTCCCCGCCCCACCCCGCCCCTCC"

    assert_global_variables_zero()

def test_shorten_repetitive_regions_helpers():
    from varseek.utils import repeat_length_at_sequence_end, repeat_length_at_sequence_start

    # a repeat unit must occur at least twice, and only units of length 1, 2, and 3 are considered
    assert repeat_length_at_sequence_end("CATTATT") == 6  # 'TT' (2), no doublet, 'ATTATT' (6)
    assert repeat_length_at_sequence_end("TTTT") == 4
    assert repeat_length_at_sequence_end("ACGTATATA") == 6
    assert repeat_length_at_sequence_end("GCTAGTAGTAG") == 9
    assert repeat_length_at_sequence_end("ACGT") == 0
    assert repeat_length_at_sequence_end("") == 0

    assert repeat_length_at_sequence_start("TTATTAG") == 6
    assert repeat_length_at_sequence_start("TAGTAGTAGC") == 9
    assert repeat_length_at_sequence_start("TTAATTAATTAAC") == 2  # 'TTAA' is a unit of length 4
    assert repeat_length_at_sequence_start("ACGT") == 0


def test_repeat_lengths_at_sequence_ends_matches_scalar_reference():
    """The build path uses the vectorized helper; the scalar pair is its readable spec, so pin them together."""
    import random

    from varseek.utils import repeat_length_at_sequence_end, repeat_length_at_sequence_start, repeat_lengths_at_sequence_ends

    random.seed(0)
    sequences = ["CATTATT", "TTTT", "ACGTATATA", "GCTAGTAGTAG", "ACGT", "", "A", "TT", "TTATTAG", "TAGTAGTAGC", "TTAATTAATTAAC"]
    for _ in range(500):  # unstructured
        sequences.append("".join(random.choice("ACGT") for _ in range(random.randint(0, 14))))
    for _ in range(500):  # repeat-rich, including units longer than the 3 that count
        unit = "".join(random.choice("AT") for _ in range(random.randint(1, 4)))
        sequences.append("".join(random.choice("ACGT") for _ in range(random.randint(0, 5))) + unit * random.randint(1, 6) + "".join(random.choice("ACGT") for _ in range(random.randint(0, 5))))

    for max_unit in (1, 2, 3, 5):
        assert list(repeat_lengths_at_sequence_ends(sequences, max_unit, at_end=True)) == [repeat_length_at_sequence_end(s, max_unit) for s in sequences]
        assert list(repeat_lengths_at_sequence_ends(sequences, max_unit, at_end=False)) == [repeat_length_at_sequence_start(s, max_unit) for s in sequences]

    # ragged input (flanks are short near a sequence boundary) must not let padding fake a repeat
    assert list(repeat_lengths_at_sequence_ends(["A", "TTTT", "", "ACGT"], at_end=True)) == [0, 4, 0, 0]
    assert list(repeat_lengths_at_sequence_ends([], at_end=True)) == []


def test_concatenate_with_trimmed_flanks_matches_rowwise_reference():
    from varseek.utils import concatenate_with_trimmed_flanks

    frame = pd.DataFrame({
        "left_flank_region": ["AAAA", "CCCC", "GGGG", "TTTT"],
        "mut_nucleotides": ["X", "X", "", "X"],
        "right_flank_region": ["AAAA", "CCCC", "GGGG", "TTTT"],
        "updated_left_flank_start": [1, 2, 0, 4],
        "updated_right_flank_end": [0, 1, 2, 4],
    })

    def rowwise(row):
        return row["left_flank_region"][row["updated_left_flank_start"]:] + row["mut_nucleotides"] + row["right_flank_region"][: len(row["right_flank_region"]) - row["updated_right_flank_end"]]

    for index in ([0, 1, 2, 3], [0, 0, 1, 1]):  # a duplicate-label index must not cross-assign rows
        reindexed = frame.set_axis(index)
        result = concatenate_with_trimmed_flanks(reindexed["left_flank_region"], reindexed["mut_nucleotides"], reindexed["right_flank_region"], reindexed["updated_left_flank_start"], reindexed["updated_right_flank_end"])
        assert list(result) == list(reindexed.apply(rowwise, axis=1))
    # last row is trimmed away entirely on both sides, leaving just the variant nucleotides
    assert list(result) == ["AAA" + "X" + "AAAA", "CC" + "X" + "CCC", "GGGG" + "" + "GG", "" + "X" + ""]


def test_shorten_repetitive_regions_left_repeat_shortens_right_flank(out_dir):
    # left flank ends with 'CATTATT' -> longest repeat is 'ATTATT' (6), so 5 nucleotides come off the right flank
    sequence = "GATCCAGTAC" + "CATTATT" + "C" + "GACTCAGTCAG"
    build_kwargs = dict(dont_create_index=True, sequences=sequence, variants="c.18del", return_variant_output=True, w=10, k=11, out=out_dir, overwrite=True)

    assert vk.build(shorten_repetitive_regions=False, **build_kwargs)[0] == "TACCATTATT" + "GACTCAGTCA"
    assert vk.build(shorten_repetitive_regions=True, **build_kwargs)[0] == "TACCATTATT" + "GACTC"


def test_shorten_repetitive_regions_right_repeat_shortens_left_flank(out_dir):
    # right flank starts with 'TTATTAG' -> longest repeat is 'TTATTA' (6), so 5 nucleotides come off the left flank
    # (the final nucleotide of the right flank is removed by optimize_flanking_regions in both cases)
    sequence = "GATCCAGTAC" + "C" + "TTATTAG" + "ACGTCAGTC"
    build_kwargs = dict(dont_create_index=True, sequences=sequence, variants="c.11del", return_variant_output=True, w=10, k=11, out=out_dir, overwrite=True)

    assert vk.build(shorten_repetitive_regions=False, **build_kwargs)[0] == "GATCCAGTAC" + "TTATTAGAC"
    assert vk.build(shorten_repetitive_regions=True, **build_kwargs)[0] == "AGTAC" + "TTATTAGAC"


def test_shorten_repetitive_regions_insertion_capped_at_flank_length(out_dir):
    # left flank ends with 'TAGTAGTAG' (9) -> 8 nucleotides come off the right flank
    sequence = "CCAGT" + "TAGTAGTAG" + "ACGTCAGTCAGGA"
    build_kwargs = dict(dont_create_index=True, sequences=sequence, variants="c.14_15insTT", return_variant_output=True, w=10, k=11, out=out_dir, overwrite=True)

    assert vk.build(shorten_repetitive_regions=False, **build_kwargs)[0] == "TTAGTAGTAG" + "TT" + "ACGTCAGTCA"
    assert vk.build(shorten_repetitive_regions=True, **build_kwargs)[0] == "TTAGTAGTAG" + "TT" + "AC"

    # the right flank here is shorter than the 5 nucleotides the left repeat would remove -> remove all of it
    short_right_flank_sequence = "GATCCAGTAC" + "CATTATT" + "C" + "GAC"
    assert vk.build(dont_create_index=True, sequences=short_right_flank_sequence, variants="c.18del", shorten_repetitive_regions=True, min_seq_len=None, remove_seqs_with_wt_kmers=False, return_variant_output=True, w=10, k=11, out=out_dir, overwrite=True)[0] == "TACCATTATT"


def test_shorten_repetitive_regions_applies_to_substitutions(out_dir):
    # left flank ends with 'CATTATT' -> 'ATTATT' (6), so 5 nucleotides come off the right flank; the right flank
    # starts with 'GACTCAGTCA', which holds no repeat, so the left flank is untouched
    sequence = "GATCCAGTAC" + "CATTATT" + "C" + "GACTCAGTCAG"
    build_kwargs = dict(dont_create_index=True, sequences=sequence, variants="c.18C>G", return_variant_output=True, w=10, k=11, out=out_dir, overwrite=True)

    assert vk.build(shorten_repetitive_regions=False, **build_kwargs)[0] == "TACCATTATT" + "G" + "GACTCAGTCA"
    assert vk.build(shorten_repetitive_regions=True, **build_kwargs)[0] == "TACCATTATT" + "G" + "GACTC"


def test_shorten_repetitive_regions_substitution_between_two_repeats(out_dir):
    # both flanks abut a repeat: left flank ends with 'TATATA' (6) -> 5 off the right flank, right flank starts
    # with 'TTT' (3) -> 2 off the left flank
    sequence = "GCAGT" + "TATATA" + "C" + "TTT" + "GACCAGTCAG"
    build_kwargs = dict(dont_create_index=True, sequences=sequence, variants="c.12C>G", return_variant_output=True, w=10, k=11, out=out_dir, overwrite=True)

    assert vk.build(shorten_repetitive_regions=False, **build_kwargs)[0] == "CAGTTATATA" + "G" + "TTTGACCAGT"
    assert vk.build(shorten_repetitive_regions=True, **build_kwargs)[0] == "GTTATATA" + "G" + "TTTGA"


def test_shorten_repetitive_regions_is_on_by_default():
    # the autouse fixture above defaults this off for the rest of the file, so pin the real default
    # here (both the Python signature and the CLI, which exposes it as an opt-OUT flag)
    import argparse

    import varseek.main as main_module

    assert inspect.signature(vk.varseek_build.build).parameters["shorten_repetitive_regions"].default is True

    parser = argparse.ArgumentParser()
    main_module.add_build_arguments(parser, required=False)
    (action,) = [a for a in parser._actions if a.dest == "shorten_repetitive_regions"]
    assert action.option_strings == ["--disable_shorten_repetitive_regions"]
    assert isinstance(action, argparse._StoreFalseAction)


def test_shorten_repetitive_regions_default_shortens_end_to_end(out_dir, monkeypatch):
    # same case as test_shorten_repetitive_regions_applies_to_substitutions, but with the argument
    # omitted entirely, so it goes through vk.build's own default rather than an explicit True
    monkeypatch.setattr(vk, "build", vk.varseek_build.build)
    sequence = "GCTACCATTATT" + "C" + "GACTCAGTCAG"

    result = vk.build(dont_create_index=True, sequences=sequence, variants="c.13C>G", return_variant_output=True, w=10, k=11, out=out_dir, overwrite=True, remove_alignment_to_reference=False)

    assert result[0] == "TACCATTATT" + "G" + "GACTC"


def test_single_insertion(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.4_5insT",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCCC"

    assert_global_variables_zero()

def test_single_insertion_mid_sequence_small_w(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20_21insA", # --> 19_20 (index 0) --> start at 15, end at 24 (0-index positions, inclusive, from original sequence)
        w=5,
        return_variant_output=True,
        k=7,
        out=out_dir
    )

    # CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCG

    assert result[0] == "CCCCTACCCCG"

    assert_global_variables_zero()


def test_multi_insertion(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.65_66insTTTTT",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCTCCCCGCCCCACCCCGCCCCTCCCCGTTTTTCCCCACCCCG"

    assert_global_variables_zero()


def test_multi_insertion_with_left_repeats(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20_21insCCAAA",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCAAACCCCGCCCCACCCCGCCCCTCCCCGCCCCA"

    assert_global_variables_zero()


def test_single_delins(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.38delinsAAA",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGCCAAACTCCCCGCCCCACCCCGCCCCTCCCCGCCC"

    assert_global_variables_zero()


def test_multi_delins(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.38_40delinsAAA",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGCCAAACCCCGCCCCACCCCGCCCCTCCCCGCCCCA"

    assert_global_variables_zero()


def test_multi_delins_with_psuedo_left_repeats(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.36_37delinsAG",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCACCCCGCCCCTCCCCGCCCCACCCCGAGCCTCCCCGCCCCACCCCGCCCCTCCCCGCC"

    assert_global_variables_zero()

def test_multi_delins_with_true_left_repeats(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.36_37delinsAC",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCACCCCGCCCCTCCCCGCCCCACCCCGACCCTCCCCGCCCCACCCCGCCCCTCCCCGC"

    assert_global_variables_zero()


def test_multi_delins_with_true_right_repeats(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.36_37delinsCA",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCACCCCGCCCCTCCCCGCCCCACCCCGCACCTCCCCGCCCCACCCCGCCCCTCCCCGCC"

    assert_global_variables_zero()

def test_single_dup(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35dup",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCACCCCGCCCCTCCCCGCCCCACCCCGGCCCCTCCCCGCCCCACCCCGCCCCTCCCC"

    assert_global_variables_zero()

def test_multi_dup(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35_37dup",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCACCCCGCCCCTCCCCGCCCCACCCCGCCGCCCCTCCCCGCCCCACCCCGCCCCTCC"

    assert_global_variables_zero()

def test_inversion_with_overlaps(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35_38inv",
        return_variant_output=True,
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
        dont_create_index=True,
        sequences=sequence_list,
        variants=mutation_list,
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG", "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCCC"]

    assert_global_variables_zero()


def test_csv_of_mutations(create_temp_files, out_dir):
    mutation_temp_csv_file, sequence_temp_fasta_path = create_temp_files

    result = vk.build(
        dont_create_index=True,
        sequences=sequence_temp_fasta_path,
        variants=mutation_temp_csv_file,
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG", "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCCC"]

    assert_global_variables_zero()

def test_csv_of_mutations_with_chunks(create_temp_files, out_dir):
    mutation_temp_csv_file, sequence_temp_fasta_path = create_temp_files

    result = vk.build(
        dont_create_index=True,
        sequences=sequence_temp_fasta_path,
        variants=mutation_temp_csv_file,
        w=30,
        k=31,
        out=out_dir,
        merge_identical=False,
        chunksize=2
    )

    fasta_file_path = out_dir / "vcrs.fa"
    with fasta_file_path.open() as f:
        sequences = [line.strip() for line in f if line.strip() and not line.startswith(">")]

    assert sequences == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "GCCCCTCCCCGCCCCACCCCGCCCCTCCCCACCCCACCCCG", "GCCCCACCCCGCCCCTCCCCGCCCCACCCCCCCCTCCCCGCCCCACCCCGCCCCTCCCCG", "CCCCTGCCCCACCCCGCCCCTCCCCGCCCCACCCC"]

    assert_global_variables_zero()



def test_intron_mutation_plus(long_sequence, out_dir):
    vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20+3T>A",
        out=out_dir
    )
    
    assert_global_variables_zero(number_intronic_position_mutations=1)

def test_intron_mutation_minus(long_sequence, out_dir):
    vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20-3T>A",
        out=out_dir
    )

    assert_global_variables_zero(number_intronic_position_mutations=1)


def test_posttranslational_mutation(long_sequence, out_dir):
    vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.20*5T>A",
        out=out_dir)

    assert_global_variables_zero(number_posttranslational_region_mutations=1)


def test_uncertain_mutation(long_sequence, out_dir):
    vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.?",
        out=out_dir)

    assert_global_variables_zero(number_uncertain_mutations=1)


def test_ambiguous_mutation(long_sequence, out_dir):
    vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.(20_28)del",
        out=out_dir)

    assert_global_variables_zero(number_ambiguous_position_mutations=1)


def test_index_error(long_sequence, out_dir):
    vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.99999999C>A",
        out=out_dir)

    assert_global_variables_zero(number_index_errors=1)


def test_mismatch_error(long_sequence, out_dir):
    vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.2G>A",
        out=out_dir)
    
    assert vk.varseek_build.variants_incorrect_wt_base == 1   

    assert_global_variables_zero()


def test_large_w(extra_long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=extra_long_sequence,
        optimize_flanking_regions = True,
        variants="c.40T>G",
        w=54,
        return_variant_output=True,
        k=55,
        out=out_dir
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCGCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCC"

    assert_global_variables_zero()


def test_large_min_seq_length(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = True,
        variants="c.35G>A",
        min_seq_len=100,
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result is None


def test_single_deletion_with_right_repeats_and_unoptimized_flanks(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = False,
        remove_seqs_with_wt_kmers = False,
        variants="c.31del",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result[0] == "CCCCGCCCCACCCCGCCCCTCCCCGCCCCACCCGCCCCTCCCCGCCCCACCCCGCCCCTC"


def test_single_deletion_with_right_repeats_and_removing_seqs_with_wt_kmers(long_sequence, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence,
        optimize_flanking_regions = False,
        variants="c.31del",
        remove_seqs_with_wt_kmers = True,
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result is None


def test_sequence_with_N(long_sequence_with_N, out_dir):
    result = vk.build(
        dont_create_index=True,
        sequences=long_sequence_with_N,
        optimize_flanking_regions = True,
        variants="c.35G>A",
        max_ambiguous = 0,
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result is None



def test_semicolon_merging(long_sequence, out_dir):
    mutation_list = ["c.35G>A", "c.35G>A"]
    sequence_list = [long_sequence, f"{long_sequence}AAAAAAA"]
    
    result = vk.build(
        dont_create_index=True,
        sequences=sequence_list,
        variants=mutation_list,
        merge_identical=True,
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG"]

    assert_global_variables_zero()



def test_parameter_values(toy_sequences_fasta_for_vk_ref, toy_variants_csv_for_vk_ref, out_dir):
    good_parameter_values_list_of_dicts = [
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir},
        {"sequences": toy_sequences_fasta_for_vk_ref, "variants": toy_variants_csv_for_vk_ref, "out": out_dir, "w": 27, "k": "31"},
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
        vk.build(**parameter_dict, overwrite=True, dont_create_index=True)

    for parameter_dict in bad_parameter_values_list_of_dicts:
        with pytest.raises(ValueError):
            vk.build(**parameter_dict, overwrite=True, dont_create_index=True)





def test_vcf(vcf_file_and_corresponding_sequences, out_dir):
    vcf_file_path, sequences_fasta_path, vcf_output_ground_truth_df = vcf_file_and_corresponding_sequences

    _ = vk.build(
        dont_create_index=True,
        variants=vcf_file_path,
        sequences=sequences_fasta_path,
        out=out_dir,
        save_variants_updated_dataframe=True,
        overwrite=True,
        w=6,
        k=7,
        max_ambiguous=999,
        optimize_flanking_regions=False,
        remove_seqs_with_wt_kmers=False,
        min_seq_len=None,
        merge_identical=False
    )

    vcf_output_pytest_df = pd.read_csv(out_dir / "variants_updated.csv")

    assert vcf_output_pytest_df.equals(vcf_output_ground_truth_df)

    assert_global_variables_zero()

def test_vcf_chunks(vcf_file_and_corresponding_sequences, out_dir):
    vcf_file_path, sequences_fasta_path, vcf_output_ground_truth_df = vcf_file_and_corresponding_sequences

    _ = vk.build(
        dont_create_index=True,
        variants=vcf_file_path,
        sequences=sequences_fasta_path,
        out=out_dir,
        save_variants_updated_dataframe=True,
        overwrite=True,
        w=6,
        k=7,
        max_ambiguous=999,
        optimize_flanking_regions=False,
        remove_seqs_with_wt_kmers=False,
        min_seq_len=None,
        merge_identical=False,
        chunksize=3
    )

    vcf_output_pytest_df = pd.read_csv(out_dir / "variants_updated.csv")

    assert vcf_output_pytest_df.equals(vcf_output_ground_truth_df)

    assert_global_variables_zero()


def test_vcf_helper_derivation_edge_cases(tmp_path):
    """Unit-level checks on the VCF->HGVS helpers for edge cases not covered by the oracle fixture."""
    from varseek.utils.seq_utils import (
        add_variant_type_column_to_vcf_derived_df,
        add_variant_column_to_vcf_derived_df,
        vcf_to_dataframe,
    )

    # A POS==1 deletion can be left-anchored (ALT==REF[0], keep the first base, delete positions 2..N)
    # or right-anchored (ALT==REF[-1], keep the last base, delete positions 1..N-1). Both must be derived
    # correctly; previously every POS==1 deletion was treated as right-anchored.
    df = pd.DataFrame({
        "POS":  [1,         1,         1,     1,     40,    40],
        "REF":  ["TCATC",   "TCATC",   "TC",  "TC",  "AG",  "AGCAT"],
        "ALT":  ["T",       "C",       "T",   "C",   "A",   "A"],
    })
    add_variant_type_column_to_vcf_derived_df(df)
    add_variant_column_to_vcf_derived_df(df, var_column="mutation")
    expected = ["g.2_5del", "g.1_4del", "g.2del", "g.1del", "g.41del", "g.41_44del"]
    assert list(df["mutation"]) == expected

    # FILTER must be a plain string (regression: it was wrapped in a 1-tuple).
    vcf_text = (
        '##fileformat=VCFv4.0\n'
        '##FILTER=<ID=q10,Description="x">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
        '19\t11\tv1\tA\tC\t30\tPASS\t.\n'
    )
    vcf_path = tmp_path / "one.vcf"
    vcf_path.write_text(vcf_text)
    d = vcf_to_dataframe(str(vcf_path), additional_columns=True)
    assert isinstance(d["FILTER"].iloc[0], str) and d["FILTER"].iloc[0] == "PASS"

    # A header-only VCF must not crash; it returns an empty frame with the expected columns.
    empty_path = tmp_path / "empty.vcf"
    empty_path.write_text(
        '##fileformat=VCFv4.0\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
    )
    d_empty = vcf_to_dataframe(str(empty_path), additional_columns=True)
    assert d_empty.empty and {"CHROM", "POS", "REF", "ALT"}.issubset(d_empty.columns)




@pytest.fixture
def toy_vcrs_fa_path_for_merge_testing(tmp_path):
    # Create a temporary FASTA file
    temp_fasta_file = tmp_path / "toy_vcrs_for_merging.fasta"

    vcrs_header_list = ["header1", "header2", "header3", "header4", "aheader5", "header6"]
    vcrs_sequence_list = ["AAAAAA", "AAAAAA", "GGGGGGG", "CCCCCCC", "AAAAAA", "CCCCCCC"]

    with open(str(temp_fasta_file), 'w', encoding="utf-8") as fasta_file:
        for i in range(len(vcrs_sequence_list)):
            fasta_file.write(f">{vcrs_header_list[i]}\n")
            fasta_file.write(f"{vcrs_sequence_list[i]}\n")
        
    return str(temp_fasta_file)

from varseek.utils import merge_fasta_file_headers, create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting
def test_fasta_merging_with_awk(toy_vcrs_fa_path_for_merge_testing, tmp_path):
    merge_fasta_file_headers(toy_vcrs_fa_path_for_merge_testing, use_IDs=False)
    
    fasta_dict_test = dict(create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(toy_vcrs_fa_path_for_merge_testing))
    fasta_dict_gt = {"aheader5;header1;header2": "AAAAAA", "header3": "GGGGGGG", "header4;header6": "CCCCCCC"}

    assert fasta_dict_test == fasta_dict_gt

def test_fasta_merging_with_awk_using_IDs(toy_vcrs_fa_path_for_merge_testing, tmp_path):
    id_to_header_csv_out = tmp_path / "id_to_header.csv"
    merge_fasta_file_headers(toy_vcrs_fa_path_for_merge_testing, use_IDs=True, id_to_header_csv_out=id_to_header_csv_out)
    
    fasta_dict_test = dict(create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(toy_vcrs_fa_path_for_merge_testing))
    fasta_dict_gt = {"vcrs_1": "AAAAAA", "vcrs_2": "CCCCCCC", "vcrs_3": "GGGGGGG"}  # the order of these might be a little funny, but it doesn't really matter

    assert fasta_dict_test == fasta_dict_gt

    id_to_header_df_test = pd.read_csv(id_to_header_csv_out)
    id_to_header_df_gt = pd.DataFrame({"vcrs_id": ["vcrs_1", "vcrs_2", "vcrs_3"], "vcrs_header": ["aheader5;header1;header2", "header4;header6", "header3"]})

    assert id_to_header_df_test.equals(id_to_header_df_gt)






def test_convert_variant_coordinates_genome_to_transcript(long_sequence, tmp_path, out_dir):
    """Genomic variants are projected onto a transcript (via a local GTF) before building.

    `long_sequence` is treated as a single-exon transcript ENST1 spanning genomic 1001..1074 (+ strand),
    so g.1035G>A maps to c.35G>A -- the same variant/output as test_single_substitution.
    """
    cdna_fasta = tmp_path / "cdna.fa"
    cdna_fasta.write_text(f">ENST1\n{long_sequence}\n")

    gtf = tmp_path / "single_exon.gtf"
    gtf.write_text('1\ttest\texon\t1001\t1074\t.\t+\t.\ttranscript_id "ENST1"; gene_id "G1";\n')

    variants_csv = tmp_path / "genomic_variants.csv"
    variants_csv.write_text("seq_ID,mutation\n1,g.1035G>A\n")

    result = vk.build(
        dont_create_index=True,
        sequences=str(cdna_fasta),
        variants=str(variants_csv),
        gtf=str(gtf),
        convert_variant_coordinates="genome_to_transcript",
        return_variant_output=True,
        w=30,
        k=31,
        out=out_dir,
    )

    assert result == ["GCCCCACCCCGCCCCTCCCCGCCCCACCCCACCCCTCCCCGCCCCACCCCGCCCCTCCCCG"]

    assert_global_variables_zero()


def test_convert_variant_coordinates_requires_gtf(long_sequence, out_dir):
    """Requesting coordinate conversion without a GTF is rejected up front."""
    with pytest.raises(Exception):
        vk.build(
            dont_create_index=True,
            sequences=long_sequence,
            variants="g.35G>A",
            convert_variant_coordinates="genome_to_transcript",
            w=30,
            k=31,
            out=out_dir,
        )


# ---------------------------------------------------------------------------
# add_gene_name_to_header: annotate HGVS-like headers with the gene name in
# parentheses (seq_ID(GENE):mutation) when a GTF is provided.
# ---------------------------------------------------------------------------

def _first_fasta_header(fasta_path):
    """Return the first header line (without the leading '>') of a FASTA file."""
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                return line[1:].strip()
    raise AssertionError(f"No header found in {fasta_path}")


@pytest.fixture
def transcript_gtf(tmp_path):
    """A minimal GTF mapping transcript ENST1 (and chromosome 1) to gene name 'ACT'."""
    gtf = tmp_path / "gene_name.gtf"
    gtf.write_text(
        '1\ttest\tgene\t1\t100\t.\t+\t.\tgene_id "G1"; gene_name "ACT";\n'
        '1\ttest\ttranscript\t1\t100\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1"; gene_name "ACT";\n'
        '1\ttest\texon\t1\t100\t.\t+\t.\tgene_id "G1"; transcript_id "ENST1"; gene_name "ACT";\n'
    )
    return str(gtf)


def test_gene_name_added_to_header_transcript(long_sequence, transcript_gtf, tmp_path, out_dir):
    """With a GTF, a transcript (ENST) variant header gains the gene name in parentheses."""
    cdna_fasta = tmp_path / "cdna.fa"
    cdna_fasta.write_text(f">ENST1\n{long_sequence}\n")

    variants_csv = tmp_path / "variants.csv"
    variants_csv.write_text("seq_ID,mutation\nENST1,c.35G>A\n")

    fasta_out = tmp_path / "vcrs.fa"
    vk.build(
        dont_create_index=True,
        sequences=str(cdna_fasta),
        variants=str(variants_csv),
        gtf=transcript_gtf,
        vcrs_fasta_out=str(fasta_out),
        w=30,
        k=31,
        out=out_dir,
        overwrite=True,
    )

    assert _first_fasta_header(fasta_out) == "ENST1(ACT):c.35G>A"
    assert_global_variables_zero()


def test_gene_name_opt_out(long_sequence, transcript_gtf, tmp_path, out_dir):
    """add_gene_name_to_header=False keeps the plain seq_ID:mutation header even with a GTF."""
    cdna_fasta = tmp_path / "cdna.fa"
    cdna_fasta.write_text(f">ENST1\n{long_sequence}\n")

    variants_csv = tmp_path / "variants.csv"
    variants_csv.write_text("seq_ID,mutation\nENST1,c.35G>A\n")

    fasta_out = tmp_path / "vcrs.fa"
    vk.build(
        dont_create_index=True,
        sequences=str(cdna_fasta),
        variants=str(variants_csv),
        gtf=transcript_gtf,
        add_gene_name_to_header=False,
        vcrs_fasta_out=str(fasta_out),
        w=30,
        k=31,
        out=out_dir,
        overwrite=True,
    )

    assert _first_fasta_header(fasta_out) == "ENST1:c.35G>A"
    assert_global_variables_zero()


def test_gene_name_no_gtf_leaves_plain_header(long_sequence, tmp_path, out_dir):
    """Without a GTF there is nothing to map from, so the header is left plain (default on)."""
    cdna_fasta = tmp_path / "cdna.fa"
    cdna_fasta.write_text(f">ENST1\n{long_sequence}\n")

    variants_csv = tmp_path / "variants.csv"
    variants_csv.write_text("seq_ID,mutation\nENST1,c.35G>A\n")

    fasta_out = tmp_path / "vcrs.fa"
    vk.build(
        dont_create_index=True,
        sequences=str(cdna_fasta),
        variants=str(variants_csv),
        vcrs_fasta_out=str(fasta_out),
        w=30,
        k=31,
        out=out_dir,
        overwrite=True,
    )

    assert _first_fasta_header(fasta_out) == "ENST1:c.35G>A"
    assert_global_variables_zero()


def test_compute_gene_name_series_transcript_and_genome(tmp_path):
    """Unit-test the GTF gene-name lookup for ENST (transcript) and chrom+pos (genome) variants."""
    from varseek.utils.varseek_build_utils import compute_gene_name_series_for_headers

    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        '1\ttest\tgene\t1000\t2000\t.\t+\t.\tgene_id "G1"; gene_name "ACT";\n'
        '1\ttest\ttranscript\t1000\t2000\t.\t+\t.\tgene_id "G1"; transcript_id "ENST00000000001"; gene_name "ACT";\n'
        '1\ttest\texon\t1000\t2000\t.\t+\t.\tgene_id "G1"; transcript_id "ENST00000000001"; gene_name "ACT";\n'
        '7\ttest\tgene\t500\t900\t.\t-\t.\tgene_id "G2"; gene_name "BRCA1";\n'
    )

    mutations = pd.DataFrame(
        {
            "seq_ID": ["ENST00000000001", "1", "7", "7", "1"],
            "mutation": ["c.10A>G", "g.1500T>C", "g.600G>A", "g.2000A>T", "g.999C>G"],
        }
    )

    gene_names = compute_gene_name_series_for_headers(mutations, "seq_ID", "mutation", str(gtf))

    assert list(gene_names) == [
        "ACT",      # ENST -> gene name
        "ACT",      # chr1:1500 inside 1000-2000
        "BRCA1",    # chr7:600 inside 500-900
        "",         # chr7:2000 outside 500-900
        "",         # chr1:999 just before the 1-based inclusive start (1000)
    ]


def test_strip_gene_name_from_seq_id_round_trip():
    """The parse-side helper recovers the bare seq_ID from an annotated header field."""
    from varseek.utils.seq_utils import strip_gene_name_from_seq_id

    assert strip_gene_name_from_seq_id("ENST00000123456(ACT)") == "ENST00000123456"
    assert strip_gene_name_from_seq_id("3(BRCA1)") == "3"
    assert strip_gene_name_from_seq_id("ENST00000123456") == "ENST00000123456"  # no annotation -> unchanged

    series = pd.Series(["ENST1(ACT)", "3(BRCA1)", "ENST9"])
    assert list(strip_gene_name_from_seq_id(series)) == ["ENST1", "3", "ENST9"]


def test_count_unique_kmers_local_window():
    from varseek.utils import count_unique_kmers, triplet_stats

    vcrs = "GTCCAGTAAA" + "G" + "AAAGTCAGCT"  # 21 nt, variant at the center base (index 10)

    # local window around the center base: 'AAAGAAA' -> {AAA, AAG, AGA, GAA}
    assert count_unique_kmers(vcrs, k=3, max_bases_left=3, max_bases_right=3) == 4
    # no window given -> the whole VCRS, matching the global triplet count
    assert count_unique_kmers(vcrs, k=3) == triplet_stats(vcrs)[0] == 14
    assert count_unique_kmers("ACGT", k=5) == 0  # shorter than k


def test_min_unique_triplets_local(out_dir):
    # the variant sits in a local AAA...AAA stretch (few distinct triplets nearby) inside an otherwise complex VCRS
    sequence = "GTCCAGT" + "AAA" + "C" + "AAA" + "GTCAGCT"
    vcrs = "GTCCAGTAAA" + "G" + "AAAGTCAGCT"
    build_kwargs = dict(dont_create_index=True, sequences=sequence, variants="c.11C>G", return_variant_output=True, w=10, k=11, out=out_dir, overwrite=True)

    assert vk.build(**build_kwargs)[0] == vcrs
    assert vk.build(min_unique_triplets=5, **build_kwargs)[0] == vcrs  # 14 distinct triplets globally -> kept
    assert vk.build(min_unique_triplets_local=4, local_length=3, **build_kwargs)[0] == vcrs  # 4 locally -> kept
    assert vk.build(min_unique_triplets_local=5, local_length=3, **build_kwargs) is None  # 4 locally -> dropped
    assert vk.build(min_unique_triplets_local=5, local_length=10, **build_kwargs)[0] == vcrs  # window covers the whole VCRS -> kept


def test_min_unique_triplets_local_requires_local_length(out_dir):
    with pytest.raises(Exception, match="local_length"):
        vk.build(dont_create_index=True, sequences="GTCCAGTAAACAAAGTCAGCT", variants="c.11C>G", w=10, k=11, out=out_dir, overwrite=True, min_unique_triplets_local=5)
