import pytest
import tempfile
import varseek as vk
import pandas as pd
import numpy as np
import os
from pdb import set_trace as st
from varseek.utils import create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting, make_mapping_dict, load_t2g_as_dict, filter_fasta, compare_dicts

def add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path, column_name="numeric_value"):
    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    df[column_name] = np.arange(len(df))

    df.to_csv(toy_mutation_metadata_df_path, index=False)


@pytest.fixture
def temporary_output_files():
    with tempfile.NamedTemporaryFile(suffix=".csv") as output_metadata_df, \
         tempfile.NamedTemporaryFile(suffix=".fasta") as output_mcrs_fasta, \
         tempfile.NamedTemporaryFile(suffix=".fasta") as output_dlist_fasta, \
         tempfile.NamedTemporaryFile(suffix=".csv") as output_id_to_header_csv, \
         tempfile.NamedTemporaryFile(suffix=".txt") as output_t2g:
        
        # Dictionary of temporary paths
        temp_files = {
            "output_metadata_df": output_metadata_df.name,
            "output_mcrs_fasta": output_mcrs_fasta.name,
            "output_dlist_fasta": output_dlist_fasta.name,
            "output_id_to_header_csv": output_id_to_header_csv.name,
            "output_t2g": output_t2g.name
        }
        
        yield temp_files  # Provide paths to test

        # Temporary files are automatically cleaned up after `yield`

def test_no_filters_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = []

    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)


def test_no_filters_mcrs_fa(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = []
    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )

    mcrs_fasta_header_to_sequence_dict_from_test = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(output_mcrs_fasta)

    # mcrs_fasta_header_to_sequence_dict_expected = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(toy_mcrs_fa_path)

    mutation_metadata_df_prefiltering = pd.read_csv(toy_mutation_metadata_df_path)

    # mutation_metadata_df_prefiltering['mcrs_id'] = mutation_metadata_df_prefiltering['mcrs_id'].astype(str)

    mcrs_fasta_header_to_sequence_dict_expected = mutation_metadata_df_prefiltering.set_index('mcrs_id')['mcrs_sequence'].to_dict()

    assert dict(mcrs_fasta_header_to_sequence_dict_from_test) == dict(mcrs_fasta_header_to_sequence_dict_expected)

def test_no_filters_dlist_fa(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = []
    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )

    dlist_fasta_header_to_sequence_dict_from_test = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(output_dlist_fasta)

    dlist_fasta_header_to_sequence_dict_expected = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(dlist_file_small_path)

    assert dict(dlist_fasta_header_to_sequence_dict_from_test) == dict(dlist_fasta_header_to_sequence_dict_expected)


def test_no_filters_id_to_header_mapping_csv(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = []
    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )

    id_to_header_dict_from_test = make_mapping_dict(output_id_to_header_csv, dict_key="id")

    id_to_header_dict_from_expected = make_mapping_dict(toy_id_to_header_mapping_csv_path, dict_key="id")

    dict(id_to_header_dict_from_test) == dict(id_to_header_dict_from_expected)


def test_no_filters_id_to_header_mapping_t2g(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = []
    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )

    t2g_from_test = load_t2g_as_dict(output_t2g)

    t2g_from_expected = load_t2g_as_dict(toy_t2g_path)

    dict(t2g_from_test) == dict(t2g_from_expected)

def test_single_filter_min_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = ['numeric_value-min=3']

    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )

    # output_metadata_df_from_test = pd.read_csv(output_metadata_df)
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    output_metadata_df_expected = output_metadata_df_expected.iloc[3:].reset_index(drop=True)

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)


def test_single_filter_min_mcrs_fa(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = ['numeric_value-min=3']
    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )

    mcrs_fasta_header_to_sequence_dict_from_test = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(output_mcrs_fasta)

    mutation_metadata_df_prefiltering = pd.read_csv(toy_mutation_metadata_df_path)
    mutation_metadata_df_filtered = mutation_metadata_df_prefiltering.iloc[3:].reset_index(drop=True)

    mcrs_fasta_header_to_sequence_dict_expected = mutation_metadata_df_filtered.set_index('mcrs_id')['mcrs_sequence'].to_dict()

    assert dict(mcrs_fasta_header_to_sequence_dict_from_test) == dict(mcrs_fasta_header_to_sequence_dict_expected)

def test_single_filter_min_dlist_fa(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = ['numeric_value-min=3']
    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )

    dlist_fasta_header_to_sequence_dict_from_test = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(output_dlist_fasta)

    mutation_metadata_df_prefiltering = pd.read_csv(toy_mutation_metadata_df_path)
    mutation_metadata_df_filtered = mutation_metadata_df_prefiltering.iloc[3:].reset_index(drop=True)

    filtered_df_mcrs_ids = set(mutation_metadata_df_filtered["mcrs_id"])

    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=True) as temp_output_file:
        output_dlist_fasta_real = temp_output_file.name
        filter_fasta(dlist_file_small_path, output_dlist_fasta_real, filtered_df_mcrs_ids)
        dlist_fasta_header_to_sequence_dict_expected = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(output_dlist_fasta_real)

    assert dict(dlist_fasta_header_to_sequence_dict_from_test) == dict(dlist_fasta_header_to_sequence_dict_expected)


def test_single_filter_min_id_to_header_mapping_csv(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = ['numeric_value-min=3']
    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )

    id_to_header_dict_from_test = make_mapping_dict(output_id_to_header_csv, dict_key="id")

    mutation_metadata_df_prefiltering = pd.read_csv(toy_mutation_metadata_df_path)
    mutation_metadata_df_filtered = mutation_metadata_df_prefiltering.iloc[3:].reset_index(drop=True)

    filtered_df_mcrs_ids = set(mutation_metadata_df_filtered["mcrs_id"])

    id_to_header_dict_from_expected = make_mapping_dict(toy_id_to_header_mapping_csv_path, dict_key="id")
    id_to_header_dict_from_expected = {k: v for k, v in id_to_header_dict_from_expected.items() if k in filtered_df_mcrs_ids}

    dict(id_to_header_dict_from_test) == dict(id_to_header_dict_from_expected)


def test_single_filter_min_id_to_header_mapping_t2g(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = ['numeric_value-min=3']
    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )

    t2g_from_test = load_t2g_as_dict(output_t2g)

    t2g_from_expected = load_t2g_as_dict(toy_t2g_path)

    mutation_metadata_df_prefiltering = pd.read_csv(toy_mutation_metadata_df_path)
    mutation_metadata_df_filtered = mutation_metadata_df_prefiltering.iloc[3:].reset_index(drop=True)

    filtered_df_mcrs_ids = set(mutation_metadata_df_filtered["mcrs_id"])

    t2g_from_expected = {k: v for k, v in t2g_from_expected.items() if k in filtered_df_mcrs_ids}

    dict(t2g_from_test) == dict(t2g_from_expected)


def test_multi_filter_min_max_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = ['numeric_value-min=3', 'numeric_value2-max=7']

    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path, column_name="numeric_value")
    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path, column_name="numeric_value2")

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    # st()

    output_metadata_df_expected = output_metadata_df_expected[(output_metadata_df_expected['numeric_value'] >= 3) & (output_metadata_df_expected['numeric_value2'] <= 7)].reset_index(drop=True)

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)


def test_multi_filter_between_equal_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    filters = ['numeric_value-between=2,7', 'chromosome_single-equal=1']

    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path, column_name="numeric_value")

    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    df['chromosome_single'] = 'X'
    df.loc[:4, 'chromosome_single'] = '1'

    df.to_csv(toy_mutation_metadata_df_path, index=False)

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    # ensure string
    output_metadata_df_from_test['chromosome_single'] = output_metadata_df_from_test['chromosome_single'].astype(str)
    output_metadata_df_expected['chromosome_single'] = output_metadata_df_expected['chromosome_single'].astype(str)

    output_metadata_df_expected = output_metadata_df_expected[(output_metadata_df_expected['numeric_value'] >= 2) & (output_metadata_df_expected['numeric_value'] <= 4)].reset_index(drop=True)
    output_metadata_df_expected = output_metadata_df_expected[output_metadata_df_expected['chromosome_single'] == '1'].reset_index(drop=True)

    # st()

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)


def test_multi_filter_contains_istrue_isfalse_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):
    mcrs_id_set = {"seq1204954474446204", "seq1693806423259989", "seq1784404960707341", "seq2241452516841814", "seq9627237534759445"} # elements 0, 2, 4, 6, 8
    filters = [f'mcrs_id-isin={mcrs_id_set}', 'bool_col1-istrue', 'bool_col2-isfalse']
    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    df['bool_col1'] = False
    df['bool_col2'] = False
    df.loc[:7, 'bool_col1'] = True
    df.loc[:3, 'bool_col2'] = True

    # should only have 4, 6 match all conditions

    df.to_csv(toy_mutation_metadata_df_path, index=False)

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    output_metadata_df_expected = output_metadata_df_expected[
        (output_metadata_df_expected["bool_col1"] == True) &          # bool_col1 is True
        (output_metadata_df_expected["bool_col2"] == False) &         # bool_col2 is False
        (output_metadata_df_expected["mcrs_id"].isin(mcrs_id_set))    # mcrs_id in specified set
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)


def test_multi_filter_contains_isnull_isnotnull_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_mcrs_fa_path, temporary_output_files):    
    output_metadata_df, output_mcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_mcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    filters = [f'int_col-isnull', 'str_col-isnotnull']

    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    df['int_col'] = np.where(df.index < 3, 1, np.nan)
    df['str_col'] = np.where(df.index >=7, np.nan, "yes")

    # should only have values 3, 4, 5, 6

    df.to_csv(toy_mutation_metadata_df_path, index=False)

    output_metadata_df_from_test = vk.filter(
        mutation_metadata_df_path=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        create_t2g=True,
        output_metadata_df=output_metadata_df,
        output_mcrs_fasta=output_mcrs_fasta,
        output_dlist_fasta=output_dlist_fasta,
        output_id_to_header_csv=output_id_to_header_csv,
        output_t2g=output_t2g,
        filters=filters
    )
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    output_metadata_df_expected = output_metadata_df_expected[
        (output_metadata_df_expected["int_col"].isnull()) &          # bool_col1 is True
        (output_metadata_df_expected["str_col"].notnull())         # bool_col2 is False
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)