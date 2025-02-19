import os
import tempfile
from pdb import set_trace as st

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

import varseek as vk
from varseek.utils import (
    compare_dicts,
    create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting,
    filter_fasta,
    load_t2g_as_dict,
    make_mapping_dict,
)

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


def add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path, column_name="numeric_value"):
    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    df[column_name] = np.arange(len(df))

    df.to_csv(toy_mutation_metadata_df_path, index=False)


@pytest.fixture
def temporary_output_files():
    with tempfile.NamedTemporaryFile(suffix=".csv") as output_metadata_df, \
         tempfile.NamedTemporaryFile(suffix=".fasta") as output_vcrs_fasta, \
         tempfile.NamedTemporaryFile(suffix=".fasta") as output_dlist_fasta, \
         tempfile.NamedTemporaryFile(suffix=".csv") as output_id_to_header_csv, \
         tempfile.NamedTemporaryFile(suffix=".txt") as output_t2g:
        
        # Dictionary of temporary paths
        temp_files = {
            "output_metadata_df": output_metadata_df.name,
            "output_vcrs_fasta": output_vcrs_fasta.name,
            "output_dlist_fasta": output_dlist_fasta.name,
            "output_id_to_header_csv": output_id_to_header_csv.name,
            "output_t2g": output_t2g.name
        }
        
        yield temp_files  # Provide paths to test

        # Temporary files are automatically cleaned up after `yield`

def test_no_filters(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):
    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    with pytest.raises(ValueError, match="No filters provided"):
        output_metadata_df_from_test = vk.filter(
            input_dir = ".",
            filters=[],
            variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
            dlist_fasta=dlist_file_small_path,
            id_to_header_csv=toy_id_to_header_mapping_csv_path,
            variants_updated_filtered_csv_out=output_metadata_df,
            vcrs_filtered_fasta_out=output_vcrs_fasta,
            dlist_filtered_fasta_out=output_dlist_fasta,
            id_to_header_filtered_csv_out=output_id_to_header_csv,
            vcrs_t2g_filtered_out=output_t2g,
            return_variants_updated_filtered_csv_df=True,
            overwrite=True
        )

    with pytest.raises(ValueError, match="No filters provided"):
        output_metadata_df_from_test = vk.filter(
            input_dir = ".",
            filters="",
            variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
            dlist_fasta=dlist_file_small_path,
            id_to_header_csv=toy_id_to_header_mapping_csv_path,
            variants_updated_filtered_csv_out=output_metadata_df,
            vcrs_filtered_fasta_out=output_vcrs_fasta,
            dlist_filtered_fasta_out=output_dlist_fasta,
            id_to_header_filtered_csv_out=output_id_to_header_csv,
            vcrs_t2g_filtered_out=output_t2g,
            return_variants_updated_filtered_csv_df=True,
            overwrite=True
        )

    with pytest.raises(ValueError, match="No filters provided"):
        output_metadata_df_from_test = vk.filter(
            input_dir = ".",
            filters=None,
            variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
            dlist_fasta=dlist_file_small_path,
            id_to_header_csv=toy_id_to_header_mapping_csv_path,
            variants_updated_filtered_csv_out=output_metadata_df,
            vcrs_filtered_fasta_out=output_vcrs_fasta,
            dlist_filtered_fasta_out=output_dlist_fasta,
            id_to_header_filtered_csv_out=output_id_to_header_csv,
            vcrs_t2g_filtered_out=output_t2g,
            return_variants_updated_filtered_csv_df=True,
            overwrite=True
        )

    with pytest.raises(ValueError, match="No filters provided"):
        output_metadata_df_from_test = vk.filter(
            input_dir = ".",
            filters="None",
            variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
            dlist_fasta=dlist_file_small_path,
            id_to_header_csv=toy_id_to_header_mapping_csv_path,
            variants_updated_filtered_csv_out=output_metadata_df,
            vcrs_filtered_fasta_out=output_vcrs_fasta,
            dlist_filtered_fasta_out=output_dlist_fasta,
            id_to_header_filtered_csv_out=output_id_to_header_csv,
            vcrs_t2g_filtered_out=output_t2g,
            return_variants_updated_filtered_csv_df=True,
            overwrite=True
        )

def test_single_filter_min_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):
    filters = ['numeric_value:greater_or_equal=3']

    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        input_dir = ".",
        filters=filters,
        variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        variants_updated_filtered_csv_out=output_metadata_df,
        vcrs_filtered_fasta_out=output_vcrs_fasta,
        dlist_filtered_fasta_out=output_dlist_fasta,
        id_to_header_filtered_csv_out=output_id_to_header_csv,
        vcrs_t2g_filtered_out=output_t2g,
        return_variants_updated_filtered_csv_df=True,
        overwrite=True
    )

    # output_metadata_df_from_test = pd.read_csv(output_metadata_df)
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    output_metadata_df_expected = output_metadata_df_expected.iloc[3:].reset_index(drop=True)

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)


def test_single_filter_min_vcrs_fa(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):
    filters = ['numeric_value:greater_or_equal=3']
    
    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        input_dir = ".",
        filters=filters,
        variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        variants_updated_filtered_csv_out=output_metadata_df,
        vcrs_filtered_fasta_out=output_vcrs_fasta,
        dlist_filtered_fasta_out=output_dlist_fasta,
        id_to_header_filtered_csv_out=output_id_to_header_csv,
        vcrs_t2g_filtered_out=output_t2g,
        return_variants_updated_filtered_csv_df=True,
        overwrite=True
    )

    vcrs_fasta_header_to_sequence_dict_from_test = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(output_vcrs_fasta)

    mutation_metadata_df_prefiltering = pd.read_csv(toy_mutation_metadata_df_path)
    mutation_metadata_df_filtered = mutation_metadata_df_prefiltering.iloc[3:].reset_index(drop=True)

    vcrs_fasta_header_to_sequence_dict_expected = mutation_metadata_df_filtered.set_index('vcrs_header')['vcrs_sequence'].to_dict()

    assert dict(vcrs_fasta_header_to_sequence_dict_from_test) == dict(vcrs_fasta_header_to_sequence_dict_expected)

def test_single_filter_min_dlist_fa(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):
    filters = ['numeric_value:greater_or_equal=3']
    
    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        input_dir = ".",
        filters=filters,
        variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        variants_updated_filtered_csv_out=output_metadata_df,
        vcrs_filtered_fasta_out=output_vcrs_fasta,
        dlist_filtered_fasta_out=output_dlist_fasta,
        id_to_header_filtered_csv_out=output_id_to_header_csv,
        vcrs_t2g_filtered_out=output_t2g,
        return_variants_updated_filtered_csv_df=True,
        overwrite=True
    )

    dlist_fasta_header_to_sequence_dict_from_test = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(output_dlist_fasta)

    mutation_metadata_df_prefiltering = pd.read_csv(toy_mutation_metadata_df_path)
    mutation_metadata_df_filtered = mutation_metadata_df_prefiltering.iloc[3:].reset_index(drop=True)

    filtered_df_vcrs_ids = set(mutation_metadata_df_filtered["vcrs_id"])

    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=True) as temp_output_file:
        output_dlist_fasta_real = temp_output_file.name
        filter_fasta(dlist_file_small_path, output_dlist_fasta_real, filtered_df_vcrs_ids)
        dlist_fasta_header_to_sequence_dict_expected = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(output_dlist_fasta_real)

    assert dict(dlist_fasta_header_to_sequence_dict_from_test) == dict(dlist_fasta_header_to_sequence_dict_expected)


def test_single_filter_min_id_to_header_mapping_csv(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):
    filters = ['numeric_value:greater_or_equal=3']
    
    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        input_dir = ".",
        filters=filters,
        variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        variants_updated_filtered_csv_out=output_metadata_df,
        vcrs_filtered_fasta_out=output_vcrs_fasta,
        dlist_filtered_fasta_out=output_dlist_fasta,
        id_to_header_filtered_csv_out=output_id_to_header_csv,
        vcrs_t2g_filtered_out=output_t2g,
        return_variants_updated_filtered_csv_df=True,
        overwrite=True
    )

    id_to_header_dict_from_test = make_mapping_dict(output_id_to_header_csv, dict_key="id")

    mutation_metadata_df_prefiltering = pd.read_csv(toy_mutation_metadata_df_path)
    mutation_metadata_df_filtered = mutation_metadata_df_prefiltering.iloc[3:].reset_index(drop=True)

    filtered_df_vcrs_ids = set(mutation_metadata_df_filtered["vcrs_id"])

    id_to_header_dict_from_expected = make_mapping_dict(toy_id_to_header_mapping_csv_path, dict_key="id")
    id_to_header_dict_from_expected = {k: v for k, v in id_to_header_dict_from_expected.items() if k in filtered_df_vcrs_ids}

    dict(id_to_header_dict_from_test) == dict(id_to_header_dict_from_expected)


def test_single_filter_min_id_to_header_mapping_t2g(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):
    filters = ['numeric_value:greater_or_equal=3']
    
    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path)

    output_metadata_df_from_test = vk.filter(
        input_dir = ".",
        filters=filters,
        variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        variants_updated_filtered_csv_out=output_metadata_df,
        vcrs_filtered_fasta_out=output_vcrs_fasta,
        dlist_filtered_fasta_out=output_dlist_fasta,
        id_to_header_filtered_csv_out=output_id_to_header_csv,
        vcrs_t2g_filtered_out=output_t2g,
        return_variants_updated_filtered_csv_df=True,
        overwrite=True
    )

    t2g_from_test = load_t2g_as_dict(output_t2g)

    t2g_from_expected = load_t2g_as_dict(toy_t2g_path)

    mutation_metadata_df_prefiltering = pd.read_csv(toy_mutation_metadata_df_path)
    mutation_metadata_df_filtered = mutation_metadata_df_prefiltering.iloc[3:].reset_index(drop=True)

    filtered_df_vcrs_ids = set(mutation_metadata_df_filtered["vcrs_id"])

    t2g_from_expected = {k: v for k, v in t2g_from_expected.items() if k in filtered_df_vcrs_ids}

    dict(t2g_from_test) == dict(t2g_from_expected)


def test_multi_filter_min_max_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):
    filters = ['numeric_value:greater_or_equal=3', 'numeric_value2:less_or_equal=7']

    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path, column_name="numeric_value")
    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path, column_name="numeric_value2")

    output_metadata_df_from_test = vk.filter(
        input_dir = ".",
        filters=filters,
        variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        variants_updated_filtered_csv_out=output_metadata_df,
        vcrs_filtered_fasta_out=output_vcrs_fasta,
        dlist_filtered_fasta_out=output_dlist_fasta,
        id_to_header_filtered_csv_out=output_id_to_header_csv,
        vcrs_t2g_filtered_out=output_t2g,
        return_variants_updated_filtered_csv_df=True,
        overwrite=True
    )
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    # st()

    output_metadata_df_expected = output_metadata_df_expected[(output_metadata_df_expected['numeric_value'] >= 3) & (output_metadata_df_expected['numeric_value2'] <= 7)].reset_index(drop=True)

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)


def test_multi_filter_between_equal_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):
    filters = ['numeric_value:between_inclusive=2,7', 'chromosome_single:equal=1']

    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    add_numeric_value_column_to_df_that_applies_range_of_len_df(toy_mutation_metadata_df_path, column_name="numeric_value")

    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    df['chromosome_single'] = 'X'
    df.loc[:4, 'chromosome_single'] = '1'

    df.to_csv(toy_mutation_metadata_df_path, index=False)

    output_metadata_df_from_test = vk.filter(
        input_dir = ".",
        filters=filters,
        variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        variants_updated_filtered_csv_out=output_metadata_df,
        vcrs_filtered_fasta_out=output_vcrs_fasta,
        dlist_filtered_fasta_out=output_dlist_fasta,
        id_to_header_filtered_csv_out=output_id_to_header_csv,
        vcrs_t2g_filtered_out=output_t2g,
        return_variants_updated_filtered_csv_df=True,
        overwrite=True
    )
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    # ensure string
    output_metadata_df_from_test['chromosome_single'] = output_metadata_df_from_test['chromosome_single'].astype(str)
    output_metadata_df_expected['chromosome_single'] = output_metadata_df_expected['chromosome_single'].astype(str)

    output_metadata_df_expected = output_metadata_df_expected[(output_metadata_df_expected['numeric_value'] >= 2) & (output_metadata_df_expected['numeric_value'] <= 4)].reset_index(drop=True)
    output_metadata_df_expected = output_metadata_df_expected[output_metadata_df_expected['chromosome_single'] == '1'].reset_index(drop=True)

    # st()

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)


def test_multi_filter_contains_istrue_isfalse_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):
    vcrs_id_set = {"seq1204954474446204", "seq1693806423259989", "seq1784404960707341", "seq2241452516841814", "seq9627237534759445"} # elements 0, 2, 4, 6, 8
    filters = [f'vcrs_id:is_in={vcrs_id_set}', 'bool_col1:is_true', 'bool_col2:is_false']
    
    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    df['bool_col1'] = False
    df['bool_col2'] = False
    df.loc[:7, 'bool_col1'] = True
    df.loc[:3, 'bool_col2'] = True

    # should only have 4, 6 match all conditions

    df.to_csv(toy_mutation_metadata_df_path, index=False)

    output_metadata_df_from_test = vk.filter(
        input_dir = ".",
        filters=filters,
        variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        variants_updated_filtered_csv_out=output_metadata_df,
        vcrs_filtered_fasta_out=output_vcrs_fasta,
        dlist_filtered_fasta_out=output_dlist_fasta,
        id_to_header_filtered_csv_out=output_id_to_header_csv,
        vcrs_t2g_filtered_out=output_t2g,
        return_variants_updated_filtered_csv_df=True,
        overwrite=True
    )
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    output_metadata_df_expected = output_metadata_df_expected[
        (output_metadata_df_expected["bool_col1"] == True) &          # bool_col1 is True
        (output_metadata_df_expected["bool_col2"] == False) &         # bool_col2 is False
        (output_metadata_df_expected["vcrs_id"].isin(vcrs_id_set))    # vcrs_id in specified set
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)


def test_multi_filter_contains_isnull_isnotnull_metadata_df(toy_mutation_metadata_df_path, dlist_file_small_path, toy_id_to_header_mapping_csv_path, toy_t2g_path, toy_vcrs_fa_path, temporary_output_files):    
    output_metadata_df, output_vcrs_fasta, output_dlist_fasta, output_id_to_header_csv, output_t2g = temporary_output_files["output_metadata_df"], temporary_output_files["output_vcrs_fasta"], temporary_output_files["output_dlist_fasta"], temporary_output_files["output_id_to_header_csv"], temporary_output_files["output_t2g"]

    filters = [f'int_col:is_null', 'str_col:is_not_null']

    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    df['int_col'] = np.where(df.index < 3, 1, np.nan)
    df['str_col'] = np.where(df.index >=7, np.nan, "yes")

    # should only have values 3, 4, 5, 6

    df.to_csv(toy_mutation_metadata_df_path, index=False)

    output_metadata_df_from_test = vk.filter(
        input_dir = ".",
        filters=filters,
        variants_updated_vk_info_csv=toy_mutation_metadata_df_path,
        dlist_fasta=dlist_file_small_path,
        id_to_header_csv=toy_id_to_header_mapping_csv_path,
        variants_updated_filtered_csv_out=output_metadata_df,
        vcrs_filtered_fasta_out=output_vcrs_fasta,
        dlist_filtered_fasta_out=output_dlist_fasta,
        id_to_header_filtered_csv_out=output_id_to_header_csv,
        vcrs_t2g_filtered_out=output_t2g,
        return_variants_updated_filtered_csv_df=True,
        overwrite=True
    )
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_path)

    output_metadata_df_expected = output_metadata_df_expected[
        (output_metadata_df_expected["int_col"].isnull()) &          # bool_col1 is True
        (output_metadata_df_expected["str_col"].notnull())         # bool_col2 is False
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(output_metadata_df_from_test, output_metadata_df_expected)