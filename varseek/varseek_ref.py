import os
import subprocess
import inspect
from typing import Union, List, Optional
import varseek as vk
from varseek.utils import set_up_logger, save_params_to_config_file, return_kb_arguments
from .constants import allowable_kwargs

logger = set_up_logger()

varseek_ref_unallowable_arguments = {
    "varseek_build": {"return_mutation_output"},
    "varseek_info": set(),
    "varseek_filter": set(),
    "kb_ref": set(),
}

varseek_ref_only_allowable_kb_ref_arguments = {
    "zero_arguments": {"--keep-tmp", "--verbose", "--aa", "--overwrite"},
    "one_argument": {"--tmp", "--kallisto", "--bustools"},
    "multiple_arguments": set()
}  # don't include d-list, t, i, k, workflow here because I do it myself later

def ref(
    sequences: Union[str, List[str]],
    mutations: Union[str, List[str]],
    mut_column: str = "mutation",
    seq_id_column: str = "seq_ID",
    mut_id_column: Optional[str] = None,
    gtf: Optional[str] = None,
    gtf_transcript_id_column: Optional[str] = None,
    w: int = 30,
    k: Optional[int] = None,
    insertion_size_limit: Optional[int] = None,
    min_seq_len: Optional[int] = None,
    optimize_flanking_regions: bool = False,
    remove_seqs_with_wt_kmers: bool = False,
    max_ambiguous: Optional[int] = None,
    required_insertion_overlap_length: Union[int, str, None] = None,
    merge_identical: bool = False,
    strandedness: bool = False,
    keep_original_headers: bool = False,
    save_wt_mcrs_fasta_and_t2g: bool = False,
    save_mutations_updated_csv: bool = False,
    store_full_sequences: bool = False,
    translate: bool = False,
    translate_start: Union[int, str, None] = None,
    translate_end: Union[int, str, None] = None,
    out: str = ".",
    reference_out_dir: Optional[str] = None,
    mcrs_fasta_out: Optional[str] = None,
    mutations_updated_csv_out: Optional[str] = None,
    id_to_header_csv_out: Optional[str] = None,
    mcrs_t2g_out: Optional[str] = None,
    wt_mcrs_fasta_out: Optional[str] = None,
    wt_mcrs_t2g_out: Optional[str] = None,
    # return_mutation_output: bool = False,
    verbose: bool = True,
    mcrs_index_out: Optional[str] = None,
    dlist=False,  # path to dlist fasta file or "None" (including the quotes)
    **kwargs
):

    # check if passed arguments are valid
    function_name_and_key_list_of_tuples = [(vk.varseek_build.build, "varseek_build"), (vk.varseek_info.info, "varseek_info"), (vk.varseek_filter.filter, "varseek_filter")]

    all_allowable_parameters = set()
    function_name_to_dict_of_all_args = {}
    function_name_to_dict_of_kwargs = {}

    # add to allowable_kwargs_for_function. allowable_kwargs_for_function['varseek_build'] will have all kwargs for varseek_build, and analogously for info, filter, kb ref
    for function_name, function_key in function_name_and_key_list_of_tuples:
        function_parameters = set(inspect.signature(function_name).parameters.keys())
        allowable_kwargs_for_function = allowable_kwargs[function_key]
        kwargs_for_function = {}
        all_args_for_function = {}
        for key, value in kwargs.items():
            if key in allowable_kwargs_for_function:
                kwargs_for_function[key] = value
            if (key in function_parameters or key in allowable_kwargs_for_function) and key not in varseek_ref_unallowable_arguments[key]:
                all_args_for_function[key] = value

        function_name_to_dict_of_kwargs[function_key] = kwargs_for_function
        function_name_to_dict_of_all_args[function_key] = all_args_for_function

        # add function_parameters to all_allowable_parameters
        all_allowable_parameters = all_allowable_parameters | function_parameters

    # handles kb ref
    for argument_type_key in varseek_ref_only_allowable_kb_ref_arguments:
        arguments_dashes_removed = {argument.lstrip('-').replace('-', '_') for argument in varseek_ref_only_allowable_kb_ref_arguments[argument_type_key]}
        all_allowable_parameters = all_allowable_parameters | arguments_dashes_removed

    # check if passed arguments are valid (which will automatically get passed to kwargs)
    for key in kwargs:
        if key not in all_allowable_parameters:
            raise ValueError(f"Invalid parameter: {key}.")
    
    # Make assertions and exceptions
    assert k >= w + 1, "k must be greater than or equal to w + 1"
    if not os.path.exists(sequences):
        raise FileNotFoundError(f"The file or path '{sequences}' does not exist")

    # Save parameters to config file
    config_file = os.path.join(out, "config", "vk_ref_config.json")
    save_params_to_config_file(config_file)
    
    # Make directories
    if not reference_out_dir:
        reference_out_dir = os.path.join(out, "reference")

    os.makedirs(out, exist_ok=True)
    os.makedirs(reference_out_dir, exist_ok=True)

    # get COSMIC info
    cosmic_email = kwargs.get("cosmic_email", None)
    if not cosmic_email:
        cosmic_email = os.getenv("COSMIC_EMAIL")
    if cosmic_email:
        logger.info(f"Using COSMIC email from COSMIC_EMAIL environment variable: {cosmic_email}")
        kwargs["cosmic_email"] = cosmic_email

    cosmic_password = kwargs.get("cosmic_password", None)
    if not cosmic_password:
        cosmic_password = os.getenv("COSMIC_PASSWORD")
    if cosmic_password:
        logger.info("Using COSMIC password from COSMIC_PASSWORD environment variable")
        kwargs["cosmic_password"] = cosmic_password

    # define some more file paths
    if not mcrs_index_out:
        mcrs_index_out = os.path.join(out, "mcrs_index.idx")
    os.makedirs(os.path.dirname(mcrs_index_out), exist_ok=True)

    if not mcrs_fasta_out:
        mcrs_fasta_out = os.path.join(out, "mcrs.fa")
    if not mcrs_filtered_fasta_out:
        mcrs_filtered_fasta_out = os.path.join(out, "mcrs_filtered.fa")
    if not mcrs_t2g_out:
        mcrs_t2g_out = os.path.join(out, "mcrs_t2g.txt")
    if not mcrs_t2g_filtered_out:
        mcrs_t2g_filtered_out = os.path.join(out, "mcrs_t2g_filtered.txt")

    if not filters:
        mcrs_fasta_for_index = mcrs_fasta_out
        mcrs_t2g_for_alignment = mcrs_t2g_out
    else:
        mcrs_fasta_for_index = mcrs_filtered_fasta_out
        mcrs_t2g_for_alignment = mcrs_t2g_filtered_out

    
    # set d-list argument
    if not dlist_genome_fasta_out:
        dlist_genome_fasta_out = os.path.join(out, "dlist_genome.fa")
    if not dlist_cdna_fasta_out:
        dlist_cdna_fasta_out = os.path.join(out, "dlist_cdna.fa")
    if not dlist_combined_fasta_out:
        dlist_combined_fasta_out = os.path.join(out, "dlist.fa")

    if dlist == "genome":
        dlist_kb_argument = dlist_genome_fasta_out
    elif dlist == "transcriptome":
        dlist_kb_argument = dlist_cdna_fasta_out
    elif dlist == "genome_and_transcriptome":
        dlist_kb_argument = dlist_combined_fasta_out
    elif dlist == "None" or dlist is None:
        dlist_kb_argument = "None"
    else:
        raise ValueError("dlist must be 'genome', 'transcriptome', 'genome_and_transcriptome', or 'None'")

    # vk build - if automating later, simply use vk.build(**function_name_to_dict_of_all_args['varseek_build'])
    vk.build(
        sequences=sequences,
        mutations=mutations,
        mut_column=mut_column,
        seq_id_column=seq_id_column,
        mut_id_column=mut_id_column,
        gtf=gtf,
        gtf_transcript_id_column=gtf_transcript_id_column,
        w=w,
        k=k,
        insertion_size_limit=insertion_size_limit,
        min_seq_len=min_seq_len,
        optimize_flanking_regions=optimize_flanking_regions,
        remove_seqs_with_wt_kmers=remove_seqs_with_wt_kmers,
        max_ambiguous=max_ambiguous,
        required_insertion_overlap_length=required_insertion_overlap_length,
        merge_identical=merge_identical,
        strandedness=strandedness,
        keep_original_headers=keep_original_headers,
        save_wt_mcrs_fasta_and_t2g=save_wt_mcrs_fasta_and_t2g,
        save_mutations_updated_csv=save_mutations_updated_csv,
        store_full_sequences=store_full_sequences,
        translate=translate,
        translate_start=translate_start,
        translate_end=translate_end,
        out=out,
        reference_out_dir=reference_out_dir,
        mcrs_fasta_out=mcrs_fasta_out,
        mutations_updated_csv_out=mutations_updated_csv_out,
        id_to_header_csv_out=id_to_header_csv_out,
        mcrs_t2g_out=mcrs_t2g_out,
        wt_mcrs_fasta_out=wt_mcrs_fasta_out,
        wt_mcrs_t2g_out=wt_mcrs_t2g_out,
        verbose=verbose,
        **function_name_to_dict_of_kwargs["varseek_build"],  # I use this rather than all kwargs to ensure that I only pass in the kwargs I expect
    )

    # vk info
    vk.info(
        mutations=vk_build_mcrs_fa_path,
        updated_df=update_df_out,
        id_to_header_csv=id_to_header_csv,  # if none then assume no swapping occurred
        columns_to_include=columns_to_include,
        mcrs_id_column="mcrs_id",
        mcrs_sequence_column="mutant_sequence",
        mcrs_source_column="mcrs_source",  # if input df has concatenated cdna and header MCRS's, then I want to know whether it came from cdna or genome
        seqid_cdna_column="seq_ID",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome  # TODO: implement these 4 column name arguments
        seqid_genome_column="chromosome",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome
        mutation_cdna_column="mutation",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome
        mutation_genome_column="mutation_genome",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome
        gtf=gtf_path,  # for distance to nearest splice junction
        mutation_metadata_df_out_path=mutation_metadata_df_out_path_vk_info,
        out_dir_notebook=out_dir_notebook,
        reference_out_dir=reference_out_dir,
        dlist_reference_source=dlist_reference_source,
        ref_prefix="index",
        w=w,
        remove_Ns=remove_Ns,
        strandedness=strandedness,
        bowtie_path=bowtie_path,
        near_splice_junction_threshold=near_splice_junction_threshold,
        threads=threads,
        reference_cdna_fasta=reference_cdna_fasta,
        reference_genome_fasta=reference_genome_fasta,
        mutations_csv=mutations_csv,
        save_exploded_df=save_exploded_df,
        verbose=verbose,
    )

    # vk filter
    if filters:
        vk.filter(
            mutation_metadata_df_path=mutation_metadata_df_out_path_vk_info,
            output_mcrs_fasta=mcrs_fasta_vk_filter,
            output_metadata_df=output_metadata_df_vk_filter,
            dlist_fasta=dlist_fasta,
            output_dlist_fasta=dlist_fasta_vk_filter,
            output_t2g=t2g_vk_filter,
            id_to_header_csv=id_to_header_csv,
            output_id_to_header_csv=id_to_header_csv_vk_filter,
            verbose=True,
            return_df=False,
            filters=fasta_filters,
        )

    # kb ref
    kb_ref_command = [
        "kb",
        "ref",
        "--workflow",
        "custom",
        "-t",
        str(threads),
        "-i",
        mcrs_index_out,
        "--d-list",
        dlist_kb_argument,
        "-k",
        str(k)
    ]

    # assumes any argument in varseek ref matches kb ref identically, except dashes replaced with underscores
    for dict_key in varseek_ref_only_allowable_kb_ref_arguments:
        for argument in list(varseek_ref_only_allowable_kb_ref_arguments[dict_key]):
            dash_count = len(argument) - len(argument.lstrip('-'))
            leading_dashes = "-"*dash_count
            argument = argument.lstrip('-').replace('-', '_')
            if argument in kwargs:
                value = kwargs[argument]
                if dict_key == 'zero_arguments':
                    if value:  # only add if value is True
                        kb_ref_command.append(f"{leading_dashes}{argument}")
                elif dict_key == 'one_argument':
                    kb_ref_command.extend([f"{leading_dashes}{argument}", value])
                else:  # multiple_arguments or something else
                    pass

    kb_ref_command.append(mcrs_fasta_for_index)
    
    subprocess.run(kb_ref_command, check=True)

    vk_ref_output_dict = {}
    vk_ref_output_dict["index"] = mcrs_index_out
    vk_ref_output_dict["t2g"] = mcrs_t2g_for_alignment
    # vk_ref_output_dict["dlist"] = dlist_fasta_vk_filter
    # vk_ref_output_dict["id_to_header"] = id_to_header_csv_vk_filter
    # vk_ref_output_dict["mcrs_fasta"] = mcrs_fasta_vk_filter
    # vk_ref_output_dict["metadata_df"] = output_metadata_df_vk_filter

    return vk_ref_output_dict

# ref.__doc__ = ref.__doc__ + vk.varseek_build.build.__doc__ + vk.varseek_info.info.__doc__ + vk.varseek_filter.filter.__doc__
