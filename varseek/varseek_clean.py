import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata

import gget

import scipy as sp
import pysam

import time

from .constants import non_single_cell_technologies
from varseek.utils import (
    plot_knee_plot,
    increment_adata_based_on_dlist_fns,
    decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode,
    remove_adata_columns,
    adjust_mutation_adata_by_normal_gene_matrix,
    match_adata_orders,
    write_to_vcf,
    write_vcfs_for_rows,
    set_up_logger,
    save_params_to_config_file,
    make_function_parameter_to_value_dict,
    check_file_path_is_string_with_valid_extension,
    print_varseek_dry_run,
    report_time_elapsed,
    is_valid_int,
    save_run_info
)

logger = set_up_logger()

def make_vcf():
    pass


#* ORIGINAL
# def clean(
#     adata,
#     adata_out=None,
#     out=".",
#     id_to_header_csv=None,
#     mutation_metadata_df=None,
#     mutation_metadata_df_columns=None,
#     min_counts=0,
#     use_binary_matrix=False,
#     drop_zero_columns=False,
#     technology="bulk",
#     adjust_mutation_adata_by_normal_gene_matrix_information=False,  # * change to True
#     filter_cells_by_min_counts=None,
#     filter_cells_by_min_genes=None,
#     filter_genes_by_min_cells=None,
#     filter_cells_by_max_mt_content=None,
#     doublet_detection=None,
#     remove_doublets=None,
#     do_cpm_normalization=None,
#     ignore_barcodes=False,
#     kb_count_out_normal_genome=None,
#     adata_path_normal_genome=None,
#     split_reads_by_Ns=False,
#     dlist_file=None,
#     mcrs_id_column="mcrs_id",
#     data_fastq=None,  # list of fastqs
#     mcrs_fasta=None,
#     dlist_fasta=None,
#     kb_count_out_mutant=None,
#     mcrs_index=None,
#     mcrs_t2g=None,
#     k=None,
#     mm=None,
#     threads=None,
#     strand=None,
#     newer_kallisto=None,
#     bustools="/home/jrich/miniconda3/envs/cartf/lib/python3.10/site-packages/kb_python/bins/linux/bustools/bustools",
#     mcrs_id_set_to_exclusively_keep=None,
#     mcrs_id_set_to_exclude=None,
#     transcript_set_to_exclusively_keep=None,
#     transcript_set_to_exclude=None,
#     gene_set_to_exclusively_keep=None,
#     gene_set_to_exclude=None,
#     adata_normal_genome_output_path=None,
#     make_vcf = False,
#     verbose=False,
#     **kwargs,
# ):

def clean(
    adata,
    adata_out=None,
    out=".",
    id_to_header_csv=None,
    mutation_metadata_df=None,
    mutation_metadata_df_columns=None,
    min_counts=1,
    use_binary_matrix=False,
    drop_empty_columns=False,
    technology="bulk",
    qc_against_gene_matrix=False,  # * change to True
    filter_cells_by_min_counts=None,
    filter_cells_by_min_genes=None,
    filter_genes_by_min_cells=None,
    filter_cells_by_max_mt_content=None,
    doublet_detection=None,
    remove_doublets=None,
    cpm_normalization=None,
    ignore_barcodes=False,
    kb_count_out_normal_genome=None,
    adata_path_normal_genome=None,
    split_reads_by_Ns=False,
    dlist_file=None,
    mcrs_id_column="mcrs_id",
    data_fastq=None,  # list of fastqs
    mcrs_fasta=None,
    dlist_fasta=None,
    kb_count_out_mutant=None,
    mcrs_index=None,
    mcrs_t2g=None,
    k=None,
    mm=None,
    threads=None,
    strand=None,
    newer_kallisto=None,
    bustools="/home/jrich/miniconda3/envs/cartf/lib/python3.10/site-packages/kb_python/bins/linux/bustools/bustools",
    mcrs_id_set_to_exclusively_keep=None,
    mcrs_id_set_to_exclude=None,
    transcript_set_to_exclusively_keep=None,
    transcript_set_to_exclude=None,
    gene_set_to_exclusively_keep=None,
    gene_set_to_exclude=None,
    adata_normal_genome_output_path=None,
    save_vcf = False,
    verbose=False,
    **kwargs,
):
    #* 1. Start timer
    start_time = time.perf_counter()
    
    #* 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_info(params_dict)

    #* 3. Dry-run and set out folder (must to it up here or else config will save in the wrong place)
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="info")
        return None
    if out is None:
        out = input_dir if input_dir else "."
    
    #* 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_info_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_info_run_info.txt")
    save_run_info(run_info_file)

    #* 5. Set up default folder/file input paths, and make sure the necessary ones exist
    
    #* 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    
    #* 7. Define kwargs defaults
    
    #* 8. Start the actual function





    if isinstance(adata, anndata.AnnData):
        adata_dir = "."
        output_type = "AnnData"
    elif isinstance(adata, str):
        adata = sc.read_h5ad(adata)
        adata_dir = os.path.dirname(adata)
        output_type = "path"
    else:
        raise ValueError("adata must be a string (file path) or an AnnData object.")

    # if kb_count_out_wt_mcrs_counterpart:
    #     adata_wt_mcrs_path = f"{kb_count_out_wt_mcrs_counterpart}/counts_unfiltered/adata.h5ad"
    #     adata_wt_mcrs = sc.read_h5ad(adata_wt_mcrs_path)
    # else:
    #     adata_wt_mcrs = None

    output_figures_dir = os.path.join(out, "figures")
    os.makedirs(output_figures_dir, exist_ok=True)

    if not adata_out:
        adata_out = os.path.join(adata_dir, "adata_cleaned.h5ad")

    adata.var["mcrs_id"] = adata.var.index

    if adata_path_normal_genome:
        if isinstance(adata_path_normal_genome, anndata.AnnData):
            adata_normal_genome = adata_path_normal_genome
            adata_normal_dir = "."
        elif isinstance(adata_path_normal_genome, str):
            adata_normal_genome = sc.read_h5ad(adata_path_normal_genome)
            adata_normal_dir = os.path.dirname(adata_path_normal_genome)
        else:
            raise ValueError("adata_path_normal_genome must be a string (file path) or an AnnData object.")

        if not adata_normal_genome_output_path:
            adata_normal_genome_output_path = os.path.join(adata_normal_dir, "adata_normal_genome_cleaned.h5ad")

    if mutation_metadata_df and type(mutation_metadata_df) == str and os.path.exists(mutation_metadata_df):
        mutation_metadata_df = pd.read_csv(mutation_metadata_df, index_col=0, usecols=mutation_metadata_df_columns)

    original_var_names = adata.var_names.copy()

    if mutation_metadata_df or id_to_header_csv:
        if mutation_metadata_df and type(mutation_metadata_df) == pd.DataFrame:
            df_to_merge = mutation_metadata_df
        elif id_to_header_csv and type(id_to_header_csv) == str and os.path.exists(id_to_header_csv):
            id_to_header_df = pd.read_csv(id_to_header_csv, index_col=0)
            df_to_merge = id_to_header_df

        adata.var = adata.var.merge(df_to_merge, on=mcrs_id_column, how="left")
        adata.var_names = original_var_names

    # TODO: uncomment once tested
    # if split_reads_by_Ns:
    #     # TODO: test this
    #     adata = decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode(
    #         adata,
    #         fastq=data_fastq,
    #         kb_count_out=kb_count_out_mutant,
    #         t2g=mcrs_t2g,
    #         mm=mm,
    #         bustools=bustools,
    #         split_Ns=split_reads_by_Ns,
    #         paired_end_fastqs=False,
    #         paired_end_suffix_length=2,
    #         technology=technology,
    #         keep_only_insertions=True,
    #     )

    # if dlist_file:
    #     # TODO: test this
    #     adata = increment_adata_based_on_dlist_fns(
    #         adata=adata,
    #         mcrs_fasta=mcrs_fasta,
    #         dlist_fasta=dlist_fasta,
    #         kb_count_out=kb_count_out_mutant,
    #         index=mcrs_index,
    #         t2g=mcrs_t2g,
    #         fastq=data_fastq,
    #         newer_kallisto=newer_kallisto,
    #         k=k,
    #         mm=mm,
    #         bustools=bustools,
    #     )

    # if qc_against_gene_matrix:
    #     adata = adjust_mutation_adata_by_normal_gene_matrix(
    #         adata, 
    #         kb_output_mutation=kb_count_out_mutant, 
    #         kb_output_standard=kb_count_out_normal_genome, 
    #         id_to_header_csv=id_to_header_csv, 
    #         mutation_metadata_csv=mutation_metadata_df, 
    #         adata_out=None, 
    #         t2g_mutation=mcrs_t2g, t2g_standard=None, 
    #         fastq_file_list=data_fastq, 
    #         mm=mm, 
    #         union=False, 
    #         technology=technology, 
    #         parity="single", 
    #         bustools=bustools,
    #         ignore_barcodes=ignore_barcodes,
    #         verbose=verbose
    #     )

    if ignore_barcodes and adata.shape[0] > 1:
        # Sum across barcodes (rows)
        summed_data = adata.X.sum(axis=0)
        
        # Retain the first barcode
        first_barcode = adata.obs_names[0]

        # Create a new AnnData object
        new_adata = anndata.AnnData(
            X=summed_data.reshape(1, -1),  # Reshape to (1, n_features)
            obs=adata.obs.iloc[[0]].copy(),  # Copy the first barcode's metadata
            var=adata.var.copy()             # Copy the original feature metadata
        )

        # Update the obs_names to reflect the first barcode
        new_adata.obs_names = [first_barcode]
        adata = new_adata.copy()

    # set all count values below min_counts to 0
    if min_counts is not None:
        adata.X = adata.X.multiply(adata.X >= min_counts)

    # # remove 0s for memory purposes
    # adata.X.eliminate_zeros()

    if use_binary_matrix:
        adata.X = (adata.X > 0).astype(int)

    # TODO: make sure the adata objects are in the same order (relevant for both bulk and sc)
    if adata_path_normal_genome:
        if technology not in non_single_cell_technologies:
            if filter_cells_by_min_counts:
                if type(filter_cells_by_min_counts) != int:  # ie True for automatic
                    from kneed import KneeLocator

                    umi_counts = np.array(adata_normal_genome.X.sum(axis=1)).flatten()
                    umi_counts_sorted = np.sort(umi_counts)[::-1]  # Sort in descending order for the knee plot

                    # Step 2: Use KneeLocator to find the cutoff
                    knee_locator = KneeLocator(
                        range(len(umi_counts_sorted)),
                        umi_counts_sorted,
                        curve="convex",
                        direction="decreasing",
                    )
                    filter_cells_by_min_counts = umi_counts_sorted[knee_locator.knee]
                    plot_knee_plot(
                        umi_counts_sorted=umi_counts_sorted,
                        knee_locator=knee_locator,
                        min_counts_assessed_by_knee_plot=filter_cells_by_min_counts,
                        output_file=f"{output_figures_dir}/knee_plot.png",
                    )
                sc.pp.filter_cells(adata_normal_genome, min_counts=filter_cells_by_min_counts)  # filter cells by min counts
            if filter_cells_by_min_genes:
                sc.pp.filter_cells(adata_normal_genome, min_genes=filter_cells_by_min_genes)  # filter cells by min genes
            if filter_genes_by_min_cells:
                sc.pp.filter_genes(adata_normal_genome, min_cells=filter_genes_by_min_cells)  # filter genes by min cells
            if filter_cells_by_max_mt_content:
                has_mt_genes = adata_normal_genome.var_names.str.startswith("MT-").any()
                if has_mt_genes:
                    adata_normal_genome.var["mt"] = adata_normal_genome.var_names.str.startswith("MT-")
                else:
                    mito_ensembl_ids = sc.queries.mitochondrial_genes("hsapiens", attrname="ensembl_gene_id")
                    mito_genes = set(mito_ensembl_ids["ensembl_gene_id"].values)

                    adata_base_var_names = adata_normal_genome.var_names.str.split(".").str[0]  # Removes minor version from var names
                    mito_genes_base = {gene.split(".")[0] for gene in mito_genes}  # Removes minor version from mito_genes

                    # Identify mitochondrial genes in adata.var using the stripped version of gene IDs
                    adata_normal_genome.var["mt"] = adata_base_var_names.isin(mito_genes_base)

                mito_counts = adata_normal_genome[:, adata_normal_genome.var["mt"]].X.sum(axis=1)

                # Calculate total counts per cell
                total_counts = adata_normal_genome.X.sum(axis=1)

                # Calculate percent mitochondrial gene expression per cell
                adata_normal_genome.obs["percent_mito"] = np.array(mito_counts / total_counts * 100).flatten()

                adata_normal_genome.obs["total_counts"] = adata_normal_genome.X.sum(axis=1).A1
                sc.pp.calculate_qc_metrics(
                    adata_normal_genome,
                    qc_vars=["mt"],
                    percent_top=None,
                    log1p=False,
                    inplace=True,
                )

                sc.pl.violin(
                    adata_normal_genome,
                    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                    jitter=0.4,
                    multi_panel=True,
                    save=True,
                )

                # * TODO: move violin plot file path
                violin_plot_path = f"{output_figures_dir}/qc_violin_plot.png"

                adata_normal_genome = adata_normal_genome[adata_normal_genome.obs.pct_counts_mt < 5, :].copy()  # filter cells by high MT content

                adata.obs["percent_mito"] = adata_normal_genome.obs["percent_mito"]
                adata.obs["total_counts"] = adata_normal_genome.obs["total_counts"]
            if doublet_detection:
                sc.pp.scrublet(adata_normal_genome, batch_key="sample")  # filter doublets
                adata.obs["predicted_doublet"] = adata_normal_genome.obs["predicted_doublet"]
                if remove_doublets:
                    adata_normal_genome = adata_normal_genome[~adata_normal_genome.obs["predicted_doublet"], :].copy()
                    adata = adata[~adata.obs["predicted_doublet"], :].copy()

            common_cells = adata.obs_names.intersection(adata_normal_genome.obs_names)
            adata = adata[common_cells, :].copy()

        # do cpm
        if cpm_normalization and not use_binary_matrix:  # normalization not needed for binary matrix
            total_counts = adata_normal_genome.X.sum(axis=1)
            cpm_factor = total_counts / 1e6

            adata.X = adata.X / cpm_factor[:, None]  # Reshape to make cpm_factor compatible with adata.X
            adata.obs["cpm_factor"] = cpm_factor

    if drop_empty_columns:
        # Identify columns (genes) with non-zero counts across samples
        nonzero_gene_mask = np.array((adata.X != 0).sum(axis=0)).flatten() > 0

        # Filter the AnnData object to keep only genes with non-zero counts across samples
        adata = adata[:, nonzero_gene_mask]

    # include or exclude certain genes
    if mcrs_id_set_to_exclusively_keep:
        adata = remove_adata_columns(
            adata,
            values_of_interest=mcrs_id_set_to_exclusively_keep,
            operation="keep",
            var_column_name="mcrs_id",
        )

    if mcrs_id_set_to_exclude:
        adata = remove_adata_columns(
            adata,
            values_of_interest=mcrs_id_set_to_exclude,
            operation="exclude",
            var_column_name="mcrs_id",
        )

    if transcript_set_to_exclusively_keep:
        adata = remove_adata_columns(
            adata,
            values_of_interest=transcript_set_to_exclusively_keep,
            operation="keep",
            var_column_name="seq_ID",
        )

    if transcript_set_to_exclude:
        adata = remove_adata_columns(
            adata,
            values_of_interest=transcript_set_to_exclude,
            operation="exclude",
            var_column_name="seq_ID",
        )

    if gene_set_to_exclusively_keep:
        adata = remove_adata_columns(
            adata,
            values_of_interest=gene_set_to_exclusively_keep,
            operation="keep",
            var_column_name="gene_name",
        )

    if gene_set_to_exclude:
        adata = remove_adata_columns(
            adata,
            values_of_interest=gene_set_to_exclude,
            operation="exclude",
            var_column_name="gene_name",
        )


    adata.var["mcrs_count"] = adata.X.sum(axis=0).A1 if hasattr(adata.X, "A1") else np.asarray(adata.X.sum(axis=0)).flatten()

    if adata_path_normal_genome:
        adata_normal_genome.write(adata_normal_genome_output_path)

    adata.write(adata_out)


    if output_type == "path":
        return adata_out
    elif output_type == "AnnData":
        return adata
