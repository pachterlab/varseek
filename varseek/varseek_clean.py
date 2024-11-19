import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata

from varseek.utils import (
    plot_knee_plot,
    increment_adata_based_on_dlist_fns,
    decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode,
    remove_adata_columns,
    adjust_mutation_adata_by_normal_gene_matrix
)


def clean(
    adata_path,
    adata_output_path=None,
    output_figures_dir=None,
    id_to_header_csv=None,
    mutation_metadata_df=None,
    mutation_metadata_df_columns=None,
    minimum_count_filter=0,
    use_binary_matrix=False,
    drop_zero_columns=False,
    assay="bulk",
    adjust_mutation_adata_by_normal_gene_matrix_information=False,  # * change to True
    filter_cells_by_min_counts=None,
    filter_cells_by_min_genes=None,
    filter_genes_by_min_cells=None,
    filter_cells_by_max_mt_content=None,
    doublet_detection=None,
    remove_doublets=None,
    do_cpm_normalization=None,
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
    verbose=False,
    **kwargs,
):

    if isinstance(adata_path, anndata.AnnData):
        adata = adata_path
        adata_dir = "."
        output_type = "AnnData"
    elif isinstance(adata_path, str):
        adata = sc.read_h5ad(adata_path)
        adata_dir = os.path.dirname(adata_path)
        output_type = "path"
    else:
        raise ValueError("adata_path must be a string (file path) or an AnnData object.")

    # if kb_count_out_wt_mcrs_counterpart:
    #     adata_wt_mcrs_path = f"{kb_count_out_wt_mcrs_counterpart}/counts_unfiltered/adata.h5ad"
    #     adata_wt_mcrs = sc.read_h5ad(adata_wt_mcrs_path)
    # else:
    #     adata_wt_mcrs = None

    if not output_figures_dir:
        output_figures_dir = os.path.join(adata_dir, "figures")

    os.makedirs(output_figures_dir, exist_ok=True)

    if not adata_output_path:
        adata_output_path = os.path.join(adata_dir, "adata_cleaned.h5ad")

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
    if split_reads_by_Ns:
        # TODO: test this
        adata = decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode(
            adata,
            fastq=data_fastq,
            kb_count_out=kb_count_out_mutant,
            t2g=mcrs_t2g,
            mm=mm,
            bustools=bustools,
            split_Ns=split_reads_by_Ns,
            paired_end_fastqs=False,
            paired_end_suffix_length=2,
            assay="bulk",
            keep_only_insertions=True,
        )

    if dlist_file:
        # TODO: test this
        adata = increment_adata_based_on_dlist_fns(
            adata=adata,
            mcrs_fasta=mcrs_fasta,
            dlist_fasta=dlist_fasta,
            kb_count_out=kb_count_out_mutant,
            index=mcrs_index,
            t2g=mcrs_t2g,
            fastq=data_fastq,
            newer_kallisto=newer_kallisto,
            k=k,
            mm=mm,
            bustools=bustools,
        )

    if adjust_mutation_adata_by_normal_gene_matrix_information:
        adata = adjust_mutation_adata_by_normal_gene_matrix(
            adata, kb_output_mutation=kb_count_out_mutant, kb_output_standard=kb_count_out_normal_genome, id_to_header_csv=id_to_header_csv, mutation_metadata_csv=mutation_metadata_df, adata_output_path=None, t2g_mutation=mcrs_t2g, t2g_standard=None, fastq_file_list=data_fastq, mm=mm, union=False, assay=assay, parity="single", bustools=bustools
        )

    # set all count values below minimum_count_filter to 0
    if minimum_count_filter is not None:
        adata.X = adata.X.multiply(adata.X >= minimum_count_filter)

    # # remove 0s for memory purposes
    # adata.X.eliminate_zeros()

    if use_binary_matrix:
        adata.X = (adata.X > 0).astype(int)

    # TODO: make sure the adata objects are in the same order (relevant for both bulk and sc)
    if adata_path_normal_genome:
        if assay == "sc":
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
        if do_cpm_normalization and not use_binary_matrix:  # normalization not needed for binary matrix
            total_counts = adata_normal_genome.X.sum(axis=1)
            cpm_factor = total_counts / 1e6

            adata.X = adata.X / cpm_factor[:, None]  # Reshape to make cpm_factor compatible with adata.X
            adata.obs["cpm_factor"] = cpm_factor

    if drop_zero_columns:
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

    adata.write(adata_output_path)

    if output_type == "path":
        return adata_output_path
    elif output_type == "AnnData":
        return adata
