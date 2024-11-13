import os
import subprocess
import varseek as vk

def count(
    rnaseq_fastq_files,
    mutation_index,
    t2g_vk,
    k=31,
    threads=2,
    trim_edges_off_reads=True,
    minimum_base_quality_trim_reads=13,
    qualified_quality_phred=0,
    unqualified_percent_limit=100,
    n_base_limit=None,
    replace_low_quality_bases_with_N=True,
    minimum_base_quality_replace_with_N=13,
    split_reads_by_Ns=True,
    run_fastqc=False,
    assay="bulk",  # TODO: implement
    parity="single",
    strand="unstranded",  # "forward", "reverse", or "unstranded"
    minimum_count_filter=None,  # TODO: set a default
    use_binary_matrix=False,
    drop_zero_columns=False,
    filter_cells_by_min_counts=False,  # TODO: automatic knee plot
    filter_cells_by_min_genes=3,
    filter_genes_by_min_cells=200,
    filter_cells_by_max_mt_content=20,
    doublet_detection=True,
    remove_doublets=False,
    do_cpm_normalization=True,
    mutation_metadata_df_columns=None,
    seqtk="seqtk",
    fastp="fastp",
    mutation_metadata_df_path=None,
    standard_index=None,
    standard_t2g=None,
    out_dir_base=".",
):
    """
    Required input arguments:
    - rnaseq_fastq_files     (list) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]
    - mutation_index         (str)  Path to mutation index
    - t2g_vk                 (str)  Path to t2g file

    Optional input arguments:
    - k                      (int)  k-mer size
    - threads                (int)  Number of threads
    - trim_edges_off_reads   (bool) If True, trim edges off reads
    - minimum_base_quality_trim_reads (int) Minimum base quality to trim reads
    - qualified_quality_phred (int)  Phred score for qualified quality
    - unqualified_percent_limit (int)  Percent of unqualified quality bases
    - n_base_limit           (int)  Maximum number of N bases allowed
    - replace_low_quality_bases_with_N (bool) If True, replace low quality bases with N
    - minimum_base_quality_replace_with_N (int) Minimum base quality to replace with N
    - split_reads_by_Ns      (bool) If True, split reads by Ns
    - run_fastqc             (bool) If True, run FastQC and MultiQC
    - assay                  (str)  "bulk" or "sc"
    - parity                 (str)  "single" or "paired"
    - strand                 (str)  "forward", "reverse", or "unstranded"
    - minimum_count_filter   (int)  Minimum count filter
    - use_binary_matrix      (bool) Use binary matrix
    - drop_zero_columns      (bool) Drop zero columns
    - filter_cells_by_min_counts (bool) Filter cells by minimum counts
    - filter_cells_by_min_genes (int)  Filter cells by minimum genes
    - filter_genes_by_min_cells (int)  Filter genes by minimum cells
    - filter_cells_by_max_mt_content (int)  Filter cells by maximum mitochondrial content
    - doublet_detection      (bool) Doublet detection
    - remove_doublets        (bool) Remove doublets
    - do_cpm_normalization   (bool) Do CPM normalization
    - mutation_metadata_df_columns (list) List of mutation metadata columns
    - seqtk                  (str)  Path to seqtk
    - fastp                  (str)  Path to fastp
    - mutation_metadata_df_path (str)  Path to mutation metadata dataframe
    - standard_index         (str)  Path to standard index
    - standard_t2g           (str)  Path to standard t2g file
    - out_dir_base           (str)  Output directory
    """
    
    # mutation_index = f"{out_dir_notebook}/mutation_reference.idx"
    # t2g_vk = os.path.join(out_dir_notebook, "t2g_filtered.txt")

    out_dir_notebook = os.path.join(out_dir_base, "vk_build_pipeline_t2t")
    kb_count_out = f"{out_dir_notebook}/kb_count_out"
    kb_count_out_standard_index = f"{out_dir_notebook}/kb_count_out_standard"
    vk_summarize_output_dir = f"{out_dir_notebook}/vk_summarize"
    fastqc_out_dir = f"{out_dir_notebook}/fastqc_out"

    os.makedirs(out_dir_base, exist_ok=True)
    os.makedirs(out_dir_notebook, exist_ok=True)
    os.makedirs(kb_count_out, exist_ok=True)
    os.makedirs(kb_count_out_standard_index, exist_ok=True)
    os.makedirs(vk_summarize_output_dir, exist_ok=True)
    os.makedirs(fastqc_out_dir, exist_ok=True)

    if type(rnaseq_fastq_files) is str:
        rnaseq_fastq_files = [rnaseq_fastq_files]

    adata_path = f"{kb_count_out}/counts_unfiltered/adata.h5ad"
    adata_path_normal_genome = f"{kb_count_out_standard_index}/counts_unfiltered/adata.h5ad"

    adata_normal_dir = os.path.dirname(adata_path_normal_genome)
    adata_normal_genome_output_path = os.path.join(adata_normal_dir, "adata_normal_genome_cleaned.h5ad")

    rnaseq_fastq_files_list_dict = vk.fastqpp(
        rnaseq_fastq_files_list=rnaseq_fastq_files,
        trim_edges_off_reads=trim_edges_off_reads,
        run_fastqc=run_fastqc,
        replace_low_quality_bases_with_N=replace_low_quality_bases_with_N,
        split_reads_by_Ns=split_reads_by_Ns,
        parity=parity,
        fastqc_out_dir=fastqc_out_dir,
        minimum_base_quality_trim_reads=minimum_base_quality_trim_reads,
        qualified_quality_phred=qualified_quality_phred,
        unqualified_percent_limit=unqualified_percent_limit,
        n_base_limit=n_base_limit,
        minimum_length=k,
        minimum_base_quality_replace_with_N=minimum_base_quality_replace_with_N,
        fastp=fastp,
        seqtk=seqtk,
        delete_intermediate_files=False,
    )

    rnaseq_fastq_files_quality_controlled = rnaseq_fastq_files_list_dict["trimmed"]
    rnaseq_fastq_files = rnaseq_fastq_files_list_dict["final"]

    # if trim_edges_off_reads:
    #     rnaseq_fastq_files_quality_controlled = trim_edges_off_reads_fastq_list(rnaseq_fastq_files=rnaseq_fastq_files, parity=parity, minimum_base_quality_trim_reads=minimum_base_quality_trim_reads, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, n_base_limit=n_base_limit, length_required=length_required)
    # else:
    #     rnaseq_fastq_files_quality_controlled = rnaseq_fastq_files

    # if run_fastqc:
    #     run_fastqc_and_multiqc(rnaseq_fastq_files_quality_controlled, fastqc_out_dir)

    # if replace_low_quality_bases_with_N:
    #     rnaseq_fastq_files_replace_low_quality_bases_with_N = replace_low_quality_bases_with_N_list(rnaseq_fastq_files_quality_controlled=rnaseq_fastq_files_quality_controlled, minimum_base_quality_replace_with_N=minimum_base_quality_replace_with_N, seqtk=seqtk)
    # else:
    #     rnaseq_fastq_files_replace_low_quality_bases_with_N = rnaseq_fastq_files_quality_controlled

    # if split_reads_by_Ns:
    #     rnaseq_fastq_files_final = split_reads_by_N_list(rnaseq_fastq_files_replace_low_quality_bases_with_N, minimum_sequence_length=k)
    # else:
    #     rnaseq_fastq_files_final = rnaseq_fastq_files_replace_low_quality_bases_with_N

    # # kb count

    # TODO: incorporate assay bulk vs sc in here
    if not os.path.exists(kb_count_out) or len(os.listdir(kb_count_out)) == 0:
        kb_count_command = [
            "kb",
            "count",
            "-t",
            str(threads),
            "-k",
            str(k),
            "-i",
            mutation_index,
            "-g",
            t2g_vk,
            "-x",
            "bulk",
            "--num",
            "--h5ad",
            "--parity",
            "single",
            "--strand",
            strand,
            "-o",
            kb_count_out,
        ] + rnaseq_fastq_files
        subprocess.run(kb_count_command, check=True)

    # # Optionally, kb ref and count on normal genome

    if not os.path.exists(standard_index) or not os.path.exists(standard_t2g):
        kb_ref_command = [
            "kb",
            "ref",
            "-t",
            str(threads),
            "-i",
            standard_index,
            "-g",
            standard_t2g,
            "-d",
            "human",
        ]
        subprocess.run(kb_ref_command, check=True)

    # TODO: incorporate assay bulk vs sc in here
    if (
        not os.path.exists(kb_count_out_standard_index)
        or len(os.listdir(kb_count_out_standard_index)) == 0
    ):
        kb_count_standard_index_command = [
            "kb",
            "count",
            "-t",
            str(threads),
            "-k",
            str(k),
            "-i",
            standard_index,
            "-g",
            standard_t2g,
            "-x",
            "bulk",
            "--h5ad",
            "--parity",
            parity,
            "--strand",
            strand,
            "-o",
            kb_count_out_standard_index,
        ] + rnaseq_fastq_files_quality_controlled
        subprocess.run(kb_count_standard_index_command, check=True)

    # # vk clean
    adata_path_clean = vk.clean(
        adata_path,
        output_figures_dir=f"{out_dir_notebook}/vk_clean_figures",
        mutation_metadata_df=mutation_metadata_df_path,
        mutation_metadata_df_columns=mutation_metadata_df_columns,
        minimum_count_filter=minimum_count_filter,
        use_binary_matrix=use_binary_matrix,
        drop_zero_columns=drop_zero_columns,
        filter_cells_by_min_counts=filter_cells_by_min_counts,
        filter_cells_by_min_genes=filter_cells_by_min_genes,
        filter_genes_by_min_cells=filter_genes_by_min_cells,
        filter_cells_by_max_mt_content=filter_cells_by_max_mt_content,
        doublet_detection=doublet_detection,
        remove_doublets=remove_doublets,
        do_cpm_normalization=do_cpm_normalization,
        adata_path_normal_genome=adata_path_normal_genome,
        mcrs_id_column="mcrs_id",
        adata_normal_genome_output_path=adata_normal_genome_output_path,
        verbose=False,
    )

    # # vk summarize
    vk.summarize(
        adata_path_clean,
        assay=assay,
        output_dir=vk_summarize_output_dir,
        overwrite=False,
        top_values=10,
    )

    vk_count_output_dict = {}
    vk_count_output_dict["adata_path"] = adata_path
    vk_count_output_dict["adata_path_normal_genome"] = adata_path_normal_genome
    vk_count_output_dict["adata_path_clean"] = adata_path_clean
    vk_count_output_dict["adata_path_normal_genome_clean"] = adata_normal_genome_output_path
    vk_count_output_dict["vk_summarize_output_dir"] = vk_summarize_output_dir

    return vk_count_output_dict
