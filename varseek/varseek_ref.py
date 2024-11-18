import os
import subprocess
import varseek as vk


def ref(
    mutations="cosmic_cmc",
    sequences="cdna",
    w=54,
    k=59,
    threads=4,
    remove_Ns=True,
    strandedness=False,
    cosmic_version=100,
    columns_to_include="all",
    dlist_reference_source="t2t",
    near_splice_junction_threshold=10,
    save_exploded_df=False,
    fasta_filters=[
        "dlist_substring-equal=none",  # filter out mutations which are a substring of the reference genome
        "pseudoaligned_to_human_reference_despite_not_truly_aligning-isnottrue",  # filter out mutations which pseudoaligned to human genome despite not truly aligning
        "dlist-equal=none",  # *** erase eventually when I want to d-list  # filter out mutations which are capable of being d-listed (given that I filter out the substrings above)
        "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference-max=999999",  # filter out mutations which overlap with other MCRSs in the reference
        "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference-max=999999",  # filter out mutations which overlap with other MCRSs in the reference
        "longest_homopolymer_length-max=999999",  # filters out MCRSs with repeating single nucleotide - eg 6
        "triplet_complexity-min=0",  # filters out MCRSs with repeating triplets - eg 0.2
    ],  # TODO: edit
    dlist=False,  # path to dlist fasta file or "None" (including the quotes)
    out_dir_base=".",
    run_name="kb_ref_run",
    mutations_csv=None,
    bowtie_path="bowtie2",
    verbose=False,
    reference_cdna_fasta=None,
    reference_genome_fasta=None,
    gtf_path=None,
):
    out_dir_notebook = os.path.join(out_dir_base, run_name)
    reference_out_dir = os.path.join(out_dir_base, "reference")

    os.makedirs(out_dir_base, exist_ok=True)
    os.makedirs(out_dir_notebook, exist_ok=True)
    os.makedirs(reference_out_dir, exist_ok=True)

    if remove_Ns:
        max_ambiguous_vk = 0
    else:
        max_ambiguous_vk = None

    merge_identical_rc = not strandedness

    vk_build_mcrs_fa_path = os.path.join(out_dir_notebook, "mcrs.fa")
    update_df_out = os.path.join(out_dir_notebook, "mutation_metadata_df.csv")
    os.makedirs(out_dir_notebook, exist_ok=True)

    assert k >= w + 1, "k must be greater than or equal to w + 1"

    id_to_header_csv = os.path.join(out_dir_notebook, "id_to_header_mapping.csv")
    mutation_metadata_df_out_path_vk_info = os.path.join(
        out_dir_notebook, "mutation_metadata_df_updated_vk_info.csv"
    )
    mutation_index = f"{out_dir_notebook}/mutation_reference.idx"
    dlist_fasta = f"{out_dir_notebook}/dlist.fa"

    mcrs_fasta_vk_filter = os.path.join(out_dir_notebook, "mcrs_filtered.fa")
    output_metadata_df_vk_filter = os.path.join(
        out_dir_notebook, "mutation_metadata_df_filtered.csv"
    )
    dlist_fasta_vk_filter = os.path.join(out_dir_notebook, "dlist_filtered.fa")
    t2g_vk_filter = os.path.join(out_dir_notebook, "t2g_filtered.txt")
    id_to_header_csv_vk_filter = os.path.join(
        out_dir_notebook, "id_to_header_mapping_filtered.csv"
    )

    if dlist:
        dlist_kb_argument = dlist_fasta_vk_filter
    else:
        dlist_kb_argument = "None"

    # # vk build

    vk.build(
        sequences=sequences,
        mutations=mutations,
        out=out_dir_notebook,
        reference_out=reference_out_dir,
        w=w,
        remove_seqs_with_wt_kmers=True,
        optimize_flanking_regions=True,
        min_seq_len=k,
        max_ambiguous=max_ambiguous_vk,
        merge_identical=True,
        merge_identical_rc=merge_identical_rc,
        cosmic_release=cosmic_version,
        cosmic_email=os.getenv("COSMIC_EMAIL"),
        cosmic_password=os.getenv("COSMIC_PASSWORD"),
        create_t2g=True,
        update_df=True,
        update_df_out=update_df_out,
        verbose=verbose,
    )

    # # vk info

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

    # # vk filter

    vk.filter(
        mutation_metadata_df_path=mutation_metadata_df_out_path_vk_info,
        output_mcrs_fasta=mcrs_fasta_vk_filter,
        output_metadata_df=output_metadata_df_vk_filter,
        dlist_fasta=dlist_fasta,
        output_dlist_fasta=dlist_fasta_vk_filter,
        create_t2g=True,
        output_t2g=t2g_vk_filter,
        id_to_header_csv=id_to_header_csv,
        output_id_to_header_csv=id_to_header_csv_vk_filter,
        verbose=True,
        return_df=False,
        filters=fasta_filters,
    )

    # # kb ref

    if not os.path.exists(mutation_index):
        kb_ref_command = [
            "kb",
            "ref",
            "--workflow",
            "custom",
            "-t",
            str(threads),
            "-i",
            mutation_index,
            "--d-list",
            dlist_kb_argument,
            "-k",
            str(k),
            mcrs_fasta_vk_filter,
        ]
        subprocess.run(kb_ref_command, check=True)

    vk_ref_output_dict = {}
    vk_ref_output_dict["index"] = mutation_index
    vk_ref_output_dict["t2g"] = t2g_vk_filter
    vk_ref_output_dict["dlist"] = dlist_fasta_vk_filter
    vk_ref_output_dict["id_to_header"] = id_to_header_csv_vk_filter
    vk_ref_output_dict["mcrs_fasta"] = mcrs_fasta_vk_filter
    vk_ref_output_dict["metadata_df"] = output_metadata_df_vk_filter

    return vk_ref_output_dict
