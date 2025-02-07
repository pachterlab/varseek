import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from pdb import set_trace as st

import gget
import pandas as pd
import pytest

import varseek as vk
from varseek.utils import add_mutation_type, convert_mutation_cds_locations_to_cdna

from .conftest import (
    compare_two_dataframes_without_regard_for_order_of_rows_or_columns,
    compare_two_fastas_without_regard_for_order_of_entries,
    compare_two_id_to_header_mappings,
    compare_two_t2gs,
)

ground_truth_folder = "/home/jrich/data/varseek_data_fresh/pytest_ground_truth"
cosmic_csv_path_starting = "/home/jrich/data/varseek_data_fresh/pytest_ground_truth/CancerMutationCensus_AllData_v100_GRCh37_mutation_workflow_with_cdna_subsampled_pytest.csv"
reference_folder_parent = "/home/jrich/data/varseek_data_fresh/reference"
reference_folder = "/home/jrich/data/varseek_data_fresh/reference/ensembl_grch37_release93"
bowtie_path="/home/jrich/opt/bowtie2-2.5.4/bowtie2-2.5.4-linux-x86_64"
sample_size=2000
columns_to_drop_info_filter = None  # ["nearby_mutations", "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference", "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference", "overlapping_kmers", "mcrs_items_with_overlapping_kmers_in_mcrs_reference", "kmer_overlap_in_mcrs_reference"]
make_new_gt = False


@pytest.fixture
def genome_and_gtf_files(tmp_path):
    global reference_folder, make_new_gt, ground_truth_folder

    if (not os.path.exists(ground_truth_folder) or not os.listdir(ground_truth_folder)):
        if not make_new_gt:
            pytest.skip("Ground truth folder is missing or empty, and make_new_gt is False. Skipping test.")


    # Setup code for cds_file, e.g., creating a mock file or providing the actual path
    genome_file_name = "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
    gtf_file_name = "Homo_sapiens.GRCh37.87.gtf"

    genome_path = f"{reference_folder}/{genome_file_name}"
    gtf_path = f"{reference_folder}/{gtf_file_name}"

    if not os.path.exists(genome_path) or not os.path.exists(gtf_path):
        genome_path = tmp_path / "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
        gtf_path = tmp_path / "Homo_sapiens.GRCh37.87.gtf"
        gget_ref_command = f"gget ref -w dna,gtf -r 93 --out_dir {tmp_path} -d human_grch37"
        subprocess.run(gget_ref_command, shell=True, check=True)

    return str(genome_path), str(gtf_path)

@pytest.fixture
def cds_and_cdna_files(tmp_path):
    global reference_folder, ground_truth_folder, make_new_gt

    if (not os.path.exists(ground_truth_folder) or not os.listdir(ground_truth_folder)):
        if not make_new_gt:
            pytest.skip("Ground truth folder is missing or empty, and make_new_gt is False. Skipping test.")


    # Setup code for cds_file, e.g., creating a mock file or providing the actual path
    cds_file_name = "Homo_sapiens.GRCh37.cds.all.fa"
    cdna_file_name = "Homo_sapiens.GRCh37.cdna.all.fa"

    cds_path = f"{reference_folder}/{cds_file_name}"
    cdna_path = f"{reference_folder}/{cdna_file_name}"
    if not os.path.exists(cds_path) or not os.path.exists(cdna_path):
        cds_path = tmp_path / "Homo_sapiens.GRCh37.cds.all.fa"
        cdna_path = tmp_path / "Homo_sapiens.GRCh37.cdna.all.fa"
        gget_ref_command = f"gget ref -w cdna,cds -r 93 --out_dir {tmp_path} -d human_grch37"
        subprocess.run(gget_ref_command, shell=True, check=True)

    return str(cds_path), str(cdna_path)

@pytest.fixture
def cosmic_csv_path(cds_and_cdna_files, tmp_path):
    global cosmic_csv_path_starting, sample_size, make_new_gt, ground_truth_folder

    if (not os.path.exists(ground_truth_folder) or not os.listdir(ground_truth_folder)):
        if not make_new_gt:
            pytest.skip("Ground truth folder is missing or empty, and make_new_gt is False. Skipping test.")

    cds_file, cdna_file = cds_and_cdna_files

    if os.path.exists(cosmic_csv_path_starting):
        subsampled_cosmic_csv_with_cdna_path = cosmic_csv_path_starting
    else:
        cosmic_csv_path_starting_original = cosmic_csv_path_starting
        cosmic_csv_path_starting = tmp_path / "CancerMutationCensus_AllData_Tsv_v100_GRCh37/CancerMutationCensus_AllData_v100_GRCh37.csv"
        if not os.path.exists(cosmic_csv_path_starting):
            gget.cosmic(
                None,
                grch_version=37,
                cosmic_version=100,
                out=str(tmp_path),
                mutation_class="cancer",
                download_cosmic=True,
                keep_genome_info=True,
                remove_duplicates=True,
                email=os.getenv('COSMIC_EMAIL'),
                password=os.getenv('COSMIC_PASSWORD'),
            )

            # if gtf is not None:
            #     mutations = merge_gtf_transcript_locations_into_cosmic_csv(mutations, gtf, gtf_transcript_id_column=gtf_transcript_id_column)
            #     columns_to_keep.extend(["start_transcript_position", "end_transcript_position", "strand"])

            # if "CancerMutationCensus" in mutations or mutations == "cosmic_cmc":
            #     improve_genome_strand_information(mutations, mutation_genome_column_name="mutation_genome")

        cosmic_csv_with_cdna_path = Path(str(cosmic_csv_path_starting).replace(".csv", "_with_cdna.csv"))
        if not os.path.exists(cosmic_csv_with_cdna_path):
            convert_mutation_cds_locations_to_cdna(
                input_csv_path=cosmic_csv_path_starting,
                output_csv_path=cosmic_csv_with_cdna_path,
                cds_fasta_path=cds_file,
                cdna_fasta_path=cdna_file,
            )


        mutations = pd.read_csv(cosmic_csv_with_cdna_path)
        mutations = add_mutation_type(mutations, mut_column = "mutation_genome")

        mutation_types = ["substitution", "deletion", "insertion", "delins", "duplication", "inversion"]

        final_df = pd.DataFrame()

        for mutation_type in mutation_types:
            # Filter DataFrame for the current mutation type
            mutation_subset = mutations[mutations["mutation_type"] == mutation_type]

            sample_size_round = min(sample_size, len(mutation_subset))
            
            # Randomly sample `sample_size` rows from the subset (adjusts if less than sample_size)
            sample = mutation_subset.sample(n=min(sample_size_round, len(mutation_subset)), random_state=42)
            
            # Append the sample to the final DataFrame
            final_df = pd.concat([final_df, sample], ignore_index=True)

        subsampled_cosmic_csv_with_cdna_path = Path(str(cosmic_csv_with_cdna_path).replace(".csv", "_subsampled_pytest.csv"))
        final_df.to_csv(subsampled_cosmic_csv_with_cdna_path, index=False)

        if make_new_gt:
            os.makedirs(os.path.dirname(cosmic_csv_path_starting_original), exist_ok=True)
            shutil.copy(subsampled_cosmic_csv_with_cdna_path, cosmic_csv_path_starting_original)

    return subsampled_cosmic_csv_with_cdna_path


#* note: temp files will be deleted upon completion of the test or running into an error - to debug with a temp file, place a breakpoint before the error occurs
def test_file_processing(cosmic_csv_path, cds_and_cdna_files, genome_and_gtf_files):
    global ground_truth_folder, reference_folder_parent, bowtie_path, gtf_path, reference_genome_fasta, make_new_gt

    # skip this run if you don't have the ground truth and are not making it
    if not os.path.exists(ground_truth_folder) or not os.listdir(ground_truth_folder):
        if not make_new_gt:
            pytest.skip("Ground truth folder is missing or empty, and make_new_gt is False. Skipping test.")

    w = 54
    k = 55
    max_ambiguous_vk = 0
    strandedness = False
    fasta_filters = [
        "dlist_substring:equal=none",  # filter out mutations which are a substring of the reference genome
        "pseudoaligned_to_human_reference_despite_not_truly_aligning:is_not_true",  # filter out mutations which pseudoaligned to human genome despite not truly aligning
        "dlist:equal=none",  #*** erase eventually when I want to d-list  # filter out mutations which are capable of being d-listed (given that I filter out the substrings above)
        "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference:less_than=999999",  # filter out mutations which overlap with other MCRSs in the reference
        "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference:less_than=999999",  # filter out mutations which overlap with other MCRSs in the reference
        "longest_homopolymer_length:less_or_equal=6",  # filters out MCRSs with repeating single nucleotide - eg 6
        "triplet_complexity:greater_or_equal=0.2"  # filters out MCRSs with repeating triplets - eg 0.2
    ]

    
    cds_file, cdna_file = cds_and_cdna_files
    reference_genome_fasta, gtf_path = genome_and_gtf_files

    if make_new_gt:
        os.makedirs(ground_truth_folder, exist_ok=True)

    with tempfile.TemporaryDirectory() as out_dir_notebook:
    
        vk.build(
            sequences=cdna_file,
            mutations=cosmic_csv_path,
            out=out_dir_notebook,
            reference_out_dir=None,
            w=w,
            remove_seqs_with_wt_kmers=True,
            optimize_flanking_regions=True,
            min_seq_len=k,
            max_ambiguous=max_ambiguous_vk,
            merge_identical=True,
            vcrs_strandedness=strandedness,
            cosmic_email = os.getenv('COSMIC_EMAIL'),
            cosmic_password = os.getenv('COSMIC_PASSWORD'),
            save_mutations_updated_csv=True
        )

        vk_build_mcrs_fa_path = os.path.join(out_dir_notebook, "mcrs.fa")
        update_df_out = os.path.join(out_dir_notebook, "CancerMutationCensus_AllData_v100_GRCh37_mutation_workflow_with_cdna_subsampled_pytest_updated.csv")
        id_to_header_csv=os.path.join(out_dir_notebook, "id_to_header_mapping.csv")
        t2g_path = os.path.join(out_dir_notebook, "mcrs_t2g.txt")

        vk_build_mcrs_fa_path_ground_truth = f"{ground_truth_folder}/mcrs.fa"
        update_df_out_ground_truth = f"{ground_truth_folder}/CancerMutationCensus_AllData_v100_GRCh37_mutation_workflow_with_cdna_subsampled_pytest_updated.csv"
        id_to_header_csv_ground_truth = f"{ground_truth_folder}/id_to_header_mapping.csv"
        t2g_path_ground_truth = f"{ground_truth_folder}/mcrs_t2g.txt"

        if make_new_gt:
            shutil.copy(vk_build_mcrs_fa_path, vk_build_mcrs_fa_path_ground_truth)
            shutil.copy(update_df_out, update_df_out_ground_truth)
            shutil.copy(id_to_header_csv, id_to_header_csv_ground_truth)
            shutil.copy(t2g_path, t2g_path_ground_truth)

        
        compare_two_fastas_without_regard_for_order_of_entries(vk_build_mcrs_fa_path, vk_build_mcrs_fa_path_ground_truth)

        compare_two_id_to_header_mappings(id_to_header_csv, id_to_header_csv_ground_truth)

        compare_two_t2gs(t2g_path, t2g_path_ground_truth)

        compare_two_dataframes_without_regard_for_order_of_rows_or_columns(update_df_out, update_df_out_ground_truth)

        vk.info(
            input_dir=out_dir_notebook,
            columns_to_include="all",
            mcrs_id_column="mcrs_id",
            mcrs_sequence_column="mutant_sequence",
            mcrs_source_column="mcrs_source",  # if input df has concatenated cdna and header MCRS's, then I want to know whether it came from cdna or genome
            seqid_cdna_column="seq_ID",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome  # TODO: implement these 4 column name arguments
            seqid_genome_column="chromosome",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome
            mutation_cdna_column="mutation",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome
            mutation_genome_column="mutation_genome",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome
            gtf=gtf_path,  # for distance to nearest splice junction
            mutation_metadata_df_out_path=None,
            out=out_dir_notebook,
            reference_out=reference_folder_parent,
            w=w,
            vcrs_strandedness=strandedness,
            max_ambiguous_mcrs=max_ambiguous_vk,
            max_ambiguous_reference=max_ambiguous_vk,
            bowtie_path=bowtie_path,
            near_splice_junction_threshold=10,
            threads=8,
            reference_cdna_fasta=cdna_file,
            reference_genome_fasta=reference_genome_fasta,
            mutations_csv=cosmic_csv_path,
            save_exploded_df=True,
        )

        dlist_fasta = f"{out_dir_notebook}/dlist.fa"
        mutation_metadata_df_out_path = os.path.join(out_dir_notebook, "mutation_metadata_df_vk_info.csv")
        mutation_metadata_df_out_exploded_path = os.path.join(out_dir_notebook, "mutation_metadata_df_vk_info_exploded.csv")
        mcrs_fasta_vk_filter = os.path.join(out_dir_notebook, "mcrs_filtered.fa")
        output_metadata_df_vk_filter = os.path.join(out_dir_notebook, "mutation_metadata_df_filtered.csv")
        dlist_fasta_vk_filter = os.path.join(out_dir_notebook, "dlist_filtered.fa")
        t2g_vk_filter = os.path.join(out_dir_notebook, "t2g_filtered.txt")
        id_to_header_csv_vk_filter = os.path.join(out_dir_notebook, "id_to_header_mapping_filtered.csv")

        dlist_fasta_ground_truth = f"{ground_truth_folder}/dlist.fa"
        mutation_metadata_df_out_path_ground_truth = f"{ground_truth_folder}/mutation_metadata_df.csv"
        mutation_metadata_df_out_exploded_path_ground_truth = f"{ground_truth_folder}/mutation_metadata_df_exploded.csv"
        mcrs_fasta_vk_filter_ground_truth = f"{ground_truth_folder}/mcrs_filtered.fa"
        output_metadata_df_vk_filter_ground_truth = f"{ground_truth_folder}/mutation_metadata_df_filtered.csv"
        dlist_fasta_vk_filter_ground_truth = f"{ground_truth_folder}/dlist_filtered.fa"
        t2g_vk_filter_ground_truth = f"{ground_truth_folder}/t2g_filtered.txt"
        id_to_header_csv_vk_filter_ground_truth = f"{ground_truth_folder}/id_to_header_mapping_filtered.csv"

        if make_new_gt:
            shutil.copy(dlist_fasta, dlist_fasta_ground_truth)
            shutil.copy(mutation_metadata_df_out_path, mutation_metadata_df_out_path_ground_truth)
            shutil.copy(mutation_metadata_df_out_exploded_path, mutation_metadata_df_out_exploded_path_ground_truth)
        
        compare_two_fastas_without_regard_for_order_of_entries(dlist_fasta, dlist_fasta_ground_truth)

        compare_two_dataframes_without_regard_for_order_of_rows_or_columns(mutation_metadata_df_out_path, mutation_metadata_df_out_path_ground_truth, columns_to_drop = columns_to_drop_info_filter)

        compare_two_dataframes_without_regard_for_order_of_rows_or_columns(mutation_metadata_df_out_exploded_path, mutation_metadata_df_out_exploded_path_ground_truth, columns_to_drop = columns_to_drop_info_filter)


        vk.filter(
            input_dir=out_dir_notebook,
            filters = fasta_filters,
            mutations_updated_vk_info_csv = mutation_metadata_df_out_path,
            mcrs_filtered_fasta_out=mcrs_fasta_vk_filter,
            mutations_updated_filtered_csv_out=output_metadata_df_vk_filter,
            dlist_fasta=dlist_fasta,
            dlist_filtered_fasta_out=dlist_fasta_vk_filter,
            output_t2g=t2g_vk_filter,
            id_to_header_csv=id_to_header_csv,
            id_to_header_filtered_csv_out=id_to_header_csv_vk_filter,
            verbose=True,
        )

        if make_new_gt:
            shutil.copy(mcrs_fasta_vk_filter, mcrs_fasta_vk_filter_ground_truth)
            shutil.copy(output_metadata_df_vk_filter, output_metadata_df_vk_filter_ground_truth)
            shutil.copy(dlist_fasta_vk_filter, dlist_fasta_vk_filter_ground_truth)
            shutil.copy(t2g_vk_filter, t2g_vk_filter_ground_truth)
            shutil.copy(id_to_header_csv_vk_filter, id_to_header_csv_vk_filter_ground_truth)

        compare_two_fastas_without_regard_for_order_of_entries(mcrs_fasta_vk_filter, mcrs_fasta_vk_filter_ground_truth)

        compare_two_dataframes_without_regard_for_order_of_rows_or_columns(output_metadata_df_vk_filter, output_metadata_df_vk_filter_ground_truth, columns_to_drop = columns_to_drop_info_filter)

        compare_two_fastas_without_regard_for_order_of_entries(dlist_fasta_vk_filter, dlist_fasta_vk_filter_ground_truth)

        compare_two_id_to_header_mappings(id_to_header_csv_vk_filter, id_to_header_csv_vk_filter_ground_truth)

        compare_two_t2gs(t2g_vk_filter, t2g_vk_filter_ground_truth)
