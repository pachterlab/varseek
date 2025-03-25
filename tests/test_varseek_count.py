import os
import shutil
from pathlib import Path
import sys
from datetime import datetime
import importlib

import pandas as pd
import pytest

from pdb import set_trace as st

import varseek as vk

from .conftest import (
    compare_two_anndata_objects,
    compare_two_vcfs,
    compare_two_fastqs,
    compare_two_vk_summarize_txt_files,
    compare_two_jsons
)

sample_size = 250_000  # number of reads
make_new_gt = False
store_out_in_permanent_paths = True
threads = 2
quality_control_fastqs = True
qc_against_gene_matrix = False
save_vcf = True
chunksize = None  # None or int>0

test_directory = Path(__file__).resolve().parent
ground_truth_folder = os.path.join(test_directory, "pytest_ground_truth_count")
reference_folder_parent = os.path.join(os.path.dirname(test_directory), "data", "reference")
ensembl_grch37_release93_folder = os.path.join(reference_folder_parent, "ensembl_grch37_release93")
vk_ref_folder = os.path.join(os.path.dirname(test_directory), "data", "vk_ref_out")
fastq_file = os.path.join(os.path.dirname(test_directory), "data", "ccle_data_base", "RNASeq_MELHO_SKIN", "SRR8615233_1_head.fastq")
vcf_data_csv = os.path.join(reference_folder_parent, "cosmic", "CancerMutationCensus_AllData_Tsv_v101_GRCh37", "CancerMutationCensus_AllData_v101_GRCh37_vcf_data.csv")
kb_ref_standard_workflow_folder = os.path.join(ensembl_grch37_release93_folder, "kb_ref_out_standard_workflow")
pytest_permanent_out_dir_base = test_directory / "pytest_output" / Path(__file__).stem
current_datetime = datetime.now().strftime("date_%Y_%m_%d_time_%H%M_%S")

#$ TOGGLE THIS SECTION TO HAVE THIS FILE RECOGNIZED BY PYTEST (commented out means it will be recognized, uncommented means it will be hidden)
# If "tests/test_count.py" is not explicitly in the command line arguments, skip this module. - notice that uncommenting this will hide it from vscode such that I can't press the debug button
if not any("test_varseek_count.py" in arg for arg in sys.argv):
    pytest.skip("Skipping test_varseek_count.py due to its slow nature; run this file by explicity including the file i.e., 'pytest tests/test_varseek_count.py'", allow_module_level=True)

@pytest.fixture
def out_dir(tmp_path, request):
    """Fixture that returns the appropriate output directory for each test."""
    if store_out_in_permanent_paths:
        current_test_function_name = request.node.name
        out = Path(f"{pytest_permanent_out_dir_base}/{current_datetime}/{current_test_function_name}")
    else:
        out = tmp_path / "out_vk_count"

    out.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return out



def apply_file_comparison(test_path, ground_truth_path, file_type):
    if file_type == "adata":
        compare_two_anndata_objects(test_path, ground_truth_path)
    elif file_type == "vcf":
        compare_two_vcfs(test_path, ground_truth_path)
    elif file_type == "fastq":
        compare_two_fastqs(test_path, ground_truth_path)
    elif file_type == "json":
        compare_two_jsons(test_path, ground_truth_path)
    elif file_type == "txt":
        compare_two_vk_summarize_txt_files(test_path, ground_truth_path)
    else:
        raise ValueError(f"File type {file_type} is not supported.")

# run pytest -vs tests/test_varseek_count.py::test_vk_count
#* note: temp files will be deleted upon completion of the test or running into an error - to debug with a temp file, place a breakpoint before the error occurs
def test_vk_count(out_dir):
    # global ground_truth_folder, reference_folder_parent, make_new_gt, ensembl_grch37_release93_folder, vk_ref_folder, fastq_file, kb_ref_standard_workflow_folder, quality_control_fastqs, qc_against_gene_matrix

    vcrs_index = os.path.join(vk_ref_folder, "vcrs_index.idx")
    vcrs_t2g = os.path.join(vk_ref_folder, "vcrs_t2g_filtered.txt")

    if not os.path.exists(vcrs_index) or not os.path.exists(vcrs_t2g):
        pytest.skip("vcrs_index.idx or vcrs_t2g_filtered.txt not found. Please make this index to continue (see comments below)")
        # vk.ref(variants="cosmic_cmc", sequences="cdna", w=47, k=51, dlist_reference_source="t2t", index_out=vcrs_index, t2g_out=vcrs_t2g, cosmic_email=cosmic_email, cosmic_password=cosmic_password, download=True)

    if not os.path.exists(fastq_file):
        pytest.skip("fastq_file not found. Please download it to continue (see comments below)")
        # fastq_file_link = "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR861/003/SRR8615233/SRR8615233_1.fastq.gz"  # ["ftp.sra.ebi.ac.uk/vol1/fastq/SRR861/003/SRR8615233/SRR8615233_1.fastq.gz", "ftp.sra.ebi.ac.uk/vol1/fastq/SRR861/003/SRR8615233/SRR8615233_2.fastq.gz"]
        # os.makedirs(fastqs_dir, exist_ok=True)
        # !wget -c --tries=5 --retry-connrefused -O {fastq_file_full} {fastq_file_link}
        # fastq_file = fastq_file_full.replace(".fastq.gz", "_head.fastq")
        # number_of_lines = sample_size * 4
        # !zcat {fastq_file_full} | head -{number_of_lines} > $fastq_file

    reference_genome_index = os.path.join(kb_ref_standard_workflow_folder, "index.idx")
    reference_genome_t2g = os.path.join(kb_ref_standard_workflow_folder, "t2g.txt")
    reference_genome_f1 = os.path.join(kb_ref_standard_workflow_folder, "f1.fa")
    if qc_against_gene_matrix and (not os.path.exists(reference_genome_index) or not os.path.exists(reference_genome_t2g)):
        pytest.skip("qc_against_gene_matrix is True, but reference_genome_index and/or reference_genome_t2g is missing. Please make them to continue (see comments below)")
        # reference_genome_fasta = os.path.join(ensembl_grch37_release93_folder, "Homo_sapiens.GRCh37.dna.primary_assembly.fa")
        # reference_genome_gtf = os.path.join(ensembl_grch37_release93_folder, "Homo_sapiens.GRCh37.87.gtf")
        # !gget ref -w dna,gtf -r 93 --out_dir {ensembl_grch37_release93_folder} -d human_grch37 && gunzip {reference_genome_fasta}.gz && gunzip {reference_genome_gtf}.gz
        # !kb ref -t {threads} -k {k} -i {reference_genome_index} -g {reference_genome_t2g} -f1 {reference_genome_f1} {reference_genome_fasta} {reference_genome_gtf}

    # skip this run if you don't have the ground truth and are not making it
    if not os.path.exists(ground_truth_folder) or not os.listdir(ground_truth_folder):
        if not make_new_gt:
            pytest.skip("Ground truth folder is missing or empty, and make_new_gt is False. Skipping test.")

    if save_vcf and not os.path.exists(vcf_data_csv):  # alternatively, I can do this in vk clean by passing in vcf_data_csv=vcf_data_csv, cosmic_tsv=cosmic_tsv, cosmic_reference_genome_fasta=cosmic_reference_genome_fasta, variants="cosmic_cmc", sequences="cdna", cosmic_version=101
        cosmic_tsv = os.path.join(reference_folder_parent, "cosmic", "CancerMutationCensus_AllData_Tsv_v101_GRCh37", "CancerMutationCensus_AllData_v101_GRCh37.tsv")
        reference_genome_fasta = os.path.join(ensembl_grch37_release93_folder, "Homo_sapiens.GRCh37.dna.primary_assembly.fa")
        if not os.path.exists(reference_genome_fasta) or not os.path.exists(cosmic_tsv):
            pytest.skip("Skipping becasue save_vcf is True, vcf_data_csv doesn't exist, and either cosmic_tsv and/or reference_genome_fasta doesn't exist. Ensure cosmic_tsv and reference_genome_fasta exist.")
        vk.utils.add_vcf_info_to_cosmic_tsv(cosmic_tsv=cosmic_tsv, reference_genome_fasta=reference_genome_fasta, cosmic_df_out=vcf_data_csv, sequences="cdna", cosmic_version=101)

    if make_new_gt:
        os.makedirs(ground_truth_folder, exist_ok=True)

    kb_count_reference_genome_dir = out_dir / "kb_count_reference_genome_dir"

    vk.count(
        fastq_file,
        index=vcrs_index,
        t2g=vcrs_t2g,
        technology="bulk",
        parity="single",
        out=out_dir,
        kb_count_reference_genome_dir=kb_count_reference_genome_dir,
        k=51,
        threads=threads,
        quality_control_fastqs=quality_control_fastqs, cut_front=True, cut_tail=True,
        reference_genome_index=reference_genome_index, reference_genome_t2g=reference_genome_t2g,
        qc_against_gene_matrix=qc_against_gene_matrix,
        save_vcf=save_vcf,
        vcf_data_csv=vcf_data_csv
    )


    # file name, file type, columns to drop for comparison
    global columns_to_drop_info_filter  # should be unnecessary but got an error without it
    files_to_compare_and_file_type = [
        ("fastqs_quality_controlled/fastp_report.json", "json"),
        ("kb_count_out_vcrs/counts_unfiltered/adata.h5ad", "adata"),
        ("adata_cleaned.h5ad", "adata"),
        ("variants.vcf", "vcf"),
        ("vk_summarize/varseek_summarize_stats.txt", "txt"),
    ]

    for file, file_type in files_to_compare_and_file_type:
        test_path = os.path.join(out_dir, file)
        ground_truth_path = os.path.join(ground_truth_folder, file)
        if not os.path.isfile(test_path) and not os.path.isfile(ground_truth_path):
            continue
        if make_new_gt:
            os.makedirs(os.path.dirname(ground_truth_path), exist_ok=True)
            shutil.copy(test_path, ground_truth_path)
        apply_file_comparison(test_path, ground_truth_path, file_type)

def test_varseek_count_on_command_line(monkeypatch):
    import varseek.main as main_module

    test_args = [
        "vk", "count", 
        "-i", "shamindex.idx",
        "-g", "shamt2g.txt",
        "-x", "bulk",
        "--k", "11",
        "--adata_reference_genome", "ref_genome.fasta",
        "--kallisto", "shamkallisto",
        "--multiplexed",
        "myfastq.fastq"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setenv("TESTING", "true")  # so that main returns params_dict (there is a conditional in main to look for this environment variable)
    
    importlib.reload(main_module)
    fastqs, params_dict = main_module.main()  # now returns params_dict because TESTING is set

    expected_fastqs = ['myfastq.fastq']

    expected_dict = {
        'index': 'shamindex.idx',
        't2g': 'shamt2g.txt',
        'technology': 'bulk',
        'k': 11,
        'adata_reference_genome': 'ref_genome.fasta',
        'kallisto': 'shamkallisto',
        'multiplexed': True
    }
    
    assert fastqs == expected_fastqs
    assert params_dict == expected_dict