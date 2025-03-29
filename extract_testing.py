import os
import shutil
import sys

varseek_directory = "/Users/joeyrich/Desktop/local/varseek"

conda_env_path = os.path.dirname(os.path.dirname(shutil.which("kb")))  # to get kb path
operating_system = "linux" if sys.platform.startswith("linux") else "darwin/m1"

kallisto = f"{conda_env_path}/lib/python3.10/site-packages/kb_python/bins/{operating_system}/kallisto/kallisto"  # or kallisto_k64
bustools = f"{conda_env_path}/lib/python3.10/site-packages/kb_python/bins/{operating_system}/bustools/bustools"

ref_fa = f"{varseek_directory}/tests/kb_files/single_cell_tests/ref_sc_test.fa"
read1_fq = f"{varseek_directory}/tests/kb_files/single_cell_tests/reads_R1.fq"
read2_fq = f"{varseek_directory}/tests/kb_files/single_cell_tests/reads_R2.fq"
test_index = f"{varseek_directory}/tests/kb_files/single_cell_tests/index_test.idx"
test_t2g = f"{varseek_directory}/tests/kb_files/single_cell_tests/t2g_test.txt"
kb_count_out_test = f"{varseek_directory}/tests/kb_files/single_cell_tests/test_kb_count_out_hamming1_mm_and_union_bulk2"
kb_extract_out_test = f"{varseek_directory}/tests/kb_files/single_cell_tests/test_kb_count_extract_alternative11_mm_and_union_bulk"
fastq_file_list = [read2_fq]

fastq_file_list = fastq_file_list
kb_count_out_dir = kb_count_out_test
t2g_file = test_t2g
technology = "BULK"
kb_extract_out_dir=kb_extract_out_test

from varseek.utils.varseek_clean_utils import make_bus_df, kb_extract_all_alternative
kb_extract_all_alternative(fastq_file_list=fastq_file_list, kb_count_out_dir=kb_count_out_dir, index_file=test_index, t2g_file=t2g_file, technology=technology, kb_extract_out_dir=kb_extract_out_dir, kallisto=kallisto, bustools=bustools, gzip_output=False, mm=True, union=True, overwrite=False)