# import varseek as vk
import os
import pandas as pd
from varseek.utils import report_time_and_memory_of_script

number_of_variants_list = [1, 4, 16, 64, 256, 1024, 4096]  # number of variants, in thousands
cosmic_mutations_path = "/Users/joeyrich/Desktop/local/data/cosmic/CancerMutationCensus_AllData_Tsv_v100_GRCh37/CancerMutationCensus_AllData_v100_GRCh37_mutation_workflow.csv"
sequences_fasta_path = "/Users/joeyrich/Desktop/local/data/ensembl/grch37_release93/Homo_sapiens.GRCh37.cds.all.fa"
output_file = "/Users/joeyrich/Desktop/local/varseek/logs/vk_ref_time_and_memory.txt"

# only if cosmic_mutations_path does not exist
cosmic_release = "100"
grch = "37"
reference_out = "cosmic"
os.environ['COSMIC_EMAIL'] = 'your_email'  # replace with your email


random_seed = 42
script_dir = os.path.dirname(os.path.abspath(__file__))  # make sure I have the right script directory
vk_ref_script_path = os.path.join(script_dir, "run_varseek_ref_for_benchmarking_time_and_memory.py")


# download reference genome
if not os.path.isfile(sequences_fasta_path):
    import subprocess
    os.makedirs(os.path.dirname(sequences_fasta_path), exist_ok=True)
    cds_file = "Homo_sapiens.GRCh37.cds.all.fa"
    
    if grch == "37":
        gget_ref_grch = "human_grch37"
    elif grch == "38":
        gget_ref_grch = "human"
    else:
        gget_ref_grch = grch

    ref_sequence_download_command = f"gget ref -w cds -r 93 --out_dir {reference_out} -d {gget_ref_grch}"

    sequences_download_command_list = ref_sequence_download_command.split(" ")

    cds_out_temp = f"{reference_out}/{cds_file}.gz"

    print(f"Downloading reference sequences with {' '.join(sequences_download_command_list)}. Note that this requires curl >=7.73.0")
    subprocess.run(sequences_download_command_list, check=True)
    os.rename(cds_out_temp, f"{sequences_fasta_path}.gz")
    subprocess.run(["gunzip", f"{sequences_fasta_path}.gz"], check=True)

# download COSMIC
if not os.path.isfile(cosmic_mutations_path):
    import gget
    cosmic_email = os.environ.get('COSMIC_EMAIL')
    cosmic_password = os.environ.get('COSMIC_PASSWORD')

    if not cosmic_email or not cosmic_password:
        raise ValueError("Please provide COSMIC email and password via the environment variables COSMIC_EMAIL and COSMIC_PASSWORD, respectively; or please download the COSMIC CMC database manually with gget_cosmic using the download_cosmic=True flag.")

    os.makedirs(os.path.dirname(cosmic_mutations_path), exist_ok=True)
    mutations = f"{reference_out}/CancerMutationCensus_AllData_Tsv_v{cosmic_release}_GRCh{grch}/CancerMutationCensus_AllData_v{cosmic_release}_GRCh{grch}_mutation_workflow.csv"
    
    gget.cosmic(
        None,
        grch_version=grch,
        cosmic_version=cosmic_release,
        out=reference_out,
        mutation_class="cancer",
        download_cosmic=True,
        keep_genome_info=True,
        remove_duplicates=True,
        email=cosmic_email,
        password=cosmic_password,
    )

    # move mutations cosmic_mutations_path
    os.rename(mutations, cosmic_mutations_path)

cosmic_df = pd.read_csv(cosmic_mutations_path)

for number_of_variants in number_of_variants_list:
    number_of_variants *= 1000
    cosmic_df_subsampled = cosmic_df.sample(n=number_of_variants, random_state=random_seed)
    argparse_flags = f"--mutations {cosmic_df_subsampled} --sequences {sequences_fasta_path}"  # make sure to provide all keyword args with two-dashes (ie full argument name)
    report_time_and_memory_of_script(vk_ref_script_path, output_file = output_file, argparse_flags = argparse_flags)
