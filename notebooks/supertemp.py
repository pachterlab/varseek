import os
import subprocess

def check_for_successful_downloads(ccle_data_out_base, save_fastq_files = False):
    bad_samples = []

    for subfolder in os.listdir(ccle_data_out_base):
        print(f"Checking {subfolder}")
        subfolder_path = os.path.join(ccle_data_out_base, subfolder)
        
        if os.path.isdir(subfolder_path) and subfolder != "multiqc_total_data":
            sample_name = subfolder.split("___")[-1]
            
            # Paths to the required subdirectories
            mutation_index_path = os.path.join(subfolder_path, "kb_count_out_mutation_index")
            standard_index_path = os.path.join(subfolder_path, "kb_count_out_standard_index")
            
            # Check if both subdirectories exist and are non-empty
            mutation_exists = os.path.exists(mutation_index_path) and os.listdir(mutation_index_path)
            standard_exists = os.path.exists(standard_index_path) and os.listdir(standard_index_path)

            if not mutation_exists and not standard_exists:
                bad_samples.append(subfolder)
                continue

            if save_fastq_files:
                fastqs_to_check = [f"{sample_name}_1", f"{sample_name}_2", "combined"]

                for fastq in fastqs_to_check:
                    gzip_check_command = f"gzip -t {subfolder_path}/{fastq}.fastq.gz"
                    try:
                        subprocess.run(gzip_check_command, shell=True, check=True)
                    except subprocess.CalledProcessError:
                        bad_samples.append(subfolder)
                        break

    print(f"Samples with failed downloads: {bad_samples}")

    return bad_samples


_ = check_for_successful_downloads("/home/jrich/data/varseek_data/sequencing/bulk/ccle/ccle_data_out_medium_nov18", save_fastq_files=True)