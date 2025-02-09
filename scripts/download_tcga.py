import os
import shutil
import subprocess

import pandas as pd

# Define parameters
base_data_folder = "/Users/joeyrich/Documents/Caltech/Pachter/TCGA/data"
cohort_metadata_file = "PATH_TO_METADATA.tsv"
dtt_or_curl = "dtt"  # "dtt", "curl"
gdc_client = "/Users/joeyrich/Documents/Caltech/Pachter/TCGA/gdc-client"  # only if dtt_or_curl == "dtt"
controlled_access_token_file = "/Users/joeyrich/Documents/Caltech/Pachter/TCGA/gdc-user-token.2024-12-05T00_31_09.852Z.txt"
threads = 20
uuid_column = 'uuid'  # uuid for mixed, uuit_tumor or uuid_normal for matched


def move_uuid_folders(base_data_folder, cohort_metadata_file, uuid_column = 'uuid', output_folder_column = 'output_folder'):
    # Read the cohort metadata file
    df = pd.read_csv(cohort_metadata_file)

    # Iterate through rows in the metadata file
    for _, row in df.iterrows():
        uuid = row[uuid_column]
        output_folder = row[output_folder_column]
        
        # Define source and destination paths
        source = os.path.join(base_data_folder, uuid)
        destination = os.path.join(base_data_folder, output_folder)
        
        # Move the folder
        if os.path.exists(source):
            shutil.move(source, destination)
            print(f"Moved {source} to {destination}")
        else:
            print(f"Source folder does not exist: {source}")

df = pd.read_csv(cohort_metadata_file, sep='\t')

# Extract columns as lists
uuids = df[uuid_column].tolist()

if dtt_or_curl == "dtt":
    uuids_string = " ".join(uuids)
    print(f"Downloading data to {base_data_folder}")
    dtt_command = f"{gdc_client} download -n {threads} -d {base_data_folder} -t {controlled_access_token_file} {uuids_string}"  # I can replace {uuids_string} with -m MANIFEST_FILE_PATH.txt
    subprocess.run(dtt_command, shell=True)

    # print("Moving files to appropriate folders")
    # move_uuid_folders(base_data_folder = base_data_folder, cohort_metadata_file = cohort_metadata_file)
elif dtt_or_curl == "curl":
    with open(controlled_access_token_file, 'r', encoding="utf-8") as file:
        token = file.read().strip()

    for uuid in uuids:
        output_folder = os.path.join(base_data_folder, uuid) # uuid_to_output_folder_dict[uuid]
        print(f"Downloading {uuid} to {output_folder}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            curl_command = f"curl --output-dir '{output_folder}' --remote-name --remote-header-name --header 'X-Auth-Token: {token}' 'https://api.gdc.cancer.gov/data/{uuid}'"
            subprocess.run(curl_command, shell=True)

print("Done!")