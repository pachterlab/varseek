import csv
import json
import os
import random
import shutil
import subprocess

import pandas as pd
import requests

random.seed(42)

# Define parameters
base_data_folder = "/Users/joeyrich/Documents/Caltech/Pachter/TCGA/data"
dtt_or_curl = "dtt"  # "dtt", "curl"
gdc_client = "/Users/joeyrich/Documents/Caltech/Pachter/TCGA/gdc-client"  # only if dtt_or_curl == "dtt"
controlled_access_token_file = "/Users/joeyrich/Documents/Caltech/Pachter/TCGA/gdc-user-token.2024-12-05T00_31_09.852Z.txt"
trial = True
number_of_samples_per_combination = 10
threads = 20

# 12 cancer types
primary_site_and_primary_diagnosis_tuple_list = [('bladder', 'transitional cell carcinoma'), ('breast', 'infiltrating duct carcinoma, nos'), ('bronchus and lung', 'squamous cell carcinoma, nos'), ('colon', 'adenocarcinoma, nos'), ('esophagus', 'adenocarcinoma, nos'), ('kidney', 'clear cell adenocarcinoma, nos'), ('liver and intrahepatic bile ducts', 'hepatocellular carcinoma, nos'), ('pancreas', 'infiltrating duct carcinoma, nos'), ('skin', 'malignant melanoma, nos'), ('stomach', 'adenocarcinoma, nos'), ('testis', 'seminoma, nos'), ('thyroid gland', 'papillary adenocarcinoma, nos')]  # primary sites without stages: ('brain', 'glioblastoma'), ('ovary', 'serous cystadenocarcinoma, nos'), ('prostate gland', 'adenocarcinoma, nos')

# 4 stages
ajcc_pathologic_stage_list = [['stage i', 'stage ia', 'stage ib'], ['stage ii', 'stage iia', 'stage iib', 'stage iic'], ['stage iii', 'stage iiia', 'stage iiib', 'stage iiic'], ['stage iv', 'stage iva', 'stage ivb', 'stage ivc']]

# tumor and normal
tissue_type_list = ['tumor', 'normal']
experimental_strategy_and_workflow_type_tuple_list = [('RNA-Seq', 'STAR 2-Pass Transcriptome')]  # can add/replace with ('WXS', 'BWA with Mark Duplicates and BQSR'), ('WGS', 'BWA with Mark Duplicates and BQSR')
data_category_and_data_format_tuple_list = [('sequencing reads', 'BAM')]  # can add/replace with ('simple nucleotide variation', 'VCF')  # can replace VCF with MAF or TSV (TSV only available for RNA-seq, not WGS/WXS)




def move_uuid_folders(base_data_folder, cohort_metadata_file):
    # Read the cohort metadata file
    df = pd.read_csv(cohort_metadata_file)

    # Iterate through rows in the metadata file
    for _, row in df.iterrows():
        uuid = row['uuid']
        output_folder = row['output_folder']
        
        # Define source and destination paths
        source = os.path.join(base_data_folder, uuid)
        destination = os.path.join(base_data_folder, output_folder)
        
        # Move the folder
        if os.path.exists(source):
            shutil.move(source, destination)
            print(f"Moved {source} to {destination}")
        else:
            print(f"Source folder does not exist: {source}")

if trial:
    primary_site_and_primary_diagnosis_tuple_list = primary_site_and_primary_diagnosis_tuple_list[:1]
    ajcc_pathologic_stage_list = [sum(ajcc_pathologic_stage_list, [])]
    tissue_type_list = tissue_type_list[:1]
    experimental_strategy_and_workflow_type_tuple_list = experimental_strategy_and_workflow_type_tuple_list[:1]
    data_category_and_data_format_tuple_list = data_category_and_data_format_tuple_list[:1]

# Define API endpoint
endpoint = "https://api.gdc.cancer.gov/files"

os.makedirs(base_data_folder, exist_ok=True)
cohort_metadata_file = os.path.join(base_data_folder, "cohort_metadata.tsv")

uuids = []
uuid_to_output_folder_dict = {}
failed_metadata_combinations = []
with open(cohort_metadata_file, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(["uuid", "file_name", "patient_id", "primary_site", "primary_diagnosis", "ajcc_pathologic_stage", "tissue_type", "experimental_strategy", "workflow_type", "data_category", "data_format", "output_folder"])

    print("Gathering metadata")
    for primary_site, primary_diagnosis in primary_site_and_primary_diagnosis_tuple_list:
        for ajcc_pathologic_stage in ajcc_pathologic_stage_list:
            for tissue_type in tissue_type_list:
                for experimental_strategy, workflow_type in experimental_strategy_and_workflow_type_tuple_list:
                    for data_category, data_format in data_category_and_data_format_tuple_list:
                        if isinstance(primary_site, str):
                            primary_site = [primary_site]
                        if isinstance(primary_diagnosis, str):
                            primary_diagnosis = [primary_diagnosis]
                        if isinstance(ajcc_pathologic_stage, str):
                            ajcc_pathologic_stage = [ajcc_pathologic_stage]
                        if isinstance(tissue_type, str):
                            tissue_type = [tissue_type]
                        if isinstance(experimental_strategy, str):
                            experimental_strategy = [experimental_strategy]
                        if isinstance(workflow_type, str):
                            workflow_type = [workflow_type]
                        if isinstance(data_category, str):
                            data_category = [data_category]
                        if isinstance(data_format, str):
                            data_format = [data_format]

                        primary_site_string = "_".join(primary_site)
                        primary_diagnosis_string = "_".join(primary_diagnosis)
                        primary_site_and_primary_diagnosis_string = primary_site_string + "_" + primary_diagnosis_string
                        ajcc_pathologic_stage_string = "_".join(ajcc_pathologic_stage)
                        tissue_type_string = "_".join(tissue_type)
                        experimental_strategy_string = "_".join(experimental_strategy)
                        workflow_type_string = "_".join(workflow_type)
                        experimental_strategy_and_workflow_type_string = experimental_strategy_string + "_" + workflow_type_string
                        data_category_string = "_".join(data_category)
                        data_format_string = "_".join(data_format)
                        
                        # Define filters
                        filters = {
                            "op": "and",
                            "content": [
                                {"op": "in", "content": {"field": "cases.project.program.name", "value": ["TCGA"]}},
                                {"op": "in", "content": {"field": "experimental_strategy", "value": experimental_strategy}},  # RNA-Seq, WXS, WGS  # see cohort builder -> available data
                                {"op": "in", "content": {"field": "analysis.workflow_type", "value": workflow_type}},  # see cohort builder -> available data
                                {"op": "in", "content": {"field": "data_category", "value": data_category}},  # sequencing reads, simple nucleotide variation  # see cohort builder -> available data
                                {"op": "in", "content": {"field": "data_format", "value": data_format}},  # BAM, VCF, MAF, TSV  # see cohort builder -> available data
                                {"op": "in", "content": {"field": "cases.primary_site", "value": primary_site}},  # see cohort builder -> general -> primary site
                                {"op": "in", "content": {"field": "cases.diagnoses.primary_diagnosis", "value": primary_diagnosis}},  # see cohort builder -> general -> primary diagnosis
                                # {"op": "in", "content": {"field": "cases.diagnoses.tissue_or_organ_of_origin", "value": tissue_or_organ_of_origin}},  # see cohort builder -> general -> tissue or organ of origin
                                # {"op": "in", "content": {"field": "cases.diagnoses.tumor_grade", "value": tumor_grade}},  # see cohort builder -> general diagnosis -> tumor grade
                                {"op": "in", "content": {"field": "cases.diagnoses.ajcc_pathologic_stage", "value": ajcc_pathologic_stage}},  # see cohort builder -> general diagnosis -> Ajcc Pathologic Stage
                                {"op": "in", "content": {"field": "cases.samples.tissue_type", "value": tissue_type}},  # tumor, normal  # see cohort builder -> biospecimen -> tissue type
                                # {"op": "in", "content": {"field": "cases.demographic.age_at_index", "value": age}},  # see cohort builder -> demographic -> age
                                # {"op": "in", "content": {"field": "cases.demographic.gender", "value": gender}},  # see cohort builder -> demographic -> gender
                                # {"op": "in", "content": {"field": "cases.demographic.ethnicity", "value": ethnicity}},  # see cohort builder -> demographic -> ethnicity
                                # {"op": "in", "content": {"field": "cases.demographic.race", "value": race}},  # see cohort builder -> demographic -> race
                            ],
                        }

                        # Define parameters
                        params = {
                            "filters": json.dumps(filters),
                            "fields": "file_id,file_name,cases.submitter_id",
                            "format": "JSON",
                            "size": "100",  # Number of results per page
                        }

                        # Make request
                        response = requests.get(endpoint, params=params)

                        # Check response status
                        if response.status_code == 200:
                            results = response.json()["data"]["hits"]
                            # Pick number_of_samples_per_combination random items
                            results_random_selection = random.sample(results, min(len(results), number_of_samples_per_combination))

                            for result in results_random_selection:
                                uuid = result['file_id']
                                file_name = result['file_name']
                                patient_id = result['cases'][0]['submitter_id']
                                uuids.append(uuid)

                                output_folder = os.path.join(base_data_folder, primary_site_and_primary_diagnosis_string, ajcc_pathologic_stage_string, tissue_type_string, experimental_strategy_and_workflow_type_string, uuid)
                                uuid_to_output_folder_dict[uuid] = output_folder
                                os.makedirs(output_folder, exist_ok=True)

                                row = [uuid, file_name, patient_id, primary_site_string, primary_diagnosis_string, ajcc_pathologic_stage_string, tissue_type_string, experimental_strategy_string, workflow_type_string, data_category_string, data_format_string, output_folder]
                                writer.writerow(row)  # Write the row
                        else:
                            print(f"Error gathering metadata for {output_folder}: {response.status_code}, {response.text}")
                            failed_metadata_combinations.append(output_folder)

if dtt_or_curl == "dtt":
    uuids_string = " ".join(uuids)
    print(f"Downloading data to {base_data_folder}")
    dtt_command = f"{gdc_client} download -n {threads} -d {base_data_folder} -t {controlled_access_token_file} {uuids_string}"  # I can replace {uuids_string} with -m MANIFEST_FILE_PATH.txt
    subprocess.run(dtt_command, shell=True)

    print("Moving files to appropriate folders")
    move_uuid_folders(base_data_folder = base_data_folder, cohort_metadata_file = cohort_metadata_file)
elif dtt_or_curl == "curl":
    with open(controlled_access_token_file, 'r') as file:
        token = file.read().strip()

    for uuid in uuids:
        print(f"Downloading {uuid} to {output_folder}")
        output_folder = uuid_to_output_folder_dict[uuid]
        curl_command = f"curl --output-dir '{output_folder}' --remote-name --remote-header-name --header 'X-Auth-Token: {token}' 'https://api.gdc.cancer.gov/data/{uuid}'"
        subprocess.run(curl_command, shell=True)

print("Done!")
print(f"Failed metadata combinations: {failed_metadata_combinations}")
