import csv
import json
import os

import requests

experimental_strategy = "RNA-Seq"  # RNA-Seq, WGS, WXS
base_data_folder = "/home/jrich/data/varseek_data/sequencing/bulk/tcga"


matched_tumor_normal_metadata_file = os.path.join(base_data_folder, f"matched_tumor_normal_metadata_{experimental_strategy}.tsv")

if experimental_strategy == "RNA-Seq":
    workflow_type = "STAR 2-Pass Transcriptome"
else:
    workflow_type = "BWA with Mark Duplicates and BQSR"

# Define the API endpoint
endpoint = "https://api.gdc.cancer.gov/files"

data_category = "Sequencing Reads"
data_format = "BAM"

# Define filters
filters = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.program.name", "value": ["TCGA"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": [experimental_strategy]}},
        {"op": "in", "content": {"field": "analysis.workflow_type", "value": [workflow_type]}},
        {"op": "in", "content": {"field": "data_category", "value": [data_category]}},
        {"op": "in", "content": {"field": "data_format", "value": [data_format]}},
        {"op": "in", "content": {"field": "cases.samples.tissue_type", "value": ["tumor", "normal"]}},
    ],
}

# Define the fields to retrieve
fields_list = [
    "file_id",
    "file_name",
    "cases.submitter_id",
    "cases.primary_site",
    "cases.diagnoses.primary_diagnosis",
    "cases.samples.tissue_type",
    "cases.diagnoses.ajcc_pathologic_stage"
]
fields = ",".join(fields_list)

# Request parameters
params = {
    "filters": json.dumps(filters),
    "fields": fields,
    "format": "JSON",
    "size": 10000  # Increase size to retrieve all results
}

# Make the request
response = requests.get(endpoint, params=params)

# Process the response
if response.status_code == 200:
    data = response.json()["data"]["hits"]
    
    # Group by patient ID
    patient_map = {}
    for record in data:
        # Extract fields
        patient_id = record["cases"][0]["submitter_id"].lower()
        tissue_type = record["cases"][0]["samples"][0]["tissue_type"].lower()
        file_id = record["file_id"]
        file_name = record["file_name"]
        primary_site = record["cases"][0].get("primary_site", None)
        primary_diagnosis = record["cases"][0].get("diagnoses", [{}])[0].get("primary_diagnosis", None)
        ajcc_pathologic_stage = record["cases"][0].get("diagnoses", [{}])[0].get("ajcc_pathologic_stage", None)
        
        # Initialize patient entry if not already present
        if patient_id not in patient_map:
            patient_map[patient_id] = {
                "uuid_tumor": None,
                "file_name_tumor": None,
                "uuid_normal": None,
                "file_name_normal": None,
                "primary_site": primary_site,
                "primary_diagnosis": primary_diagnosis,
                "ajcc_pathologic_stage": ajcc_pathologic_stage
            }
        
        # Assign tumor and normal values
        if tissue_type == "tumor":
            patient_map[patient_id]["uuid_tumor"] = file_id
            patient_map[patient_id]["file_name_tumor"] = file_name
        elif tissue_type == "normal":
            patient_map[patient_id]["uuid_normal"] = file_id
            patient_map[patient_id]["file_name_normal"] = file_name
    
    # Filter patients with both tumor and normal
    patients_with_both = {
        patient_id: data
        for patient_id, data in patient_map.items()
        if data["uuid_tumor"] and data["uuid_normal"]
    }

    with open(matched_tumor_normal_metadata_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["patient_id", "uuid_tumor", "file_name_tumor", "uuid_normal", "file_name_normal", "primary_site", "primary_diagnosis", "ajcc_pathologic_stage", "experimental_strategy", "workflow_type", "data_category", "data_format"])

        for patient_id, data in patients_with_both.items():
            row = [patient_id, data["uuid_tumor"], data["file_name_tumor"], data["uuid_normal"], data["file_name_normal"], data["primary_site"], data["primary_diagnosis"], data["ajcc_pathologic_stage"], experimental_strategy, workflow_type, data_category, data_format]
            writer.writerow(row)  # Write the row
        
else:
    print(f"Error: {response.status_code}, {response.text}")


# RESULTS OF RNA-SEQ:  # in my notebook, I also group by stage for an even better understanding
# [('bladder', 'transitional cell carcinoma'),  # 18 tumor-normal matches
#  ('breast', 'infiltrating duct carcinoma, nos'), # 90 tumor-normal matches
#  ('bronchus and lung', 'squamous cell carcinoma, nos'),  # 46 tumor-normal matches
#  ('colon', 'adenocarcinoma, nos'),  # 24 tumor-normal matches
#  ('esophagus', 'adenocarcinoma, nos'),  # 9 tumor-normal matches
#  ('kidney', 'clear cell adenocarcinoma, nos'), # 72 tumor-normal matches
#  ('liver and intrahepatic bile ducts', 'hepatocellular carcinoma, nos'),  # 49 tumor-normal matches
#  ('pancreas', 'infiltrating duct carcinoma, nos'),  # 2 tumor-normal matches
#  ('skin', 'malignant melanoma, nos'), 
#  ('stomach', 'adenocarcinoma, nos'),  # 5 tumor-normal matches
#  ('testis', 'seminoma, nos'), 
#  ('thyroid gland', 'papillary adenocarcinoma, nos')  # 49 tumor-normal matches
# ]

# prostate, adenocarcinoma, nos: 51 tumor-normal matches
# bronchus and lung, adenocarcinoma, nos: 42 tumor-normal matches