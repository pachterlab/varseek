import json

cancer_types_to_keep = ["NSCLC", "melanoma", "colorectal_adenocarcinoma", "ovary_adenocarcinoma", "Ductal Adenocarcinoma", "exocrine", "gastric_adenocarcinoma", "upper_aerodigestive_squamous", "hepatocellular_carcinoma", "bladder_carcinoma", "renal_cell_carcinoma"]
# cancer_types_to_keep = ["LUNG", "SKIN", "LARGE_INTESTINE", "OVARY", "PANCREAS", "STOMACH", "UPPER_AERODIGESTIVE_TRACT", "HAEMATOPOIETIC_AND_LYMPHOID_TISSUE", "LIVER", "KIDNEY"]  # tissues to keep

number_to_keep = 5  # Maximum number of records to keep per strategy

json_path = "/home/jrich/data/varseek_data/sequencing/bulk/ccle/ccle_metadata_updated.json"
output_json_path = "/home/jrich/data/varseek_data/sequencing/bulk/ccle/ccle_metadata_medium_best.json"

with open(json_path, 'r') as file:
    data = json.load(file)


filtered_data = []

# Filter records by library strategies
for cancer_type in cancer_types_to_keep:
    # Get records matching the current strategy
    strategy_records = [
        study for study in data 
        if study['library_strategy'] == 'RNA-Seq' and 
        'subtype_disease' in study and 
        'lineage_subtype' in study and 
        (study['subtype_disease'] == cancer_type or study['lineage_subtype'] == cancer_type)
    ]
    
    # Limit the number of records to `number_to_keep` or however many are available
    filtered_data.extend(strategy_records[:number_to_keep])

# Write the filtered results to a new JSON file
with open(output_json_path, 'w') as file:
    json.dump(filtered_data, file, indent=4)

print(f"Filtered data written to {output_json_path}")