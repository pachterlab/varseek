import json
import subprocess

import pandas as pd
import requests


def download_msk_data(output_path = "data_mutation_downloaded.txt"):
    # URL of the file to be downloaded
    url = "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/msk_impact_2017/data_mutations.txt"

    # Send a GET request to the URL
    response = requests.get(url)

    lines = response.content.decode().splitlines(keepends=True)

    # Extract relevant information from the file
    version = lines[0].strip().split(' ')[1]
    oid = lines[1].strip().split(':')[1].strip()
    size = int(lines[2].strip().split(' ')[1])

    # Create the JSON object
    lfs_metadata = {
        "operation": "download",
        "transfer": ["basic"], 
        "objects": [
            {"oid": oid, "size": size}
        ]
    }

    # Convert the dictionary to a JSON string
    lfs_metadata_json = json.dumps(lfs_metadata)
    lfs_metadata_json

    github_url = f"https://github.com/cBioPortal/datahub.git/info/lfs/objects/batch"

    curl_command = [
        "curl",
        "-X", "POST",
        "-H", "Accept: application/vnd.git-lfs+json",
        "-H", "Content-type: application/json",
        "-d", lfs_metadata_json,
        github_url
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)
    response_json = json.loads(result.stdout)

    href = response_json["objects"][0]["actions"]["download"]["href"]

    response = requests.get(href)
    filename = 'data_sv_downloaded.txt'

    # Save the file content
    with open(filename, 'wb') as file:
        file.write(response.content)

    print(f"File downloaded and saved as {filename}")

def convert_to_csv(input_path):
    df = pd.read_csv(input_path, sep='\t')

    output_path = input_path.replace('.txt', '.csv')

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    text_file_path = "data_mutation_downloaded.txt"
    download_msk_data(output_path = text_file_path)
    convert_to_csv(input_path = text_file_path)