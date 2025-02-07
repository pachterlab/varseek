# !pip install pyliftover

import re

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
from pyliftover import LiftOver

lo = LiftOver('hg38', 'hg19')
# old_chrom = 'chr12'
# old_position = 25245350
# new_chrom, new_pos, strand, _ = lo.convert_coordinate(old_chrom, old_position)[0]  # expect 25398284
# print(f'Converted position: {old_chrom}:{old_position} -> {new_chrom}:{new_pos} (strand: {strand})')

def gdc_preprocessing_genome_positions(gdc_df):
    # Function to extract chromosome number
    def extract_chromosome(dna_change):
        return re.search(r'^chr(\w+):', dna_change).group(1)

    # Function to extract start position
    def extract_start_position(dna_change):
        return int(re.search(r'g\.(\d+)', dna_change).group(1))

    # Function to extract end position
    def extract_end_position(row):
        dna_change = row['dna_change']
        start_position = row['GRCh38_GENOME_START']
        if '>' in dna_change:  # Substitution
            return start_position
        elif 'ins' in dna_change:  # Insertion or complex mutation
            return int(re.search(r'_(\d+)', dna_change).group(1))
        elif 'del' in dna_change:  # Deletion
            del_match = re.search(r'del([A-Z]+)', dna_change)
            if del_match:
                length = len(del_match.group(1))
                return start_position + length - 1
            else:  # Complex deletion-insertion
                return int(re.search(r'_(\d+)', dna_change).group(1))
        return start_position

    # Apply the functions to create new columns
    gdc_df['GRCh38_CHROMOSOME'] = gdc_df['dna_change'].progress_apply(extract_chromosome)
    gdc_df['GRCh38_GENOME_START'] = gdc_df['dna_change'].progress_apply(extract_start_position)
    gdc_df['GRCh38_GENOME_END'] = gdc_df.progress_apply(extract_end_position, axis=1)

    return gdc_df

def parse_grch37_liftover(entry):
    match = re.match(r'chr(\w+):(\d+)-(\d+)', entry)
    if match:
        return match.groups()
    return None, None, None

def convert_gdc_to_37(df, output_path = None):

    df = gdc_preprocessing_genome_positions(df)

    # df.rename(columns={'dna_change': 'GRCh38_dna_change'}, inplace=True)

    df['liftover_format'] = 'chr' + df['GRCh38_CHROMOSOME'] + ':' + df['GRCh38_GENOME_START'].astype(str) + '-' + df['GRCh38_GENOME_END'].astype(str)


    mutation_list = df['liftover_format'].tolist()
    pattern = r'chr(\d+|X|Y|M|MT):(\d+)-(\d+)'

    good_output_list = []
    bad_output_list = []

    for line in mutation_list:
        if line.startswith('#'):
            continue
        match = re.match(pattern, line)
        if match:
            chrom = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
        try:
            new_chrom_start, new_start, _, _ = lo.convert_coordinate(f"chr{chrom}", start)[0]
            new_chrom_end, new_end, _, _ = lo.convert_coordinate(f"chr{chrom}", end)[0]
            assert new_chrom_start == new_chrom_end
            good_output_list.append(f"{new_chrom_start}:{new_start}-{new_end}")
        except Exception as e:
            bad_output_list.append(line)


    df_no_failures = df[~df['liftover_format'].isin(bad_output_list)].reset_index(drop=True)
    df_no_failures['GRCH37_liftover'] = good_output_list

    df_no_failures[['GRCh37_chromosome', 'GRCh37_GENOME_START', 'GRCh37_GENOME_END']] = df_no_failures['GRCH37_liftover'].progress_apply(
        lambda x: pd.Series(parse_grch37_liftover(x))
    )

    if output_path:
        df_no_failures.to_csv(output_path, sep='\t', index=False)
        print(f"Updated DataFrame saved to {output_path}")

if __name__ == "__main__":
    file_path = '/home/jrich/Desktop/CART_prostate_sc/data/reference/GDC/frequent-mutations.2024-05-15-3.tsv'
    df = pd.read_csv(file_path, sep='\t')
    
    convert_gdc_to_37(df, output_path = '/home/jrich/Desktop/CART_prostate_sc/data/reference/GDC/frequent-mutations_with_GRCh37.2024-05-15-3.tsv')