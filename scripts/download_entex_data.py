import argparse
import os

import pandas as pd

from varseek.utils import download_entex_fastq_links, make_entex_df


def main(args):
    
    data_download_base = args.data_download_base

    if not args.tissue:
        tissue = None
    else:
        tissue = args.tissue
    
    entex_df_path = f"{data_download_base}/entex_df.csv"
    if os.path.exists(entex_df_path):
        entex_df = pd.read_csv(entex_df_path)
    else:
        entex_df = make_entex_df()
        entex_df.to_csv(entex_df_path, index=False)

    if args.show_tissue_names_only:
        print("Valid tissue names\n", sorted(set(entex_df['tissue']), key=str.lower))
        return
    
    download_entex_fastq_links(entex_df, tissue=tissue, data_download_base=data_download_base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA-seq processing script with configurable parameters.")
    
    # Define arguments with argparse
    parser.add_argument("--tissue", type=str, required=False, help="tissue to select. default all")
    parser.add_argument("--data_download_base", type=str, required=False, default = ".", help="parent path for downloads. default '.'")
    parser.add_argument("--show_tissue_names_only", action="store_true", help="Simply shows valid tissue names and exits.")

    # Parse arguments
    args = parser.parse_args()

    # Run main processing with parsed arguments
    main(args)
