#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path
import argparse
from tqdm import tqdm


DNA = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def fasta_reader(filename):
    """
    Simple FASTA parser.

    Yields
    ------
    header : str
    sequence : str
    """
    header = None
    seq = []

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)

                header = line[1:].split()[0]
                seq = []

            else:
                seq.append(line.upper())

        if header is not None:
            yield header, "".join(seq)


def build_kmer_map(fastas, k=31):
    """
    Returns

        dict[int, list[(reference, position)]]

    where the key is the packed 2-bit representation of the k-mer.
    """

    mask = (1 << (2 * k)) - 1

    table = defaultdict(list)

    for fasta in fastas:

        for ref, seq in tqdm(fasta_reader(fasta), desc="Building k-mer map", unit="sequence", total=sum(1 for _ in fasta_reader(fasta))):

            rolling = 0
            valid = 0

            for i, base in enumerate(seq):

                if base not in DNA:
                    rolling = 0
                    valid = 0
                    continue

                rolling = ((rolling << 2) | DNA[base]) & mask
                valid += 1

                if valid >= k:
                    start = i - k + 1
                    table[rolling].append((ref, start))

    return table


def decode_kmer(value, k):
    """Convert packed integer back into DNA sequence."""

    alphabet = "ACGT"

    chars = [""] * k

    for i in range(k - 1, -1, -1):
        chars[i] = alphabet[value & 0b11]
        value >>= 2

    return "".join(chars)


def write_table(table, outfile, k):

    with open(outfile, "w") as out:

        out.write("kmer\treference\tposition\n")

        for packed, locations in table.items():

            kmer = decode_kmer(packed, k)

            for ref, pos in locations:
                out.write(f"{kmer}\t{ref}\t{pos}\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "fasta",
        nargs="+",
        help="One or more FASTA files.",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=31,
    )

    parser.add_argument(
        "-o",
        "--out",
        required=True,
    )

    args = parser.parse_args()

    table = build_kmer_map(args.fasta, args.k)

    write_table(table, args.out, args.k)


if __name__ == "__main__":
    main()