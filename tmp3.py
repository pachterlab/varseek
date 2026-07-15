from tqdm import tqdm


def _parse_fasta(path):
    """Yield (header, sequence) tuples from a FASTA file."""
    header, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header, seq = line[1:], []
            else:
                seq.append(line)
    if header is not None:
        yield header, "".join(seq)


def kmer_match(ref_fa, read_fa, k=31):
    """Return the positions (0-based indices) of reads in ``read_fa`` that
    contain at least one k-mer found anywhere in ``ref_fa``.

    Semantics: a read matches if any of its k-mers is present anywhere in the
    reference. To stay within memory for very large references (e.g. a whole
    genome), we index the (small) reads file into a ``kmer -> [read indices]``
    map and then *stream* the (large) reference file, marking a read as matched
    as soon as one of the reference's k-mers hits one of its k-mers. This is
    equivalent to building a k-mer set over the reference and scanning the
    reads, but uses memory proportional to the reads rather than the reference.
    """
    # Build kmer -> list of read indices from the (small) reads file.
    read_index = {}
    num_reads = 0
    for pos, (_, seq) in enumerate(tqdm(_parse_fasta(read_fa), desc="Indexing reads")):
        num_reads = pos + 1
        seen = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in seen:
                continue
            seen.add(kmer)
            read_index.setdefault(kmer, []).append(pos)

    # Stream the (large) reference, marking reads that share any k-mer.
    matched = bytearray(num_reads)
    remaining = num_reads
    get = read_index.get
    for _, seq in tqdm(_parse_fasta(ref_fa), desc="Streaming reference"):
        n = len(seq) - k + 1
        for i in range(n):
            hit = get(seq[i:i + k])
            if hit is not None:
                for idx in hit:
                    if not matched[idx]:
                        matched[idx] = 1
                        remaining -= 1
        if remaining == 0:
            break

    return [i for i in range(num_reads) if matched[i]]


if __name__ == "__main__":
    read_fasta = "/home/jrich/Desktop/varseek-examples/data/cosmic_vs_denovo/vk_ref_cosmic/vcrs.fa"
    ref_fastas = ["/home/jrich/Desktop/varseek-examples/data/kb_quant_tests/kb_ref_out_cdna/cdna.fasta", "/home/jrich/data/reference/t2t_CHM13v2/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"]
    for ref_fasta in ref_fastas:
        matched_positions = kmer_match(ref_fasta, read_fasta, k=41)
        print(f"Number of reads in {read_fasta} that match at least one k-mer in {ref_fasta}: {len(matched_positions)}", flush=True)
