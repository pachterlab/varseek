import os
import subprocess
import tempfile
import time

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


def _read_name_to_positions(read_fa):
    """Map each read's SAM QNAME (header up to first whitespace) to the list of
    0-based positions at which it appears in ``read_fa``.

    A list is used because two reads could in principle share a truncated header.
    """
    name_to_positions = {}
    for pos, (header, _) in enumerate(_parse_fasta(read_fa)):
        name = header.split()[0] if header else header
        name_to_positions.setdefault(name, []).append(pos)
    return name_to_positions


def kmer_match_bowtie2(
    ref_fa,
    read_fa,
    k=31,
    threads=2,
    bowtie2="bowtie2",
    bowtie2_build="bowtie2-build",
    strandedness=False,
    N_penalty=1,
    max_ambiguous=0,
    work_dir=None,
):
    """Return the positions (0-based indices) of reads in ``read_fa`` that
    contain at least one k-mer that aligns exactly to ``ref_fa``, using bowtie2.

    This mirrors the IO of ``tmp3.py``'s ``kmer_match`` (which uses an in-memory
    Python k-mer set), but instead builds a bowtie2 index over ``ref_fa`` and
    extracts every k-mer of every read (``-F k,1``) to align against it. A read
    matches if any of its k-mers aligns with an exact, full-length (score 0)
    hit. Modeled on ``align_to_normal_genome_and_build_dlist`` in
    ``varseek/utils/varseek_info_utils.py``.
    """
    # Set up a working directory for the index and SAM output.
    cleanup = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="kmer_match_bowtie2_")
        cleanup = True
    else:
        os.makedirs(work_dir, exist_ok=True)

    index_dir = os.path.join(work_dir, "bowtie_index")
    index_prefix = os.path.join(index_dir, "ref")
    output_sam_file = os.path.join(work_dir, "alignment.sam")

    runtimes = {}

    try:
        # --- Build the bowtie2 index over the reference FASTA. ---
        if not os.path.exists(index_dir) or not os.listdir(index_dir):
            os.makedirs(index_dir, exist_ok=True)
            print("Building bowtie2 index over reference")
            index_start = time.perf_counter()
            subprocess.run(
                [
                    bowtie2_build,
                    "--threads",
                    str(threads),
                    ref_fa,
                    index_prefix,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            runtimes["index_build_seconds"] = time.perf_counter() - index_start
            print(f"Index build runtime: {runtimes['index_build_seconds']:.2f} s")

        # --- Align every k-mer of every read to the reference index. ---
        # -F k,1 extracts each length-k substring (stride 1) of every read and
        # aligns it as its own query, named "{read_header}_{offset}".
        # --score-min C,0,0 with -N 0 / --no-1mm-upfront forces exact matches.
        print("Running bowtie2 alignment")
        bowtie2_alignment_command = [
            bowtie2,
            "-a",  # report all alignments
            "-f",  # input reads are FASTA
            "--threads",
            str(threads),
            "--xeq",
            "--score-min",
            "C,0,0",  # only perfect-score (exact) alignments
            "--np",
            str(N_penalty),
            "--n-ceil",
            f"C,0,{max_ambiguous}",
            "-F",
            f"{k},1",  # extract every k-mer (stride 1) from each read
            "-R",
            "1",
            "-N",
            "0",
            "-L",
            "31",
            "-i",
            "C,1,0",
            "--no-1mm-upfront",
            "--no-unal",  # do not write unaligned k-mers
            "--no-hd",  # suppress SAM header lines
            "-x",
            index_prefix,
            "-U",
            read_fa,
            "-S",
            output_sam_file,
        ]
        if strandedness:
            bowtie2_alignment_command.insert(3, "--norc")

        alignment_start = time.perf_counter()
        subprocess.run(
            bowtie2_alignment_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        runtimes["alignment_seconds"] = time.perf_counter() - alignment_start
        print(f"Alignment runtime: {runtimes['alignment_seconds']:.2f} s")

        # --- Recover which reads had at least one aligning k-mer. ---
        name_to_positions = _read_name_to_positions(read_fa)
        matched_positions = set()
        with open(output_sam_file) as sam:
            for line in tqdm(sam, desc="Scanning bowtie2 alignments"):
                if not line or line.startswith("@"):
                    continue
                qname = line.split("\t", 1)[0]
                # -F names each k-mer "{read_header}_{offset}"; strip the offset.
                read_name = qname.rsplit("_", 1)[0]
                for pos in name_to_positions.get(read_name, ()):
                    matched_positions.add(pos)

        return sorted(matched_positions)
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    read_fasta = "/home/jrich/Desktop/varseek-examples/data/cosmic_vs_denovo/vk_ref_cosmic/vcrs.fa"
    ref_fastas = ["/home/jrich/data/reference/t2t_CHM13v2/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"]  # ["/home/jrich/Desktop/varseek-examples/data/kb_quant_tests/kb_ref_out_cdna/cdna.fasta", "/home/jrich/data/reference/t2t_CHM13v2/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"]
    work_dir = "/home/jrich/Desktop/varseek-examples/data/bowtie2_kmer_match_test"
    for ref_fasta in ref_fastas:
        matched_positions = kmer_match_bowtie2(ref_fasta, read_fasta, k=41, threads=32, work_dir=work_dir)
        print(f"Number of reads in {read_fasta} that match at least one k-mer in {ref_fasta}: {len(matched_positions)}")



# subprocess.run([bowtie2_build, "--threads", str(threads), ref_fa, index_prefix,], check=True)