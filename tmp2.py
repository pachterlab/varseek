import os
from pathlib import Path
from typing import Literal, get_args, Optional, Annotated
from pydantic import validate_call, Field, AfterValidator, ConfigDict
import subprocess
import logging
import json
import requests
import kb_python
import numpy as np
import tempfile
import pandas as pd
import time

logger = logging.getLogger(__name__)

SPECIES = Literal["human"]
WORKFLOWS = Literal["standard", "nac", "custom"]
REFERENCE_TYPES = Literal["genome", "cdna", "transcriptome", "genome_plus_transcriptome"]

# species: reference_type: k: url
species_to_url = {
    "human": {
        "genome": {
            "41": "https://example.com/human_index.idx"
        },
        "cdna": {
            "41": "https://example.com/human_cdna_index.idx"
        },
        "transcriptome": {
            "41": "https://example.com/human_transcriptome_index.idx"
        },
        # "genome_plus_transcriptome": {
        #     "41": "https://example.com/human_genome_plus_transcriptome_index.idx"
        # },
    },
}

species_to_url

class Types:
    @staticmethod
    def must_be_odd(v: int) -> int:
        if v % 2 == 0:
            raise ValueError("Value must be odd")
        return v

    Odd3to63 = Annotated[
        int,
        Field(ge=3, le=63),
        AfterValidator(must_be_odd),
    ]

    @staticmethod
    def existing_file(p: str) -> str:
        if not Path(p).is_file():
            raise ValueError(f"{p} is not an existing file")
        return p
    
    ExistingFile = Annotated[
        str,
        AfterValidator(existing_file),
    ]

    @staticmethod
    def new_file(p: str) -> str:
        if Path(p).exists():
            raise ValueError(f"{p} already exists")
        return p

    NewFile = Annotated[
        str,
        AfterValidator(new_file),
    ]
    
    ExistingDirectory = Annotated[
        str,
        AfterValidator(lambda p: str(Path(p).resolve()) if Path(p).is_dir() else ValueError(f"{p} is not an existing directory")),
    ]
    
    PotentialDirectory = Annotated[
        str,
        AfterValidator(lambda p: str(Path(p).resolve())),
    ]

    @staticmethod
    def _validate_file(
        p: str,
        *,
        exists: Optional[bool],
        extensions: tuple[str, ...],
        zip_allowed: bool = False,
    ) -> str:
        path = Path(p)

        if exists is True and not path.is_file():
            raise ValueError(f"{p} is not an existing file")

        if exists is False and path.exists():
            raise ValueError(f"{p} already exists")

        path.parent.mkdir(parents=True, exist_ok=True)

        suffixes = path.suffixes

        # Allow .gz or .zip after the primary extension
        if zip_allowed and suffixes and suffixes[-1] in (".gz", ".zip"):
            suffixes = suffixes[:-1]

        # Only the final extension matters — filenames can contain dots in the stem
        # (e.g. Homo_sapiens.GRCh38.dna.primary_assembly.fa).
        ext = suffixes[-1] if suffixes else ""

        if ext not in extensions:
            allowed = ", ".join(
                f"{e}[.gz/.zip]" if zip_allowed else e
                for e in extensions
            )
            raise ValueError(f"{p} must have extension: {allowed}")

        return p

    ExistingIndex = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=True,
                extensions=(".idx",),
            )
        ),
    ]

    ExistingTxt = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=True,
                extensions=(".txt",),
            )
        ),
    ]

    ExistingFasta = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=True,
                extensions=(".fa", ".fasta", ".fna"),
                zip_allowed=True,
            )
        ),
    ]

    ExistingExecutable = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=True,
                extensions=("",),  # any filename
            )
        ),
    ]

    NewIndex = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=False,
                extensions=(".idx",),
            )
        ),
    ]

    NewTxt = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=False,
                extensions=(".txt",),
            )
        ),
    ]

    NewFasta = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=False,
                extensions=(".fa", ".fasta", ".fna"),
                zip_allowed=True,
            )
        ),
    ]

    NewGtf = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=False,
                extensions=(".gtf",),
            )
        ),
    ]

    ExistingFasta = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=True,
                extensions=(".fa", ".fasta", ".fna"),
                zip_allowed=True,
            )
        ),
    ]

    ExistingGtf = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=True,
                extensions=(".gtf",),
            )
        ),
    ]

    PotentialIndex = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=None,
                extensions=(".idx",),
            )
        ),
    ]

    PotentialTxt = Annotated[
        str,
        AfterValidator(
            lambda p: Types._validate_file(
                p,
                exists=None,
                extensions=(".txt",),
            )
        ),
    ]

    @staticmethod
    def existing_executable(p: str) -> str:
        if not Path(p).is_file():
            raise ValueError(f"{p} is not an existing file")
        if not os.access(p, os.X_OK):
            raise ValueError(f"{p} is not executable")
        return p

    ExistingExecutable = Annotated[
        str,
        AfterValidator(existing_executable),
    ]

def create_identity_t2g(fasta, t2g):
    with open(fasta, encoding="utf-8") as fin, open(t2g, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith(">"):
                header = line[1:].split(None, 1)[0]
                fout.write(f"{header}\t{header}\n")

def download_box_url(url, output_folder=".", output_file_name=None, verbose=True):
    if not output_file_name:
        output_file_name = url.split("/")[-1]
    if "/" not in output_file_name:
        output_file_path = os.path.join(output_folder, output_file_name)
    else:
        output_file_path = output_file_name
        output_folder = os.path.dirname(output_file_name)
    os.makedirs(output_folder, exist_ok=True)

    # Download the file
    response = requests.get(url, stream=True, timeout=(10, 90))
    if response.status_code == 200:
        with open(output_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        if verbose:  
            print(f"File downloaded successfully to {output_file_path}")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")

@validate_call
def run_kb_ref(
    index: Types.PotentialIndex,
    workflow: WORKFLOWS,
    dna_fasta: Types.ExistingFasta,
    t2g: Optional[Types.NewTxt] = None,
    f1: Optional[Types.NewFasta] = None,
    f2: Optional[Types.NewFasta] = None,
    c1: Optional[Types.NewTxt] = None,
    c2: Optional[Types.NewTxt] = None,
    k: Optional[Types.Odd3to63] = None,
    threads: Optional[int] = None,
    kallisto: Optional[Types.ExistingExecutable] = None,
    bustools: Optional[Types.ExistingExecutable] = None,
    gtf: Optional[Types.ExistingGtf] = None,
    tmp : Optional[Types.PotentialDirectory] = None,
    skip_index : bool = False
):
    kb_ref_command = ["kb", "ref", "--workflow", workflow, "--make-unique", "-i", index]
    if workflow != "custom":
        kb_ref_command += ["-g", t2g, "-f1", f1, "--d-list=None"]
    
    if workflow == "nac":
        if f2 is None or c1 is None or c2 is None:
            raise ValueError("f2, c1, and c2 are required for 'nac' workflow")
        kb_ref_command += ["-f2", f2, "-c1", c1, "-c2", c2]
    else:
        for arg in [f2, c1, c2]:
            if arg is not None:
                logger.warning(f"{arg} is ignored for '{workflow}' workflow")
    
    if k:
        kb_ref_command += ["-k", str(k)]
    if threads:
        kb_ref_command += ["-t", str(threads)]
    if kallisto:
        kb_ref_command += ["--kallisto", kallisto]
    if bustools:
        kb_ref_command += ["--bustools", bustools]
    if tmp:
        kb_ref_command += ["--tmp", tmp]
    logging_level = logging.getLevelName(logger.getEffectiveLevel())
    # if logging_level == "DEBUG":
    #     kb_ref_command += ["--verbose"]
    kb_ref_command += [dna_fasta]
    
    if workflow == "custom":
        if gtf is not None:
            logger.warning("gtf file is ignored for 'custom' workflow")
        # create_identity_t2g(dna_fasta, t2g)
    else:
        if gtf is None:
            raise ValueError(f"gtf file is required for '{workflow}' workflow")
        kb_ref_command += [gtf]
    
    if skip_index:
        if not os.path.exists(index):
            open(index, "w").close()  # make empty file at index - will allow kb ref to create f1 without creating index
    
    logger.debug(f"Running kb ref command: {' '.join(kb_ref_command)}")
    start = time.perf_counter()
    subprocess.run(kb_ref_command, check=True)
    logger.info(f"kb ref runtime: {time.perf_counter() - start:.2f} s")

@validate_call
def make_reference_index(
    dna_fasta: Types.ExistingFasta,
    reference_type: REFERENCE_TYPES,
    out_dir: Types.PotentialDirectory = "reference_index",
    index: Optional[Types.NewIndex] = None,
    t2g: Optional[Types.NewTxt] = None,
    gtf: Optional[Types.ExistingGtf] = None,
    k: Optional[Types.Odd3to63] = None,
    threads: Optional[int] = None,
    kallisto: Optional[Types.ExistingExecutable] = None,
    bustools: Optional[Types.ExistingExecutable] = None,
    overwrite: bool = False,  #! not yet working with type-checking
    species: Optional[SPECIES] = None,
    tmp: Optional[Types.PotentialDirectory] = None
):
    if index is None:
        index = os.path.join(out_dir, f"{reference_type}.idx")
    if t2g is None:
        t2g = os.path.join(out_dir, f"{reference_type}_t2g.txt")
    f1 = os.path.join(out_dir, "cdna.fasta")
    
    if os.path.exists(index) or os.path.exists(t2g):
        if overwrite:
            logger.warning(f"Overwriting existing index/t2g files at {index} and/or {t2g}")
        else:
            raise FileExistsError(f"Index or t2g files already exist at {index} and/or {t2g}. Use overwrite=True to overwrite.")
    
    if species is not None:
        logger.info(f"Downloading reference index for species: {species}")
        download_box_url(species_to_url[species][reference_type][str(k)], output_folder=out_dir, output_file_name=f"{reference_type}.idx", verbose=True)
        logger.info(f"Downloaded reference index to {index}")
        return  # Exit after downloading the reference files
    
    if reference_type == "genome":  # custom-dna
        t2g, f1 = None, None
        run_kb_ref(
            index = index,
            t2g = t2g,
            f1 = f1,
            workflow = "custom",  #* map to dna directly
            dna_fasta = dna_fasta,
            gtf = gtf,
            k = k,
            threads = threads,
            kallisto = kallisto,
            bustools = bustools,
            tmp = tmp
        )
    elif reference_type == "cdna":  # standard OR nac-cell
        run_kb_ref(
            index = index,
            t2g = t2g,
            f1 = f1,
            workflow = "standard",  #* map to cdna
            dna_fasta = dna_fasta,
            gtf = gtf,
            k = k,
            threads = threads,
            kallisto = kallisto,
            bustools = bustools,
            tmp = tmp
        )
    elif reference_type == "transcriptome":  # nac-total
        f2 = os.path.join(out_dir, "nascent.fasta")
        c1 = os.path.join(out_dir, "cdna.txt")
        c2 = os.path.join(out_dir, "nascent.txt")
        run_kb_ref(
            index = index,
            t2g = t2g,
            f1 = f1,
            workflow = "nac",  #* map to spliced + unspliced
            dna_fasta = dna_fasta,
            gtf = gtf,
            k = k,
            threads = threads,
            kallisto = kallisto,
            bustools = bustools,
            f2 = f2,
            c1 = c1,
            c2 = c2,
            tmp = tmp
        )
    elif reference_type == "genome_plus_transcriptome":  # custom-dna + standard/nac-cell/nac-total
        #* make cdna fasta
        cdna_index = os.path.join(out_dir, "cdna.idx")
        cdna_t2g = os.path.join(out_dir, "cdna_t2g.txt")
        if not os.path.exists(f1):
            run_kb_ref(
                index = cdna_index,
                t2g = cdna_t2g,
                f1 = f1,
                workflow = "standard",
                dna_fasta = dna_fasta,
                k = k,
                threads = threads,
                kallisto = kallisto,
                bustools = bustools,
                gtf = gtf,
                tmp = tmp,
                skip_index = True
            )

        #* combine genome and cdna fasta into a single fasta for kb ref
        genome_plus_transcriptome_fasta = dna_fasta.replace(".fasta", "_plus_cdna.fasta").replace(".fa", "_plus_cdna.fa").replace(".fna", "_plus_cdna.fna")
        with open(genome_plus_transcriptome_fasta, "wb") as wfd:
            for f in [dna_fasta, f1]:
                with open(f, "rb") as fd:
                    wfd.write(fd.read())

        t2g, f1 = None, None
        run_kb_ref(
            index = index,
            t2g = t2g,
            f1 = f1,
            workflow = "custom",
            dna_fasta = genome_plus_transcriptome_fasta,  #* map to dna + cdna directly
            k = k,
            threads = threads,
            kallisto = kallisto,
            bustools = bustools,
        )
    else:
        raise ValueError(f"Invalid reference_type: {reference_type}. Must be one of {get_args(REFERENCE_TYPES)}")

    logger.info(f"Reference index created at {index}")

def fasta_to_fastq(fasta_file, fastq_file = None, overwrite=False):
    if fastq_file is None:
        fastq_file = os.path.splitext(fasta_file)[0] + ".fastq"
    if os.path.exists(fastq_file):
        if not overwrite:
            logger.warning(f"Fastq file {fastq_file} already exists. Use overwrite=True to overwrite. Skipping conversion.")
            return fastq_file
        else:
            logger.warning(f"Fastq file {fastq_file} already exists. Overwriting.")
    with open(fasta_file, 'r') as fasta, open(fastq_file, 'w') as fastq:
        for line in fasta:
            if line.startswith('>'):
                header = line[1:].strip()
                sequence = next(fasta).strip()
                fastq.write(f"@{header}\n{sequence}\n+\n{'I' * len(sequence)}\n")
    return fastq_file

RUNTIMES = {key: {} for key in get_args(REFERENCE_TYPES)}

@validate_call
def pseudoalign(
    vcrs_fasta: Types.ExistingFasta,
    reference_type: REFERENCE_TYPES,
    out_dir: Types.PotentialDirectory = "kallisto_bus_out",
    dna_fasta: Optional[Types.ExistingFasta] = None,
    index_dir: Optional[Types.PotentialDirectory] = None,
    index: Optional[Types.PotentialIndex] = None,
    t2g: Optional[Types.PotentialTxt] = None,
    gtf: Optional[Types.ExistingGtf] = None,
    k: Optional[Types.Odd3to63] = None,
    threads: Optional[int] = None,
    kallisto: Optional[Types.ExistingExecutable] = None,
    bustools: Optional[Types.ExistingExecutable] = None,
    overwrite: bool = False,  #! not yet working with type-checking
    species: Optional[SPECIES] = None,
    tmp: Optional[Types.PotentialDirectory] = None
):
    if index is None:
        index = os.path.join(index_dir, f"{reference_type}.idx")
    if t2g is None:
        t2g = os.path.join(index_dir, f"{reference_type}.txt")
    if not os.path.exists(index) or overwrite:
        if not os.path.exists(index):
            logger.info(f"Index file not found at {index}. Will create file.")
        else:
            logger.info(f"Overwriting existing index file at {index}.")
        start = time.perf_counter()
        make_reference_index(
            dna_fasta=dna_fasta,
            reference_type=reference_type,
            out_dir=index_dir,
            index=index,
            t2g=t2g,
            gtf=gtf,
            k=k,
            threads=threads,
            kallisto=kallisto,
            bustools=bustools,
            overwrite=overwrite,
            species=species,
            tmp=tmp
        )
        runtime = time.perf_counter() - start
        logger.info(f"kb ref runtime: {runtime:.2f} s")
        RUNTIMES[reference_type]["kb_ref"] = runtime

    vcrs_fastq = fasta_to_fastq(vcrs_fasta)

    if kallisto is None:
        kb_dir = Path(kb_python.__file__).parent
        kallisto_exec = "kallisto_k64" if (k is not None and k > 32) else "kallisto"
        kallisto = str(kb_dir / "bins" / "linux" / "kallisto" / kallisto_exec)
    
    if bustools is None:
        kb_dir = Path(kb_python.__file__).parent
        bustools = str(kb_dir / "bins" / "linux" / "bustools" / "bustools")
    
    kallisto_bus_command = [kallisto, "bus", "-i", index, "-o", out_dir, "-x", "BULK", "--num", "--unstranded", "--union"]
    if threads:
        kallisto_bus_command += ["-t", str(threads)]
    logging_level = logging.getLevelName(logger.getEffectiveLevel())
    # if logging_level == "DEBUG":
    #     kallisto_bus_command += ["--verbose"]
    kallisto_bus_command += [vcrs_fastq]

    # kb_count_command = ["kb", "count", "-i", index, "-g", t2g, "-o", out_dir, "-x", "bustools", "--single", "--parity", "single", "--strand", "unstranded", "--union", "--mm", "--num"]
    # if threads:
    #     kb_count_command += ["-t", str(threads)]
    # if kallisto:
    #     kb_count_command += ["--kallisto", kallisto]
    # if bustools:
    #     kb_count_command += ["--bustools", bustools]
    # if tmp:
    #     kb_count_command += ["--tmp", tmp]
    # kb_count_command += [vcrs_fastq]

    bus_file = os.path.join(out_dir, "output.bus")
    bus_txt = os.path.join(out_dir, "output.txt")
    if not os.path.exists(bus_file) or overwrite:
        if not os.path.exists(bus_file):
            logger.info(f"BUS file {bus_file} does not exist. Will create directory and run kallisto bus.")
        else:
            logger.info(f"Overwriting existing BUS file at {bus_file}.")
        logger.debug(f"Running kallisto bus command: {' '.join(kallisto_bus_command)}")
        start = time.perf_counter()
        subprocess.run(kallisto_bus_command, check=True)
        runtime = time.perf_counter() - start
        logger.info(f"kallisto bus runtime: {runtime:.2f} s")
        RUNTIMES[reference_type]["kallisto_bus"] = runtime

    bustools_text_command = [bustools, "text", "-f", "-o", bus_txt, bus_file]
    if not os.path.exists(bus_txt) or overwrite:
        if not os.path.exists(bus_txt):
            logger.info(f"Bus text file {bus_txt} does not exist. Will create file.")
        else:
            logger.info(f"Overwriting existing bus text file at {bus_txt}.")
        logger.debug(f"Running bustools text command: {' '.join(bustools_text_command)}")
        subprocess.run(bustools_text_command, check=True)

    bus_df = pd.read_csv(bus_txt, sep="\t", header=None, names=["barcode", "umi", "gene", "count", "read"], usecols=["read"])
    aligned_headers = sorted(set(bus_df["read"].unique()))
    n_pseudoaligned_set = len(aligned_headers)

    # if kallisto is None:
    #     kb_dir = Path(kb_python.__file__).parent
    #     kallisto = str(kb_dir / "bins" / "linux" / "kallisto" / "kallisto")

    # kallisto_quant_command = [kallisto, "quant", "-i", index, "--single", "-l", "51", "-s", "5", "--pseudobam", "-o", out_dir]
    # if threads:
    #     kallisto_quant_command += ["-t", str(threads)]
    # kallisto_quant_command += [vcrs_fastq]

    # logger.debug(f"Running kallisto quant command: {' '.join(kallisto_quant_command)}")
    # subprocess.run(kallisto_quant_command, check=True)

    # import pysam
    # pseudobam_path = os.path.join(out_dir, "pseudoalignments.bam")
    # aligned_headers_set = set()
    # with pysam.AlignmentFile(pseudobam_path, "rb") as bam:
    #     for read in bam:
    #         if not read.is_unmapped:
    #             aligned_headers_set.add(read.query_name)
    # n_pseudoaligned_set = len(aligned_headers_set)

    #* Cross-check against kallisto's own tally in run_info.json.
    run_info_path = os.path.join(out_dir, "run_info.json")
    with open(run_info_path, encoding="utf-8") as fh:
        run_info = json.load(fh)
    n_pseudoaligned = run_info["n_pseudoaligned"]
    n_processed = run_info["n_processed"]
    assert n_pseudoaligned == n_pseudoaligned_set, f"Mismatch between run_info.json ({n_pseudoaligned}) and BUS ({n_pseudoaligned_set}) pseudoaligned read counts"
    logger.info(f"reference_type: {reference_type}, pseudoaligned: {n_pseudoaligned} / {n_processed} reads ({n_pseudoaligned/n_processed:.2%})")
    return aligned_headers

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def run_pseudoalign_on_vcrs_df(
    df: pd.DataFrame,
    reference_type: REFERENCE_TYPES,
    index_dir: Types.PotentialDirectory,
    out_dir: Types.PotentialDirectory,
    dna_fasta: Optional[Types.ExistingFasta] = None,
    gtf: Optional[Types.ExistingGtf] = None,
    k: Optional[Types.Odd3to63] = None,
    threads: Optional[Annotated[int, Field(gt=0)]] = None,
    fasta_col: str = "fasta_format",
    species: Optional[SPECIES] = None,
):
    if fasta_col not in df.columns:
        raise ValueError(f"Column '{fasta_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    vcrs_fasta = tempfile.NamedTemporaryFile(delete=True, suffix=".fasta").name
    with open(vcrs_fasta, "w", encoding="utf-8") as fasta_file:
        fasta_file.write("".join(df[fasta_col].values))
    
    rows_to_remove = pseudoalign(
        reference_type=reference_type,
        index_dir=index_dir,
        out_dir=out_dir,
        vcrs_fasta=vcrs_fasta,
        dna_fasta=dna_fasta,
        gtf=gtf,
        threads=threads,
        k=k,
        species=species,
    )

    len_df = len(df)
    mask = np.ones(len_df, dtype=bool)
    mask[rows_to_remove] = False
    df = df.iloc[mask]

    logger.info(f"After pseudoalignment, {len(df)} / {len_df} rows remain ({len(df)/len_df:.2%})")
    return df
    
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    base_dir = "/home/jrich/Desktop/varseek-examples/data/kallisto_bus_tests3"
    vcrs_fasta = "/home/jrich/Desktop/varseek-examples/data/cosmic_vs_denovo/vk_ref_cosmic/vcrs.fa"
    dna_fasta = "/home/jrich/data/reference/t2t_CHM13v2/data/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"  # "/home/jrich/data/reference/ensembl_grch38_release114/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    gtf = "/home/jrich/data/reference/t2t_CHM13v2/data/GCF_009914755.1/genomic.gtf"  # "/home/jrich/data/reference/ensembl_grch38_release114/Homo_sapiens.GRCh38.114.gtf"
    k = 41
    total_headers = sum(1 for line in open(vcrs_fasta) if line.startswith(">"))

    reference_types = ["genome", "cdna", "transcriptome", "genome_plus_transcriptome"]
    reference_type_to_aligned_headers = {}
    for reference_type in reference_types:
        print(f"Running pseudoalign for reference_type: {reference_type}")
        index_dir = f"{base_dir}/kb_ref_out_{reference_type}"
        out_dir = f"{base_dir}/kallisto_bus_out_{reference_type}"
        os.makedirs(out_dir, exist_ok=True)
        reference_type_to_aligned_headers[reference_type] = pseudoalign(
            reference_type=reference_type,
            index_dir=index_dir,
            out_dir=out_dir,
            vcrs_fasta=vcrs_fasta,
            dna_fasta=dna_fasta,
            gtf=gtf,
            threads=32,
            k=k,
        )

    # write to text file
    with open(f"{base_dir}/results.txt", "w") as f:
        f.write(f"Total fasta headers: {total_headers}\n")
        for reference_type, aligned_headers in reference_type_to_aligned_headers.items():
            n_aligned = len(aligned_headers)
            f.write(f"Aligned headers for {reference_type}: {n_aligned} / {total_headers} ({n_aligned/total_headers:.2%})\n")
        f.write("\nRuntimes:\n")
        for reference_type, runtimes in RUNTIMES.items():
            f.write(f"{reference_type}:\n")
            for step, runtime in runtimes.items():
                f.write(f"  {step}: {runtime:.2f} s\n")
