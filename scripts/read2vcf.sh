#!/usr/bin/env bash
set -euo pipefail

check_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required tool '$1' is not installed or not in PATH."
    exit 1
  fi
}

# Defaults
THREADS=1
FASTA_REF=""
REF_INDEX=""
SEED_LENGTH=""  # bowtie -L
SCORE_MIN=""     # bowtie --score-min
OUTPUT="out.vcf.gz"
INCLUDE_EXPR="INFO/AD[1] >= 3"
SKIP_INDELS=""
FASTQ1=""
FASTQ2=""
OUT_BAM_DIR=""
MERGE_BAM_FILES=false
MERGED_BAM="merged_merged_pseudobulk.bam"

# Helper
usage() {
  cat <<EOF
Usage: $0 [options] [input1 [input2 ...]]

Options:
  --threads INT          Number of threads (default: 1)
  -f, --fasta-ref FILE   Reference fasta file (required)
  -x REF_INDEX_BASENAME  Bowtie2 index basename (required if FASTQ inputs)
  -o, --output FILE      Output VCF (default: out.vcf.gz)
  -i, --include EXPR     bcftools filter expression (default: 'INFO/AD[1] >= 3')
  -I, --skip-indels      Skip indels in the output
  -1 FILE                Paired-end FASTQ read 1
  -2 FILE                Paired-end FASTQ read 2
  --out-bam-dir DIR      Directory to write output BAM files (default: directory of first FASTQ)
  --merge-bam-files      Merge BAM files into a single BAM (default: false)
  --merged-bam FILE      Directory to read/write merged BAM file (default: merged_merged_pseudobulk.bambam)
  -h, --help             Show this help

Positional arguments:
  Input BAM or FASTQ files, or a directory containing such. If -1/-2 are specified, those are used instead.

Example:
  $0 --threads 4 -f ref.fa -x ref_idx -o out.vcf.gz -i 'FORMAT/AD[1]>=5' aln1.bam aln2.bam
  $0 --threads 4 -f ref.fa -x ref_idx -o out.vcf.gz -1 reads_1.fq -2 reads_2.fq
  $0 --threads 4 -f ref.fa -x ref_idx -o out.vcf unpaired_reads.fq
EOF
  exit 1
}

# Parse arguments
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      THREADS="$2"
      shift 2
      ;;
    -f|--fasta-ref)
      FASTA_REF="$2"
      shift 2
      ;;
    -x)
      REF_INDEX="$2"
      shift 2
      ;;
    -L)
      SEED_LENGTH="$2"
      shift 2
      ;;
    --score-min)
      SCORE_MIN="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    -i|--include)
      INCLUDE_EXPR="$2"
      shift 2
      ;;
    -I|--skip-indels)
      SKIP_INDELS=1
      shift
      ;;
    -1)
      FASTQ1="$2"
      shift 2
      ;;
    -2)
      FASTQ2="$2"
      shift 2
      ;;
    --out-bam-dir)
      OUT_BAM_DIR="$2"
      shift 2
      ;;
    --merge-bam-files)
      MERGE_BAM_FILES=true
      shift
      ;;
    --merged-bam)
      MERGED_BAM="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# Append remaining positional arguments
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

check_tool bcftools
check_tool samtools
check_tool bowtie2

# Validation
if [[ ${#ARGS[@]} -eq 0 && -z "$FASTQ1" ]]; then
  echo "Error: No input files provided."
  usage
fi

if [[ -z "$FASTA_REF" ]]; then
  echo "Error: --fasta-ref is required."
  usage
fi

if [[ -n "$FASTQ1" && -z "$REF_INDEX" ]]; then
  echo "Error: -x is required when FASTQs are provided."  # I check for positional FASTQ and REF_INDEX later
  usage
fi

if [[ -n "$FASTQ2" && -z "$FASTQ1" ]]; then
  echo "Error: -2 specified without -1."
  usage
fi

if [[ "$OUTPUT" != *.vcf* ]]; then
  echo "Error: --output must be a .vcf file path."
  usage
fi

if ! [[ "$THREADS" =~ ^[0-9]+$ ]] || [[ "$THREADS" -lt 1 ]]; then
  echo "Error: --threads must be an integer >= 1."
  usage
fi

if [[ "$INCLUDE_EXPR" == "INFO/AD[1] >= 1" || -z "$INCLUDE_EXPR" ]]; then
  echo "Warning: filtering by a minimum count threshold is highly recommended."
  echo "Additionally, indels observed once will not be output regardless of settings (bcftools mpileup behavior)."
fi

BOWTIE2_OPTIONS=""
if [[ -n "$SEED_LENGTH" ]]; then
  BOWTIE2_OPTIONS+=" -L $SEED_LENGTH"
fi
if [[ -n "$SCORE_MIN" ]]; then
  BOWTIE2_OPTIONS+=" --score-min $SCORE_MIN"
fi

# Collect BAMs to pass to bcftools mpileup
BAM_FILES=()

# If FASTQs are specified vis FASTQ1/FASTQ2, process them
if [[ -n "$FASTQ1" ]]; then
  # Build index if needed
  if [[ ! -f "${REF_INDEX}.1.bt2" ]]; then
    echo "Bowtie2 index not found, building..."
    bowtie2-build "$FASTA_REF" "$REF_INDEX"
  fi

  # puts out_bam in the same directory as the first entry of FASTQ1 (if OUT_BAM_DIR not provided), and makes filename "reads_aligned_to_<ref>.bam"
  FIRST_FASTQ="${FASTQ1%% *}"
  if [[ -z "$OUT_BAM_DIR" ]]; then
    OUT_BAM_DIR="$(dirname "$FIRST_FASTQ")"
  fi
  REF_FASTA_BASE="$(basename "$FASTA_REF" | sed -E 's/\.(fa|fasta|fna)(\.gz)?$//' | tr '.' '_')"
  OUT_BAM="$OUT_BAM_DIR/$reads_aligned_to_${REF_FASTA_BASE}.bam"

  FASTQ1_CSV=$(IFS=,; echo "${FASTQ1[*]}")

  mkdir -p "$OUT_BAM_DIR"
  if [[ -f "$OUT_BAM" ]]; then
    echo "Skipping alignment: $OUT_BAM already exists."
  else
    if [[ -n "$FASTQ2" ]]; then
      FASTQ2_CSV=$(IFS=,; echo "${FASTQ2[*]}")
      echo "Aligning paired-end FASTQs..."
      bowtie2 --xeq --very-sensitive $BOWTIE2_OPTIONS --threads "$THREADS" -x "$REF_INDEX" -1 "$FASTQ1_CSV" -2 "$FASTQ2_CSV" \
        | samtools sort --threads "$THREADS" -o "$OUT_BAM"
    else
      echo "Aligning single-end FASTQ..."
      bowtie2 --xeq --very-sensitive $BOWTIE2_OPTIONS --threads "$THREADS" -x "$REF_INDEX" -U "$FASTQ1_CSV" \
        | samtools sort --threads "$THREADS" -o "$OUT_BAM"
    
    if [[ ! -f "${OUT_BAM}.bai" ]]; then
      echo "Indexing BAM '$OUT_BAM'..."
      samtools index -@ "$THREADS" "$OUT_BAM"
    fi

    BAM_FILES+=("$OUT_BAM")
    fi
  fi
  
fi

# Handles the case where input files are directories
EXPANDED_ARGS=()

# Loop over all positional arguments
for INPUT in "${ARGS[@]}"; do
  if [[ -d "$INPUT" ]]; then
    echo "Found directory: $INPUT"
    FILES_FOUND=0
    while IFS= read -r -d $'\0' FILE; do
      # Extract file extension
      EXT="${FILE##*.}"
      case "$EXT" in
        bam|fq|fastq)
          FILES_FOUND=1
          EXPANDED_ARGS+=("$FILE")
          ;;
        *)
          echo "Skipping unsupported file: $FILE"
          ;;
      esac
    done < <(find "$INPUT" -maxdepth 1 -type f -print0 | sort -z)  # maxdepth 1 means only files in the directory, not subdirectories
    if [[ $FILES_FOUND -eq 0 ]]; then
      echo "Error: directory '$INPUT' contains no FASTQ or BAM files."
      exit 1
    fi
  else
    EXPANDED_ARGS+=("$INPUT")
  fi
done

# Replace ARGS with the expanded list
ARGS=("${EXPANDED_ARGS[@]}")

# Process positional inputs - first, collect all FASTQs
FASTQ_INPUTS=()

# Process positional inputs
for INPUT in "${ARGS[@]}"; do
  EXT="${INPUT##*.}"
  case "$EXT" in
    bam)
      BAM_FILES+=("$INPUT")
      ;;
    fq|fastq)
      if [[ -z "$REF_INDEX" ]]; then
        echo "Error: FASTQ input '$INPUT' requires -x reference index."
        exit 1
      fi
      if [[ ! -f "${REF_INDEX}.1.bt2" ]]; then
        echo "Bowtie2 index not found, building..."
        bowtie2-build "$FASTA_REF" "$REF_INDEX"
      fi
      FASTQ_INPUTS+=("$INPUT")
      ;;
    *)
      echo "Error: Unsupported file type '$INPUT'"
      exit 1
      ;;
  esac
done

# If there are FASTQs to align, do it one at a time
if [[ ${#FASTQ_INPUTS[@]} -gt 0 ]]; then
  for FASTQ_FILE in "${FASTQ_INPUTS[@]}"; do
    # Build output BAM path
    if [[ -z "$OUT_BAM_DIR" ]]; then
      OUT_BAM_DIR="$(dirname "$FASTQ_FILE")"
    fi
    FQ_BASE="$(basename "$FASTQ_FILE")"
    FQ_BASE="${FQ_BASE%%.*}"
    REF_FASTA_BASE="$(basename "$FASTA_REF" | sed -E 's/\.(fa|fasta|fna)(\.gz)?$//' | tr '.' '_')"
    OUT_BAM="$OUT_BAM_DIR/${FQ_BASE}_aligned_to_${REF_FASTA_BASE}.bam"

    mkdir -p "$OUT_BAM_DIR"
    if [[ -f "$OUT_BAM" ]]; then
      echo "Skipping alignment: $OUT_BAM already exists."
    else
      echo "Aligning single-end FASTQ '$FASTQ_FILE' to '$OUT_BAM'..."
      bowtie2 --xeq --very-sensitive $BOWTIE2_OPTIONS --threads "$THREADS" -x "$REF_INDEX" -U "$FASTQ_FILE" \
        | samtools sort --threads "$THREADS" -o "$OUT_BAM" -
      samtools index -@ "$THREADS" "$OUT_BAM"  # anytime I make a new BAM, I index it, even if it exists
    fi

    if [[ ! -f "${OUT_BAM}.bai" ]]; then
      echo "Indexing BAM '$OUT_BAM'..."
      samtools index -@ "$THREADS" "$OUT_BAM"
    fi

    BAM_FILES+=("$OUT_BAM")
  done
fi


# # If there are FASTQs to align, do it once for all - not a bad approach, but having many FASTQs can cause a bowtie error (arguments too long)
# if [[ ${#FASTQ_INPUTS[@]} -gt 0 ]]; then
#   OUT_BAM_DIR="$(dirname "${FASTQ_INPUTS[0]}")"
#   REF_FASTA_BASE="$(basename "$FASTA_REF" | sed -E 's/\.(fa|fasta|fna)(\.gz)?$//' | tr '.' '_')"
#   OUT_BAM="$OUT_BAM_DIR/reads_aligned_to_${REF_FASTA_BASE}.bam"

#   echo "Aligning all FASTQ inputs into '$OUT_BAM'..."
#   bowtie2 --xeq --very-sensitive $BOWTIE2_OPTIONS --threads "$THREADS" -x "$REF_INDEX" -U "$(IFS=,; echo "${FASTQ_INPUTS[*]}")" \
#     | samtools sort --threads "$THREADS" -o "$OUT_BAM"

#   BAM_FILES+=("$OUT_BAM")
# fi

if [[ ${#BAM_FILES[@]} -eq 0 ]]; then
  echo "Error: No BAM files to process."
  exit 1
fi

# Determine output format
if [[ "$OUTPUT" == *.gz ]]; then
  OUTPUT_TYPE="-Oz"
else
  OUTPUT_TYPE="-Ov"
fi

if $MERGE_BAM_FILES; then
    echo "Merging BAM files..."

    # Create BAM list file
    if [[ -n "${OUT_BAM_DIR:-}" ]]; then
        BAM_LIST_PATH="${OUT_BAM_DIR%/}/all_bams.txt"
    else
        BAM_LIST_PATH="all_bams.txt"
    fi
    printf "%s\n" "${BAM_FILES[@]}" > "$BAM_LIST_PATH"

    # make directory for merged BAM if it doesn't exist
    MERGED_BAM_DIR="$(dirname "$MERGED_BAM")"
    if [[ -n "$MERGED_BAM_DIR" && "$MERGED_BAM_DIR" != "." ]]; then
        mkdir -p "$MERGED_BAM_DIR"
    fi

    # Merge if not already done
    SORTED_BAM="${MERGED_BAM%.bam}.sorted.bam"
    if [[ ! -f "$SORTED_BAM" ]]; then
        samtools merge -@ "$THREADS" -b "$BAM_LIST_PATH" "$MERGED_BAM"
        samtools sort -@ "$THREADS" -o "$SORTED_BAM" "$MERGED_BAM"
        samtools index -@ "$THREADS" "$SORTED_BAM"
    else
        echo "Merged and sorted BAM already exists: $SORTED_BAM"
    fi

    INPUT_BAMS=("$SORTED_BAM")
else
    echo "Using individual BAM files without merging."
    INPUT_BAMS=("${BAM_FILES[@]}")
fi

echo "Processing with bcftools mpileup + filter..."
# echo "BAMs: ${BAM_FILES[*]}"
# echo "Output: $OUTPUT ($OUTPUT_TYPE)"
# echo "Filter expression: ${INCLUDE_EXPR:-None}"



bcftools mpileup \
    --threads "$THREADS" \
    -f "$FASTA_REF" \
    -a INFO/AD \
    -Q 0 \
    -d 10000 \
    ${SKIP_INDELS:+-I} \
    "${INPUT_BAMS[@]}" \
| bcftools filter \
  ${INCLUDE_EXPR:+-i "$INCLUDE_EXPR"} \
| bcftools norm -m - \
| bcftools view -e 'ALT="<*>"' \
  "$OUTPUT_TYPE" -o "$OUTPUT"

# | bcftools call -m -A \

echo "Program complete. VCF output written to $OUTPUT"
