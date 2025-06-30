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
if [[ -z "$FASTA_REF" ]] || [[ -z "$OUTPUT" ]]; then
  echo "Error: --fasta-ref and --output are required."
  usage
fi

if [[ -n "$FASTQ1" && -z "$REF_INDEX" ]]; then
  echo "Error: -x is required when FASTQs are provided."
  usage
fi

if [[ -n "$FASTQ2" && -z "$FASTQ1" ]]; then
  echo "Error: -2 specified without -1."
  usage
fi

if [[ ${#ARGS[@]} -eq 0 && -z "$FASTQ1" ]]; then
  echo "Error: No input files provided."
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

# If FASTQs are specified, process them
if [[ -n "$FASTQ1" ]]; then
  # Build index if needed
  if [[ ! -f "${REF_INDEX}.1.bt2" ]]; then
    echo "Bowtie2 index not found, building..."
    bowtie2-build "$FASTA_REF" "$REF_INDEX"
  fi

  if [[ -n "$FASTQ2" ]]; then
    echo "Aligning paired-end FASTQs..."
    OUT_BAM="$(basename "$FASTQ1" .fq).bam"
    bowtie2 --xeq --very-sensitive $BOWTIE2_OPTIONS --threads "$THREADS" -x "$REF_INDEX" -1 "$FASTQ1" -2 "$FASTQ2" \
      | samtools sort --threads "$THREADS" -o "$OUT_BAM"
    BAM_FILES+=("$OUT_BAM")
  else
    echo "Aligning single-end FASTQ..."
    OUT_BAM="$(basename "$FASTQ1" .fq).bam"
    bowtie2 --xeq --very-sensitive $BOWTIE2_OPTIONS --threads "$THREADS" -x "$REF_INDEX" "$FASTQ1" \
      | samtools sort --threads "$THREADS" -o "$OUT_BAM"
    BAM_FILES+=("$OUT_BAM")
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
    done < <(find "$INPUT" -maxdepth 1 -type f -print0 | sort -z)
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
      echo "Aligning single-end FASTQ '$INPUT'..."
      OUT_BAM="$(basename "$INPUT" .fq).bam"
      bowtie2 --xeq --very-sensitive $BOWTIE2_OPTIONS --threads "$THREADS" -x "$REF_INDEX" "$INPUT" \
        | samtools sort --threads "$THREADS" -o "$OUT_BAM"
      BAM_FILES+=("$OUT_BAM")
      ;;
    *)
      echo "Error: Unsupported file type '$INPUT'"
      exit 1
      ;;
  esac
done

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

echo "Processing with bcftools mpileup + filter..."
echo "BAMs: ${BAM_FILES[*]}"
echo "Output: $OUTPUT ($OUTPUT_TYPE)"
echo "Filter expression: ${INCLUDE_EXPR:-None}"

bcftools mpileup \
  --threads "$THREADS" \
  -f "$FASTA_REF" \
  -a INFO/AD \
  -Q 0 \
  -d 1000000 \
  ${SKIP_INDELS:+-I} \
  "${BAM_FILES[@]}" \
| bcftools filter \
  ${INCLUDE_EXPR:+-i "$INCLUDE_EXPR"} \
| bcftools norm -m - \
| bcftools view -e 'ALT="<*>"' \
  "$OUTPUT_TYPE" -o "$OUTPUT"

# | bcftools call -m -A \

echo "Program complete. VCF output written to $OUTPUT"
