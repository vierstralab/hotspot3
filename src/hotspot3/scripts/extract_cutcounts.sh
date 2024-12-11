#!/bin/bash
set -e -o pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 [-T reference.fasta] <input.bam> <chromosome(s)>"
  exit 1
fi

AWK_EXE=$(which mawk 2>/dev/null || which awk)

while [[ $# -gt 0 ]]; do
  case "$1" in
    -T)
      REFERENCE_FASTA="$2"
      shift 2
      ;;
    *.bam|*.cram)
      BAM_FILE="$1"
      shift
      ;;
    *)
      CHROMS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$BAM_FILE" || ${#CHROMS[@]} -eq 0 ]]; then
  echo "Error: Missing BAM/CRAM file or chromosomes."
  echo "Usage: $0 [-T reference.fasta] <input.bam|input.cram> <chromosome(s)...>"
  exit 1
fi

SAMTOOLS_CMD="samtools view -b"
if [[ -n "$REFERENCE_FASTA" ]]; then
  SAMTOOLS_CMD+=" -T $REFERENCE_FASTA"
fi
SAMTOOLS_CMD+=" $BAM_FILE ${CHROMS[*]}"

eval "$SAMTOOLS_CMD" \
  | bam2bed --do-not-sort \
  | "$AWK_EXE" -v FS="\t" -v OFS="\t" '
    {
      chrom = $1;
      read_start = $2;
      read_end = $3;
      strand = $6;
      if (strand == "+") {
        cut_start = read_start;
        cut_end = read_start + 1;
      } else {
        cut_start = read_end;
        cut_end = read_end + 1;
      }
      print chrom, cut_start, cut_end;
    }' \
  | sort-bed - \
  | uniq -c \
  | "$AWK_EXE" -v OFS='\t' '{ print $2,$3,$4,$1 }' \
  | { echo -e "#chr\tstart\tend\tcount"; cat; }
