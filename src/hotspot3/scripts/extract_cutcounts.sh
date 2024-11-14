#!/bin/bash
set -e -o pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <input.bam> <chromosome>"
  exit 1
fi

AWK_EXE=$(which mawk 2>/dev/null || which awk)
BAM_FILE=$1
CHROM=$2

INPUT_CMD="samtools view -b $BAM_FILE $CHROM"


eval "$INPUT_CMD" \
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
