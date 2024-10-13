#!/bin/bash
set -e -o pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <input.bam> <output.bed.gz>"
  exit 1
fi

AWK_EXE=$(which mawk 2>/dev/null || which awk)

bam2bed --do-not-sort < "$1" \
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
  | { echo -e "#chr\tstart\tend\tcount"; cat; } \
  | bgzip > "$2"

tabix -p bed "$2"