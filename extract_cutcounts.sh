AWK_EXE=$(which mawk 2>/dev/null || which awk)

bam2bed --do-not-sort < "$1" \
  | "$AWK_EXE" -v FS="\t" -v OFS="\t" '
    {
      strand = $6;
      read_start = $2;
      read_end = $3;
      read_id = $1;
      if (strand == "+") {
        cut_start = read_start;
        cut_end = read_start + 1;
      } else {
        cut_start = read_end;
        cut_end = read_end + 1;
      }
      print read_id, cut_start, cut_end;
    }' \
  | sort-bed - \
  | uniq -c \
  | "$AWK_EXE" -v OFS='\t' '{ print $2,$3,$4,$1 }' \
  | bgzip > $2

tabix -p bed $2