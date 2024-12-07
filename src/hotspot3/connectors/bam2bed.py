import subprocess
import importlib.resources as pkg_resources
from collections import defaultdict
import pysam
import pandas as pd
from io import StringIO

from hotspot3.io.logging import WithLogger
from hotspot3.helpers.models import NotEnoughDataForContig


def run_bam2bed(*args):
    with pkg_resources.path('hotspot3.scripts', 'extract_cutcounts.sh') as script:
        result = subprocess.run(
            f'bash {script} {" ".join(args)}',
            check=True,
            text=True,
            capture_output=True,
            shell=True,
        )
    return result


class BamFileCutsExtractor(WithLogger):
    def extract_all_chroms(self, bam_path, tabix_bed_path, chromosomes):
        """
        Run bam2bed conversion script.
        Very fast but can't be parallelized.
        """
        run_bam2bed(bam_path, *chromosomes, '|', 'bgzip', '>', tabix_bed_path)
        pysam.tabix_index(tabix_bed_path, preset='bed', force=True)


    def extract_chromosome_to_df(self, bam_path, chromosome: str) -> pd.DataFrame:
        """
        Run bam2bed for a single chromosome. Returns a pandas DataFrame.
        """
        result = run_bam2bed(bam_path, chromosome)
        df = pd.read_table(StringIO(result.stdout))
        return df.drop(columns=['#chr'])


    def extract_reads_pysam(self, bam_path, chromosome) -> pd.DataFrame:
        """
        Extract reads with pysam for a single chromosome.
        Slower than bam2bed conversion but can be heavily parallelized.
        """
        bed_counts = defaultdict(int)
        try:
            with pysam.AlignmentFile(bam_path) as bamfile:
                for read in bamfile.fetch(chromosome):
                    if read.is_unmapped or read.is_secondary or read.is_supplementary:
                        continue

                    if read.is_reverse:
                        cut_start = read.reference_end
                    else:
                        cut_start = read.reference_start
                    bed_counts[cut_start] += 1
        except ValueError:
            raise NotEnoughDataForContig

        bed_df = pd.DataFrame(
            [(start, count) for start, count in bed_counts.items()],
            columns=["start", "count"]
        ).sort_values("start")
        bed_df["end"] = bed_df["start"] + 1
        bed_df["chrom"] = chromosome

        return bed_df[['chrom', 'start', 'end', 'count']]