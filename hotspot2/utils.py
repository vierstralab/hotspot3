import pandas as pd
import dataclasses
import pysam
import io
import functools
import subprocess
import numpy as np


@dataclasses.dataclass
class ProcessorOutputData:
    identificator: str
    data_df: pd.DataFrame
    params_df: pd.DataFrame=None


def read_chrom_sizes(chrom_sizes):
    if chrom_sizes is None:
        raise NotImplementedError("hg38 chromosome sizes are not embedded yet. Please provide a chromosome sizes file.")
    return pd.read_table(
        chrom_sizes,
        header=None,
        names=['chrom', 'size']
    ).set_index('chrom')['size'].to_dict()


def read_df_for_chrom(df_path, chrom_name):
    return pd.read_parquet(
        df_path,
        filters=[('chrom', '==', chrom_name)],
        engine='pyarrow'
    )


def merge_and_add_chromosome(results: list[ProcessorOutputData]) -> ProcessorOutputData:
    data = []
    params = []
    categories = [x.identificator for x in results]
    for res in sorted(results, key=lambda x: x.identificator):
        df = res.data_df
        df['chrom'] = pd.Categorical(
            [res.identificator] * df.shape[0],
            categories=categories,
        )
        data.append(df)
        if res.params_df is None:
            continue
        params.append(res.params_df)
        
    data = pd.concat(data, ignore_index=True)
    if len(params) == 0:
        return ProcessorOutputData('all', data)
    params = pd.concat(params, ignore_index=True)
    return ProcessorOutputData('all', data, params)


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def arg_to_list(arg, size):
    if not isinstance(arg, str) and is_iterable(arg):
        assert len(arg) == size, f"Expected {size} elements, got {len(arg)} ({arg})"
        return arg
    return [arg] * size


def df_to_tabix(df: pd.DataFrame, tabix_path):
    """
    Convert a DataFrame to a tabix-indexed file.
    Renames 'chrom' column to '#chr' if exists.

    Parameters:
        - df: DataFrame to convert - bed format. First columns (chr start end).
        - tabix_path: Path to the tabix-indexed file.

    Returns:
        - None
    """
    with pysam.BGZFile(tabix_path, 'w') as bgzip_out:
        with io.TextIOWrapper(bgzip_out, encoding='utf-8') as text_out:
            df.rename(columns={'chrom': '#chr'}).to_csv(text_out, sep='\t', index=False)

    pysam.tabix_index(tabix_path, preset='bed', force=True)


class NoContigPresentError(Exception):
    ...


def ensure_contig_exists(func):
    """
    Decorator for functions that require a contig to be present in the input data.

    Returns None if the contig is not present.
    Otherwise, returns the result of the function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NoContigPresentError:
            return None
    return wrapper


def run_modwt(signal, level=3):
    """
    Wrapper for the modwt command-line tool.
    """
    # Convert the NumPy array of integers to a newline-separated string
    input_data = "\n".join(map(str, signal))

    # Define the command and arguments
    cmd = ['modwt', '--level', f"{level}", '--to-stdout', '--boundary', 'reflected', '--filter', 'haar', '-']

    # Use Popen to stream input data
    process = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=input_data)

    if process.returncode != 0:
        print(f"Error running modwt: {stderr}")
        return None

    return np.array2string(stdout, sep='\n')


import cProfile
import pstats
import numpy as np

# Assuming you have the run_modwt_optimized or the original function ready

def profile_modwt(func, signal, level=3):
    # Create a profiler instance
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # Run the function to profile
    result = func(signal, level)  # Call your function here

    # Stop profiling
    profiler.disable()

    # Create Stats object to print profiling data
    stats = pstats.Stats(profiler).sort_stats('cumtime')  # Sort by cumulative time
    stats.print_stats(10)  # Print top 10 lines of stats

    return result

