import pandas as pd
import dataclasses
import pysam
import io


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
    def wrapper(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NoContigPresentError:
            return None
    return wrapper