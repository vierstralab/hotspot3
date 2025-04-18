import pandas as pd
import numpy as np

from hotspot3.helpers.models import FitResults, WindowedFitResults, SPOTEstimationData
from hotspot3.helpers.colors import get_bb_color
from hotspot3.helpers.stats import mean_from_r_p


bed12_columns = ['chrom', 'start', 'end', 'id', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']


def fit_stats_df_to_fallback_fit_results(fit_result_df: pd.DataFrame):
    assert len(fit_result_df) == 1, f"Expected one fit, got {len(fit_result_df)}"
    segment_fit_params = fit_result_df[
        ['bg_p', 'bg_r', 'rmsea', 'bg_bases_prop', 'bg_tr']
    ].iloc[0].rename(
        {
            'bg_p': 'p',
            'bg_r': 'r',
            'bg_bases_prop': 'fit_quantile',
            'bg_tr': 'fit_threshold'
        }
    ).to_dict()

    return FitResults(**segment_fit_params)


def get_spot_score_fit_data(fit_data: pd.DataFrame):
    total_tags_background = fit_data['bg_tags'].values
    total_tags = fit_data['mappable_tags'].values
    total_bases = fit_data['mappable_bases'].values

    spot_scores = np.clip(1 - total_tags_background / total_tags, 0, 1)
    valid_scores = np.isfinite(spot_scores) & (spot_scores > 0) & (spot_scores < 1)
    
    return SPOTEstimationData(
        total_tags=total_tags,
        total_tags_background=total_tags_background,
        segment_spot_scores=spot_scores,
        total_bases=total_bases,
        valid_scores=valid_scores
    )

def convert_fit_results_to_series(
        fit_results: FitResults,
        fit_type: str,
        success_fit: bool,
        reached_max_q: bool=False
    ) -> pd.Series:
    return pd.Series({
        'bg_r': fit_results.r,
        'bg_p': fit_results.p,
        'rmsea': fit_results.rmsea,
        'bg_tr': fit_results.fit_threshold,
        'mappable_bases': fit_results.n_total,
        'bg_bases': fit_results.n_total - fit_results.n_signal,
        'bg_bases_prop': fit_results.fit_quantile,
        'mappable_tags': fit_results.total_tags,
        'bg_tags': fit_results.total_tags - fit_results.signal_tags,
        'fit_type': fit_type,
        'success_fit': success_fit,
        'max_bg_reached': reached_max_q
    })

def set_dtype(intervals_stats: pd.DataFrame, fit_series: pd.Series):
    """
    Workaround func to avoid future warnings about setting bool values to float (default) columns
    Initially sets value of first row to every row in the DataFrame
    """
    added_cols = ~fit_series.index.isin(intervals_stats.columns)
    if np.any(added_cols):
        for col in fit_series.index[added_cols]:
            intervals_stats[col] = fit_series[col]


def set_series_row_to_df(df: pd.DataFrame, row: pd.Series, index):
    set_dtype(df, row)
    df.loc[index, row.index] = row
    return df


def fit_results_to_df(fit_results: WindowedFitResults, per_window_trs: np.ndarray):
    return pd.DataFrame({
        'sliding_r': fit_results.r,
        'sliding_p': fit_results.p,
        'enough_bg': fit_results.enough_bg_mask,
        'tr': per_window_trs,
    })

def peaks_to_bed12(peaks_df, fdr_tr):
    """
    Convert peaks to bed9 format.
    """
    peaks_df['strand'] = '.'
    peaks_df['score'] = convert_to_score(peaks_df['max_density'], 100)
    peaks_df['thickStart'] = peaks_df['summit']
    peaks_df['thickEnd'] = peaks_df['summit'] + 1
    peaks_df['itemRgb'] = get_bb_color(fdr_tr, mode='peaks')

    peaks_df['blockCount'] = 3
    peaks_df['blockSizes'] = '1,1,1'
    peaks_df['blockStarts'] = '0,' + peaks_df.eval('summit - start').astype(str) + ',' + peaks_df.eval('end - start - 1').astype(str)

    return peaks_df[bed12_columns]


def hotspots_to_bed12(hotspots_df, fdr_tr, significant_stretches):
    """
    Convert hotspots to bed9 format.
    """
    hotspots_df['strand'] = '.'
    hotspots_df['score'] = convert_to_score(hotspots_df['max_neglog10_fdr'], 10)
    hotspots_df['thickStart'] = hotspots_df['start']
    hotspots_df['thickEnd'] = hotspots_df['end']
    hotspots_df['itemRgb'] = get_bb_color(fdr_tr, mode='hotspots')
    block_count = []
    block_sizes = []
    block_starts = []
    lengths = hotspots_df.eval('end - start').values
    for i, (starts, ends) in enumerate(significant_stretches):
        block_count.append(len(starts) + 2)

        sizes = np.pad(ends - starts, (1, 1), mode='constant', constant_values=(1, 1))
        block_sizes.append(','.join(map(str, sizes)))

        starts = np.pad(starts, (1, 1), mode='constant', constant_values=(0, lengths[i] - 1))
        block_starts.append(','.join(map(str, starts)))

    hotspots_df['blockCount'] = block_count
    hotspots_df['blockSizes'] = block_sizes
    hotspots_df['blockStarts'] = block_starts

    return hotspots_df[bed12_columns]


def convert_to_score(array, mult, max_score=1000):
    return np.round(array * mult).astype(np.int64).clip(0, max_score)
