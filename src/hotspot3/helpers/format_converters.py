import pandas as pd

from hotspot3.helpers.models import FitResults
from hotspot3.helpers.colors import get_bb_color
from hotspot3.helpers.stats import mean_from_r_p
import numpy as np

bed12_columns = ['chrom', 'start', 'end', 'id', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']


def fit_stats_df_to_fallback_fit_results(df: pd.DataFrame):
    chrom_fit = df.query(f'fit_type == "global"')
    assert len(chrom_fit) == 1, f"Expected one global fit, got {len(chrom_fit)}"
    chrom_fit = chrom_fit[
        ['p_bg', 'r_bg', 'rmsea', 'quantile_tr', 'signal_tr']
    ].iloc[0].rename(
        {
            'p_bg': 'p',
            'r_bg': 'r',
            'quantile_tr': 'fit_quantile',
            'signal_tr': 'fit_threshold'
        }
    ).to_dict()

    return FitResults(**chrom_fit)


def convert_fit_results_to_series(
        fit_results: FitResults,
        mean: float,
        fit_type: str,
        success_fit: bool
    ) -> pd.Series:
    return pd.Series({
        'r_bg': fit_results.r,
        'p_bg': fit_results.p,
        'rmsea': fit_results.rmsea,
        'signal_tr': fit_results.fit_threshold,
        'quantile_tr': fit_results.fit_quantile,
        'bases_total': fit_results.n_total,
        'bases_bg': fit_results.n_total - fit_results.n_signal,
        'mean_total': mean,
        'mean_bg': mean_from_r_p(fit_results.r, fit_results.p),
        'fit_type': fit_type,
        'success_fit': success_fit
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


def hotspots_to_bed12(self, hotspots_df, fdr_tr, significant_stretches):
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
