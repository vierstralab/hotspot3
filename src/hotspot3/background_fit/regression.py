import pandas as pd
from scipy.stats import pearsonr
import numpy as np

from hotspot3.io.logging import WithLogger
from hotspot3.models import SPOTEstimationResults
from hotspot3.stats import upper_bg_quantile, weighted_median


class SignalToNoiseFit(WithLogger):

    def fit(self, fit_data: pd.DataFrame):
        """
        Fit the signal to the noise using a linear regression.
        """
        fit_data['background'] = upper_bg_quantile(
            fit_data['r_bg'],
            fit_data['p_bg']
        )
        is_segment_fit = fit_data['fit_type'] == 'segment'

        total_tags_background = fit_data[is_segment_fit].eval(
            'mean_bg * bases_bg'
        )
        total_tags = fit_data[is_segment_fit].eval('mean_total * bases_total').values
        length = fit_data[is_segment_fit].eval('end - start').values

        spot_results = self.spot_score_and_outliers(
            total_tags,
            total_tags_background,
            length,
        )

        fit_data.loc[is_segment_fit, 'outlier_distance'] = spot_results.outlier_distance
        fit_data.loc[is_segment_fit, 'is_inlier'] = spot_results.inliers_mask
        fit_data['fit_r2'] = spot_results.r2
        fit_data['fit_r2_total'] = spot_results.r2_total
        fit_data['SPOT'] = spot_results.spot_score
        return fit_data, spot_results
    
    def calc_spot_score(self, total_tags, total_tags_background, length):
        return 1 - weighted_median(total_tags_background / total_tags, length)
    
    def calc_outlier_distance(self, total_tags, total_tags_background, spot):
        return total_tags / total_tags_background * (1 - spot)
    
    def spot_score_and_outliers(self, total_tags, total_tags_background, length):
        spot_score = self.calc_spot_score(
            total_tags,
            total_tags_background,
            length
        )
        outlier_distance = self.calc_outlier_distance(
            total_tags,
            total_tags_background,
            spot_score
        )
        inliers_mask = outlier_distance < self.config.outlier_segment_threshold
        b = -np.log(1 - spot_score)
        r2 = pearsonr(
            np.log(total_tags[inliers_mask]),
            np.log(total_tags_background[inliers_mask]) + b,
        )[0] ** 2
        r2_total = pearsonr(
            np.log(total_tags),
            np.log(total_tags_background) + b,
        )[0] ** 2
        return SPOTEstimationResults(spot_score, r2, r2_total, inliers_mask, outlier_distance)

    def find_outliers(self, fit_data: pd.DataFrame) -> np.ndarray:
        return fit_data.eval(f'outlier_distance >= {self.config.outlier_segment_threshold} & fit_type == "segment"').values
