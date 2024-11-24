import pandas as pd
import numpy as np

from hotspot3.io.logging import WithLogger
from hotspot3.helpers.models import SPOTEstimationResults, SPOTEstimationData
from hotspot3.helpers.stats import upper_bg_quantile, weighted_median
from hotspot3.helpers.format_converters import get_spot_score_fit_data


class SignalToNoiseFit(WithLogger):

    def fit(self, fit_data: pd.DataFrame):
        """
        Fit the signal to the noise using a linear regression.
        """
        fit_data['background'] = upper_bg_quantile(
            fit_data['bg_r'],
            fit_data['bg_p']
        )
        is_segment_fit = fit_data['fit_type'] == 'segment'

        spot_data = get_spot_score_fit_data(fit_data[is_segment_fit])

        spot_results = self.spot_score_and_outliers(spot_data)

        fit_data.loc[is_segment_fit, 'outlier_distance'] = spot_results.outlier_distance
        fit_data.loc[is_segment_fit, 'is_inlier'] = spot_results.inliers_mask
        fit_data['SPOT'] = spot_results.spot_score
        fit_data['SPOT_std'] = spot_results.spot_score_std
        fit_data['segment_SPOT'] = spot_results.segment_spots
        return fit_data, spot_results
    
    def calc_outlier_distance(self, total_tags, total_tags_background, spot):
        return total_tags / total_tags_background * (1 - spot)
    
    def spot_score_and_outliers(self, spot_data: SPOTEstimationData):
        total_tags = spot_data.total_tags
        total_tags_background = spot_data.total_tags_background
        weight = spot_data.weight

        segment_spots = total_tags_background / total_tags
        spot_score = weighted_median(segment_spots, weight)
        spot_score_std = np.sqrt(np.sum((segment_spots - spot_score) ** 2 * weight) / np.sum(weight))

        outlier_distance = self.calc_outlier_distance(
            total_tags,
            total_tags_background,
            spot_score
        )
        
        inliers_mask = outlier_distance < self.config.outlier_segment_threshold
        self.logger.info(f"Signal to noise fit results: SPOT={spot_score:.2f}, SPOT_std={spot_score_std:.2f}")
        return SPOTEstimationResults(
            spots=segment_spots, spot_score=spot_score,spot_score_std=spot_score_std, inliers_mask=inliers_mask, outlier_distance=outlier_distance, segment_spots=segment_spots
        )

    def find_outliers(self, fit_data: pd.DataFrame) -> np.ndarray:
        return fit_data.eval(f'outlier_distance >= {self.config.outlier_segment_threshold} & fit_type == "segment"').values
