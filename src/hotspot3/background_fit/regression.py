import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.special import logit, expit

from hotspot3.io.logging import WithLogger
from hotspot3.helpers.models import SPOTEstimationResults, SPOTEstimationData
from hotspot3.helpers.stats import upper_bg_quantile, weighted_median, weighted_std
from hotspot3.helpers.format_converters import get_spot_score_fit_data
from hotspot3.helpers.stats import roundup_float


class SignalToNoiseFit(WithLogger):

    def fit(self, fit_data: pd.DataFrame):
        """
        Fit the signal to the noise using a linear regression.
        """
        fit_data['background'] = upper_bg_quantile(
            fit_data['bg_r'],
            fit_data['bg_p']
        )
        valid_segment_fit = fit_data.eval('fit_type == "segment" & success_fit')

        spot_data = get_spot_score_fit_data(fit_data[valid_segment_fit])
        fit_data.loc[valid_segment_fit, 'segment_SPOT'] = spot_data.segment_spot_scores

        spot_results = self.fit_spot_length_regression(spot_data)

        # One value per segment
        fit_data.loc[valid_segment_fit, 'outlier_distance'] = spot_results.outlier_distance
        fit_data.loc[valid_segment_fit, 'is_inlier'] = spot_results.inliers_mask
        fit_data.loc[valid_segment_fit, 'min_bg_tag_proportion'] = spot_results.min_bg_tags_fraction
        
        # One value for the dataset
        fit_data['slope'] = spot_results.slope
        fit_data['intercept'] = spot_results.intercept
        fit_data['SPOT'] = spot_results.spot_score
        fit_data['SPOT_std'] = spot_results.spot_score_std

        return fit_data, spot_results
    
    def calc_dataset_spot_score(self, spot_data: SPOTEstimationData):
        spot_scores = spot_data.segment_spot_scores[spot_data.valid_scores]
        total_bases = spot_data.total_bases[spot_data.valid_scores]
        spot_score = weighted_median(spot_scores, total_bases)
        return spot_score
    
    def calc_spot_score_error(self, resid, spot_score, total_bases):
        detrended_segment_spot = expit(resid + logit(spot_score))
        return weighted_std(detrended_segment_spot, total_bases, spot_score)
    
    def fit_spot_length_regression(self, spot_data: SPOTEstimationData):
        model = RANSACRegressor(random_state=42, min_samples=0.2)
        spot_score = self.calc_dataset_spot_score(spot_data)

        X = np.log(spot_data.total_bases)[:, None]
        y = logit(spot_data.segment_spot_scores)

        where = spot_data.valid_scores
        model.fit(X[where], y[where])
        inliers = np.zeros_like(where, dtype=bool)
        inliers[where] = model.inlier_mask_
        slope, intercept = model.estimator_.coef_[0], model.estimator_.intercept_
        
        y_pred = model.predict(X)
        resid = y - y_pred
        min_bg_tags_fraction = roundup_float(
            expit(
                -y_pred - np.log(self.config.outlier_segment_threshold)
            ), 
            3
        ) # Ceil to 3th decimal place to avoid precision issues
        outlier_dist = np.exp(resid)

        spot_score_std = self.calc_spot_score_error(resid, spot_score, spot_data.total_bases)

        self.logger.info(f"Signal to noise fit: SPOT={spot_score:.2f}±{spot_score_std:.2f}")

        return SPOTEstimationResults(
            spot_score=spot_score,
            spot_score_std=spot_score_std,
            inliers_mask=inliers,
            slope=slope,
            intercept=intercept,
            outlier_distance=outlier_dist,
            min_bg_tags_fraction=min_bg_tags_fraction
        )

    def find_outliers(self, fit_data: pd.DataFrame) -> np.ndarray:
        return fit_data.eval(f'outlier_distance >= {self.config.outlier_segment_threshold} & fit_type == "segment"').values
