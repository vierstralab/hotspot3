import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score
import numpy as np

from hotspot3.io.logging import WithLogger
from hotspot3.models import RegressionResults
from hotspot3.stats import upper_bg_quantile


class SignalToNoiseFit(WithLogger):

    def fit(self, fit_data: pd.DataFrame):
        """
        Fit the signal to the noise using a linear regression.
        """
        fit_data['background'] = upper_bg_quantile(
            fit_data['r'],
            fit_data['p']
        )
        is_segment_fit = fit_data['fit_type'] == 'segment'

        bg_mean = fit_data[is_segment_fit].eval('log(r * p / (1 - p))').values[:, None]
        mean = fit_data[is_segment_fit].eval('log(mean)').values
        length = fit_data[is_segment_fit].eval('end - start').values


        regression_results = self._fit(bg_mean, mean, length)

        fit_data.loc[is_segment_fit, 'outlier_distance'] = regression_results.outlier_distance
        fit_data.loc[is_segment_fit, 'is_inlier'] = regression_results.inliers_mask
        fit_data['fit_r2'] = regression_results.r2
        fit_data['slope'] = regression_results.slope
        fit_data['intercept'] = regression_results.intercept

        return fit_data
    
    def _fit(self, x, y_true, sample_weight):
        model = RANSACRegressor(random_state=0)
        model.fit(x, y_true, sample_weight=sample_weight)

        y_pred = model.predict(x)
        inliers_mask = model.inlier_mask_
        r2 = r2_score(
            y_true[inliers_mask],
            y_pred[inliers_mask],
            sample_weight=sample_weight[inliers_mask]
        )
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_

        residuals = y_true - y_pred

        inlier_std = np.sqrt(
            np.cov(
                residuals[inliers_mask],
                aweights=sample_weight[inliers_mask]
            )
        )
        outlier_distance = y_true - y_pred / inlier_std

        return RegressionResults(slope, intercept, r2, inliers_mask, outlier_distance)
