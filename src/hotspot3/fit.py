import numpy.ma as ma
import numpy as np
from scipy import stats as st
from hotspot3.models import NoContigPresentError, ProcessorConfig, FitResults
from hotspot3.utils import wrap_masked
import bottleneck as bn
from scipy.special import betainc


class BackgroundFit:
    def __init__(self, config: ProcessorConfig=None):
        if config is None:
            config = ProcessorConfig()
        self.config = config

    def fit(self) -> FitResults:
        raise NotImplementedError

    @wrap_masked
    def p_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.array(1 - mean / var, dtype=np.float32)
        return p
    
    @wrap_masked
    def r_from_mean_and_var(self, mean: np.ndarray, var: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.array(mean ** 2 / (var - mean), dtype=np.float32)
        return r


class GlobalBackgroundFit(BackgroundFit):
    def fit(self, array: ma.MaskedArray) -> FitResults:
        agg_cutcounts = ma.masked_invalid(array).compressed()

        max_cutoff = np.quantile(agg_cutcounts, 0.90)
        validation_set = agg_cutcounts[agg_cutcounts <= max_cutoff]
        unique, counts = np.unique(validation_set, return_counts=True)

        tr = np.quantile(agg_cutcounts, self.config.signal_quantile)
        p, r = self.fit_for_tr(agg_cutcounts, tr)
        rmsea = self.calc_rmsea_for_tr(counts, unique, r, p, tr)

        while rmsea > 0.05 or tr > max_cutoff:
            tr -= 1
            p, r = self.fit_for_tr(agg_cutcounts, tr)
            rmsea = self.calc_rmsea_for_tr(counts, unique, r, p, tr)
        
        quantile = np.sum(np.sort(agg_cutcounts) <= tr) / agg_cutcounts.shape[0]

        return FitResults(p.squeeze(), r.squeeze(), rmsea.squeeze(), quantile)

    def fit_for_tr(self, agg_cutcounts, tr):
        agg_cutcounts = agg_cutcounts[agg_cutcounts <= tr]
        mean, var = self.estimate_global_mean_and_var(agg_cutcounts)

        p = self.p_from_mean_and_var(mean, var)
        r = self.r_from_mean_and_var(mean, var)
        return p, r


    def estimate_global_mean_and_var(self, agg_cutcounts: np.ndarray):
        has_enough_background = np.count_nonzero(agg_cutcounts) / agg_cutcounts.size > self.config.nonzero_windows_to_fit
        if not has_enough_background:
            raise NoContigPresentError
        
        mean = np.mean(agg_cutcounts)
        variance = np.var(agg_cutcounts, ddof=1)
        return mean, variance

    def calc_rmsea_for_tr(self, obs, unique_cutcounts, r, p, tr):
        N = sum(obs)
        exp = st.nbinom.pmf(unique_cutcounts, r, 1 - p) / st.nbinom.cdf(tr - 1, r, 1 - p) * N
        # chisq = sum((obs - exp) ** 2 / exp)
        G_sq = 2 * sum(obs * np.log(obs / exp))
        df = len(obs) - 2
        return np.sqrt(np.maximum(G_sq / df - 1, 0) / (N - 1))


class WindowBackgroundFit(BackgroundFit):
    def fit(self, array: ma.MaskedArray) -> FitResults:
        agg_cutcounts = array.copy()
        stratified_cutcounts = self.stratify(agg_cutcounts, 75)
        sliding_ranks = self.centered_running_rank(stratified_cutcounts, self.config.bg_window, min_count=1)
        sliding_ranks = self.interpolate_nan(sliding_ranks, 75)

        high_signal_mask = (sliding_ranks > self.config.signal_quantile).filled(False)
        agg_cutcounts[high_signal_mask] = np.nan
        mean, var = self.sliding_mean_and_variance(agg_cutcounts, min_count=self.config.min_mappable_bg)
        p = self.p_from_mean_and_var(mean, var)
        success_fit_mask = ~p.mask
        p = p.filled(np.nan).astype(np.float16)
        r = self.r_from_mean_and_var(mean, var).filled(np.nan).astype(np.float32)

        bad_fit = ma.where(mean >= var)[0]
        if len(bad_fit) > 0:
            p[bad_fit] = 0
            r[bad_fit] = np.inf

            bad_fit_params = np.empty((len(bad_fit), 3), dtype=np.float64)
            bad_fit_params[:, 0] = bad_fit
            bad_fit_params[:, 1] = mean[bad_fit]
            bad_fit_params[:, 2] = var[bad_fit]
        else:
            bad_fit_params = None
        
        del mean, var
  
        rmsea = np.full_like(p, np.nan)
        return FitResults(
            p, r, rmsea, np.nan, 
            successful_fit_mask=success_fit_mask,
            bad_fit_params=bad_fit_params
        )

    def sliding_mean_and_variance(self, array: ma.MaskedArray, min_count):
        window = self.config.bg_window

        mean = self.centered_running_nanmean(array, window, min_count=min_count)
        var = self.centered_running_nanvar(array, window, min_count=min_count)
        mean = ma.masked_invalid(mean)
        var = ma.masked_invalid(var)
        return mean, var

    @wrap_masked
    def move_sum_with_dtype(self, array, window, min_count):
        array = np.asarray(array, np.float32)
        return bn.move_sum(array, window, min_count=min_count).astype(np.float32)
    
    @wrap_masked
    def move_rank_with_dtype(self, array, window, min_count):
        array = np.asarray(array, np.float32)
        return (bn.move_rank(array, window, min_count=min_count).astype(np.float32) + 1.) / 2.
    
    @wrap_masked
    def centered_running_nanvar(self, array, window, min_count):
        return correct_offset(bn.move_var, array, window, ddof=1, min_count=min_count).astype(np.float32)

    @wrap_masked
    def centered_running_nanmean(self, array, window, min_count):
        return correct_offset(bn.move_mean, array, window, min_count=min_count).astype(np.float32)
    
    @wrap_masked
    def centered_running_nansum(self, array: np.ndarray, window, min_count):
        return correct_offset(self.move_sum_with_dtype, array, window, min_count=min_count)
    
    @wrap_masked
    def stratify(self, array, step):
        res = np.full_like(array, np.nan)
        res[::step] = array[::step]
        return res
    
    @wrap_masked
    def interpolate_nan(self, arr, step):
        indices = np.arange(0, len(arr), step, dtype=np.uint32)
        subsampled_arr = arr[::step].copy()
        
        nan_mask = np.isnan(subsampled_arr)
        if nan_mask.any():
            valid_indices = np.where(~nan_mask)[0]
            nan_indices = np.where(nan_mask)[0]
            subsampled_arr[nan_mask] = np.interp(
                nan_indices, valid_indices, subsampled_arr[valid_indices]
            )

        return np.interp(
            np.arange(len(arr), dtype=np.uint32),
            indices,
            subsampled_arr,
            left=None,
            right=None,
        )

    @wrap_masked
    def centered_running_rank(self, array, window, min_count):
        """
        Calculate the rank of the center value in a moving window.
        Ignore NaN values.
        """
        assert window % 2 == 1, "Window size should be odd"
        half_window = window // 2 + 1
        not_nan = (~np.isnan(array)).astype(np.float32)

        trailing_ranks = self.move_rank_with_dtype(array, half_window, min_count=min_count)
        trailing_ranks *= self.move_sum_with_dtype(not_nan, half_window, min_count) - 1

        leading_ranks = self.move_rank_with_dtype(array[::-1], half_window, min_count=min_count)[::-1]
        leading_ranks *= self.move_sum_with_dtype(not_nan[::-1], half_window, min_count)[::-1] - 1
        
        combined_ranks = trailing_ranks + leading_ranks
        denom = self.centered_running_nansum(not_nan, window, min_count=min_count) - 1
        denom[denom == 0] = 1.
        combined_ranks /= denom
        return combined_ranks

    @wrap_masked
    def calc_rmsea_all_windows(
        self,
        aggregated_cutcounts,
        r, p,
        bin_edges, # left edge inclusive, must be unique
    ):
        """
        Calculate RMSEA for all sliding windows.
        """
        # assert (bg_window - 1) % step == 0, "Stride should be a divisor of the window size."
        bg_sum_mappable = self.centered_running_nansum(~np.isnan(aggregated_cutcounts), self.config.bg_window)

        G_sq = np.zeros_like(aggregated_cutcounts)
        prev_cdf = betainc(r, bin_edges[0], 1 - p, dtype=np.float32)
        norm_coef = betainc(r, bin_edges[-1], 1 - p, dtype=np.float32) - prev_cdf
        for i in range(len(bin_edges) - 1):
            left = bin_edges[i]
            right = bin_edges[i + 1]

            observed = self.centered_running_nansum(
                (aggregated_cutcounts >= left) & (aggregated_cutcounts < right),
                self.config.bg_window
            )

            if i == len(bin_edges) - 2:
                new_cdf = norm_coef
            else:
                new_cdf = betainc(r, right, 1 - p, dtype=np.float32)
            expected = (new_cdf - prev_cdf) * bg_sum_mappable / norm_coef
            prev_cdf = new_cdf

            G_sq += calc_g_sq(
                observed, expected
            )

        df = len(bin_edges) - 1 - 2 # number of bins - number of parameters
        return np.sqrt(np.clip(G_sq / df - 1, 0, None) / (bg_sum_mappable - 1))


def correct_offset(function, array, window, *args, **kwargs):
    """
    Correct offset of a trailing running window to make it centered.
    """
    assert window % 2 == 1, "Window size should be odd"
    offset = window // 2
    result = function(
        np.pad(array, (0, offset), mode='constant', constant_values=np.nan),
        window,
        *args,
        **kwargs
    )
    return result[offset:]


def calc_g_sq(obs, exp):
    ratio = obs / exp
    ratio[ratio == 0] = 1
    return obs * np.log(ratio) * 2
