"""
Creation:
    Author: Rob Moss
    Date: 2024-07-04
Description:
    Test drawing LHS samples from arbitrary distributions.
"""

import numpy as np
import scipy.stats
import unittest

from metacast.sensitivity_analyses.lhs_and_prcc import lhs_prcc
from setuptests import (
    SetUpOfTests,
    prob_over_many_days_to_prob_on_a_day,
    vaccination_parameters_setup,
)


class TestLHSArbitraryDists(SetUpOfTests):
    """
    Test that we can draw LHS samples from arbitrary distributions.
    """

    def test_sampling_from_betas(self):
        """
        Test drawing samples from several different Beta distributions.
        """
        nu_unvaccinated_lower = prob_over_many_days_to_prob_on_a_day(
            0.5, self.end_day
        )
        nu_unvaccinated_upper = prob_over_many_days_to_prob_on_a_day(
            0.95, self.end_day
        )
        nu_unvaccinated_scale = nu_unvaccinated_upper - nu_unvaccinated_lower
        dist_nu = scipy.stats.beta(
            a=1, b=1, loc=nu_unvaccinated_lower, scale=nu_unvaccinated_scale
        )
        dist_nu_lag = scipy.stats.beta(
            a=100, b=100, loc=1 / 28, scale=1 / 7 - 1 / 28
        )
        dist_l_v = scipy.stats.beta(a=100, b=0.5, loc=0.3, scale=0.3)
        dist_l_h = scipy.stats.beta(a=0.5, b=100, loc=0.7, scale=0.25)
        parameter_samples_dict = {
            'nu_unvaccinated': dist_nu.ppf,
            'nu_vaccination_lag': dist_nu_lag.ppf,
            'l_v': dist_l_v.ppf,
            'h_v': dist_l_h.ppf,
        }

        metapop_model_vaccination, y_vaccination = (
            self.change_to_vaccination_dimensions(include_parameters=False)
        )

        fixed_parameters = {
            'beta': self.beta,
            'rho': self.rho,
            **self.non_subpop_parameters,
            **self.hospitalisation_probs,
        }

        results_df, prccs = lhs_prcc(
            parameters_df=parameter_samples_dict,
            sample_size=10,
            model_run_method=self.vaccination_lhs_sim,
            metapop_model=metapop_model_vaccination,
            t=self.t,
            y0=y_vaccination,
            fixed_parameter=fixed_parameters,
        )

        # Verify that the parameter samples are consistent with the chosen
        # distributions, noting that we only draw 10 samples.

        # Check the mean and bounds for `nu_unvaccinated`.
        nu_unv_mean = results_df['nu_unvaccinated'].mean()
        nu_unv_mean_expected = 0.5 * (
            nu_unvaccinated_lower + nu_unvaccinated_upper
        )
        nu_unv_mean_diff = abs(nu_unv_mean - nu_unv_mean_expected)
        assert nu_unv_mean_diff < 0.001
        assert results_df['nu_unvaccinated'].min() >= nu_unvaccinated_lower
        assert results_df['nu_unvaccinated'].max() <= nu_unvaccinated_upper

        # Check the mean for `nu_vaccination_lag` and ensure that values are
        # concentrated around the mean due to the chosen shape parameters.
        nu_lag_mean = results_df['nu_vaccination_lag'].mean()
        nu_lag_mean_expected = 0.5 * (1 / 28 + 1 / 7)
        nu_lag_mean_diff = abs(nu_lag_mean - nu_lag_mean_expected)
        nu_lag_mean_rel_err = nu_lag_mean_diff / nu_lag_mean_expected
        assert nu_lag_mean_rel_err < 0.01
        nu_lag_var = results_df['nu_vaccination_lag'].var()
        nu_lag_var_expected = dist_nu_lag.var()
        nu_lag_var_diff = abs(nu_lag_var - nu_lag_var_expected)
        nu_lag_var_rel_err = nu_lag_var_diff / nu_lag_var_expected
        assert nu_lag_var_rel_err < 1
        # Variance of uniform is (1/12) * (b - a)^2
        nu_lag_var_uniform = (1 / 12) * (1 / 7 - 1 / 28) ** 2
        assert nu_lag_var < 0.5 * nu_lag_var_uniform

        # Check that the mean for `l_v` is close to the upper bound, due to
        # the chosen shape parameters.
        l_v_mean = results_df['l_v'].mean()
        l_v_rel = 0.9
        l_v_threshold = (1 - l_v_rel) * 0.3 + l_v_rel * 0.6
        assert l_v_mean > l_v_threshold

        # Check that the mean for `h_v` is close to the lower bound, due to
        # the chosen shape parameters.
        h_v_mean = results_df['h_v'].mean()
        h_v_rel = 0.9
        h_v_threshold = (1 - h_v_rel) * 0.7 + l_v_rel * 0.95
        assert h_v_mean < h_v_threshold

    def vaccination_lhs_sim(
        self, sample, t, y0, fixed_parameter, metapop_model
    ):
        parameters = vaccination_parameters_setup(
            **sample,
            other_parameters=fixed_parameter,
            metapop_model=metapop_model,
        )
        metapop_model.parameters = parameters
        results = metapop_model.integrate(y0, t)
        focused_results = {
            'Total hospitalisations': results.loc[
                90, ('observed_states', 'H_cumulative')
            ],
            'Peak hospitalisations': max(
                results.loc[:, ('observed_states', 'H')]
            ),
        }
        results_and_sample = sample | focused_results  #
        return results_and_sample


if __name__ == '__main__':
    unittest.main()
