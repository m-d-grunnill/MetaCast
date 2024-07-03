"""
Creation:
    Author: Martin Grunnill
    Date: 2024-03-19
Description: 
    
"""
from unittest import TestCase
import unittest
from setuptests import SetUpOfTests, vaccination_parameters_setup, prob_over_many_days_to_prob_on_a_day
import pandas as pd
from metacast.sensitivity_analyses.lhs_and_prcc import lhs_prcc
import os
import dask

class TestLHSandPRCC(SetUpOfTests):


    def vaccination_lhs_sim(self,
                            sample,  # First argument must be for a dictionary of the sampled parameters
                            t,
                            y0,
                            fixed_parameter,
                            metapop_model):
        parameters = vaccination_parameters_setup(**sample, other_parameters=fixed_parameter,
                                                  metapop_model=metapop_model)
        metapop_model.parameters = parameters
        results = metapop_model.integrate(y0, t)
        focused_results = {'Total hospitalisations': results.loc[90, ('observed_states', 'H_cumulative')],
                           'Peak hospitalisations': max(results.loc[:, ('observed_states', 'H')])}
        results_and_sample = sample | focused_results  #
        return results_and_sample

    def test_lhs_prcc(self):
        parameter_samples_records = [
            {'parameter': 'nu_unvaccinated',
             'Lower Bound': prob_over_many_days_to_prob_on_a_day(0.5, self.end_day),
             'Upper Bound': prob_over_many_days_to_prob_on_a_day(0.95, self.end_day)},  # Rate of vaccinations.

            {'parameter': 'nu_vaccination_lag', 'Lower Bound': 1 / 28, 'Upper Bound': 1 / 7},
            # Lag between vaccination and vaccination being effective.
            {'parameter': 'l_v', 'Lower Bound': 0.3, 'Upper Bound': 0.6},
            # Reduction in susceptibility to disease X after vaccination.
            {'parameter': 'h_v', 'Lower Bound': 0.7, 'Upper Bound': 0.95}
            # Reduction in hospitalisation with disease X after vaccination.
        ]
        parameter_samples_df = pd.DataFrame.from_records(parameter_samples_records)
        parameter_samples_df.set_index('parameter',
                                       inplace=True)  # The parameters must be set as the index of the dataframe.

        metapop_model_vaccination, y_vaccination = self.change_to_vaccination_dimensions(include_parameters=False)

        fixed_parameters = {'beta': self.beta, 'rho': self.rho, **self.non_subpop_parameters,
                            **self.hospitalisation_probs}

        results_and_sample_df, prccs = lhs_prcc(parameters_df=parameter_samples_df,
                                                sample_size=10,
                                                model_run_method=self.vaccination_lhs_sim,
                                                metapop_model=metapop_model_vaccination,
                                                t=self.t,
                                                y0=y_vaccination,
                                                fixed_parameter=fixed_parameters)
        # Can't seem to get a dask client to work within a unit test??????
        # number_cpu = os.cpu_count()
        # number_cpu
        # client = dask.distributed.Client(n_workers=number_cpu - 1, threads_per_worker=1)
        # client
        # results_and_sample_df, prccs = lhs_prcc(parameters_df=parameter_samples_df,
        #                                         sample_size=1000,
        #                                         model_run_method=self.vaccination_lhs_sim,
        #                                         metapop_model=metapop_model_vaccination,
        #                                         client=client,  # this argument tells lhs_prcc to use this dask client.
        #                                         t=self.t,
        #                                         y0=y_vaccination,
        #                                         fixed_parameter=fixed_parameters)
        # client.close()




if __name__ == '__main__':
    unittest.main()
