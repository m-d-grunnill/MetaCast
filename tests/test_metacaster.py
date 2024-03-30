"""
Creation:
    Author: Martin Grunnill
    Date: 2024-03-19
Description: 
    
"""
from setuptests import SetUpOfTests
import unittest
import numpy as np

class TestMetaCaster(SetUpOfTests):
    def test_calculate_fois(self):
        fois = self.metapop_model.calculate_fois(y=self.y_low_high, parameters=self.parameters_low_high, t=0)
        self.assertEqual(fois['low'], fois['high'])
        self.assertAlmostEqual(fois['low'], 0.0020037142857142858)

    def test_dimensions(self):
        isolation_groups = ['negative', 'positive']
        rapid_pathogen_test_dimensions = [self.risk_groups, isolation_groups]
        self.metapop_model.dimensions = rapid_pathogen_test_dimensions
        self.assertEqual(self.metapop_model.dimensions, [{'high', 'low'}, {'negative', 'positive'}])
        self.metapop_model.dimensions = self.vaccination_dimensions
        self.assertEqual(self.metapop_model.dimensions,
                         [{'high', 'low'}, {'unvaccinated', 'vaccinated', 'vaccination_lag'}])

    def test_subpop_transfer(self):
        metapop_model_vaccination, y_vaccination, parameters_vaccination = self.change_to_vaccination_dimensions()
        y_deltas = np.zeros(metapop_model_vaccination.total_states)
        y_deltas = metapop_model_vaccination.subpop_transfer(y=y_vaccination,
                                                             y_deltas=y_deltas,
                                                             t=0,
                                                             from_coordinates=('high', 'unvaccinated'),
                                                             parameters=parameters_vaccination)
        self.assertAlmostEqual(y_deltas.sum(), 0)
        for state, from_index in metapop_model_vaccination.state_index[('high', 'unvaccinated')].items():
            from_y_delta = y_deltas[from_index]
            self.assertLessEqual(from_y_delta, 0)
            to_index = metapop_model_vaccination.state_index[('high', 'vaccination_lag')][state]
            to_y_delta = y_deltas[to_index]
            self.assertEqual(from_y_delta, -to_y_delta)

    def test_ode(self):
        parameters_tuple = tuple(self.parameters_low_high.values())
        y_deltas = self.metapop_model.ode(self.y_low_high, 0, *parameters_tuple)
        self.assertAlmostEqual(y_deltas[-2], 11.56857, 3)
        metapop_model_vaccination, y_vaccination, parameters_vaccination = self.change_to_vaccination_dimensions()
        parameters_vaccination_tuple = tuple(parameters_vaccination.values())
        y_deltas_vaccination = metapop_model_vaccination.ode(y_vaccination, 0, *parameters_vaccination_tuple)
        self.assertAlmostEqual(y_deltas_vaccination[-2], 11.56857, 3)
        self.assertEqual(y_deltas[-2], y_deltas_vaccination[-2])

    def test_integrate(self):
        self.metapop_model.parameters = self.parameters_low_high
        results_low_high = self.metapop_model.integrate(self.y_low_high, self.t)
        self.assertAlmostEqual(results_low_high.loc[90,('observed_states','H_cumulative')], 29297.017609, 3)
        self.assertAlmostEqual(max(results_low_high.loc[:, ('observed_states', 'H')]), 2966.336876, 3)
        metapop_model_vaccination, y_vaccination, parameters_vaccination = self.change_to_vaccination_dimensions()
        metapop_model_vaccination.parameters = parameters_vaccination
        results_vaccination = metapop_model_vaccination.integrate(y_vaccination, self.t)
        self.assertAlmostEqual(results_vaccination.loc[90, ('observed_states', 'H_cumulative')], 20895.090581, 3)
        self.assertAlmostEqual(max(results_vaccination.loc[:, ('observed_states', 'H')]), 2020.866274, 3)

    def test_get_state_index_dict_of_coordinate(self):
        high_state_dicts = self.metapop_model.get_state_index_dict_of_coordinate('high')
        self.assertEqual(len(high_state_dicts), 1)
        high_state_dict = high_state_dicts['high']
        low_state_dicts = self.metapop_model.get_state_index_dict_of_coordinate('low')
        self.assertEqual(len(low_state_dicts), 1)
        low_state_dict = low_state_dicts['low']
        for low_state_key, low_state_index in low_state_dict.items():
            self.assertNotIn(low_state_index, list(high_state_dict.values()))
            self.assertIn(low_state_key, self.metapop_model.states)

        metapop_model_vaccination, y_vaccination, parameters_vaccination = self.change_to_vaccination_dimensions()
        high_vaccine_state_dicts = metapop_model_vaccination.get_state_index_dict_of_coordinate('high')
        self.assertEqual(len(high_vaccine_state_dicts), 3)
        for key in high_vaccine_state_dicts.keys():
            self.assertEqual(key[0], 'high')

    def test_coordinates_to_subpop_suffix(self):
        self.assertEqual('_[high]',self.metapop_model.coordinates_to_subpop_suffix('high'))
        self.assertEqual('_[high,unvaccinated]',
                         self.metapop_model.coordinates_to_subpop_suffix(['high', 'unvaccinated']))

    def test_shape(self):
        self.assertEqual(self.metapop_model.shape, (2,))
        self.metapop_model.dimensions = [2, 5]
        self.assertEqual(self.metapop_model.shape, (2, 5))
        metapop_model_vaccination, y_vaccination, parameters_vaccination = self.change_to_vaccination_dimensions()
        self.assertEqual(metapop_model_vaccination.shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
