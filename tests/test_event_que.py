"""
Creation:
    Author: Martin Grunnill
    Date: 2024-03-19
Description: 
    
"""
from unittest import TestCase
from src.metacast import MetaCaster
from src.metacast.event_handling import EventQueue, TransferEvent
import numpy as np
import copy
import unittest

def subpop_model(y, y_deltas, parameters, states_index, subpop_suffix, foi):
    """
    Calculate derivatives of variables in disease X's subpopulation model

    Parameters
    ----------
    y : numpy.Array
        An array of the state variables at this time point.
    y_deltas : numpy.Array
        The derivatives of y at this time. MetaCaster gives y_delta as a numpy array of zeros to which this function adds the derivatives to.
    parameters : dict {str: Number or callable}
        A dictionary of parameter values or callables used to calculate parameter values at this time point.
    subpop_suffix : string
        This string is of the form '_['coodinate_1,coordinate_2,....']' and is appended to a string denoting parameter specifcally applied to this subpopulation. Alternatively a coordinates argument can be given.
    states_index : dict {str:int}
        This dictionary is used to look up the indexes on y and y_delta for states is this subpopulation.
    foi : float
        Force of infection (lambda) experienced be susceptible hosts in this subpopulation. Note the term lambda could not be used as it is used for lambda functions within python. Therefore, the term foi (Force Of Infection) is used.

    Returns
    -------
    y_deltas : numpy.Array
        Derivatives of variables in disease X's subpopulation model.

    """
    infections = foi * y[states_index['S']]
    progression_from_exposed = parameters['sigma'] * y[states_index['E']]
    probability_of_hospitalisation = parameters['p' + subpop_suffix]  # this is our subpopulation specific parameter
    progression_from_infectious = y[states_index['I']] * parameters['gamma']
    recovery = progression_from_infectious * (1 - probability_of_hospitalisation)
    hospitalisation = progression_from_infectious * probability_of_hospitalisation
    hospital_recovery = y[states_index['H']] * parameters['eta']

    # Updating y_deltas with derivative calculations from this subpopulation.
    y_deltas[states_index['S']] += - infections
    y_deltas[states_index['E']] += infections - progression_from_exposed
    y_deltas[states_index['I']] += progression_from_exposed - progression_from_infectious
    y_deltas[states_index['H']] += hospitalisation - hospital_recovery
    y_deltas[states_index['R']] += recovery + hospital_recovery
    y_deltas[
        -2] += hospitalisation - hospital_recovery  # The last few elements of y_delta can be used for observed states such Total hospital incidence.
    y_deltas[-1] += hospitalisation  # or Total hospitalisations.

    return y_deltas


class TestEventQueue(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.subpop_model = subpop_model

        #### Setting up 'high' 'low' risk group metapopulation
        cls.risk_groups = ['low', 'high']
        cls.isolation_groups = ['negative', 'positive']
        cls.rapid_pathogen_test_dimensions = [cls.risk_groups, cls.isolation_groups]
        cls._metapop_model = MetaCaster(dimensions=cls.rapid_pathogen_test_dimensions,
                                        subpop_model=cls.subpop_model,
                                        states=['S', 'E', 'I', 'H', 'R'],  # States of our model
                                        infected_states=['E', 'I', 'H'],
                                        # Infected states of our model, this is different from infectious states.
                                        infectious_states='I',
                                        # Infectious states of our model. These will be involved in force of
                                        # infection calculations.
                                        symptomatic_states=['I', 'H'],  # List symptomatic states
                                        observed_states=['H', 'H_cumulative'],
                                        # observed_states is where we name what is being tracked in the last few
                                        # elements of y_deltas.
                                        universal_params=['sigma', 'gamma', 'eta'],
                                        # These are parameters that are not specific to subpopulations.
                                        subpop_params=['p']
                                        )

        # Setup time points
        cls.end_day = 90
        cls.time_step = 1
        cls.t = np.arange(0, cls.end_day + cls.time_step, cls.time_step)

        # Seting up high and low risk group prevalence's
        cls.N = 1e6
        cls.proportion_high_risk = 0.1
        cls.low_risk_population = cls.N * (1 - cls.proportion_high_risk)
        cls.high_risk_population = cls.N * cls.proportion_high_risk
        cls.prevelance = 0.01
        cls.low_risk_infected = {'E': 2615, 'I': 6348, 'H': 37}
        cls.high_risk_infected = {'E': 242, 'I': 665, 'H': 93}

        # Setting up y for low high risk groups and isolation metapopulation
        cls.y_isolation = np.zeros(cls._metapop_model.total_states)
        low_risk_state_pops = {'S': cls.low_risk_population * (1 - cls.prevelance), **cls.low_risk_infected}
        for state, index in cls._metapop_model.state_index[('low', 'negative')].items():
            if state in low_risk_state_pops:
                cls.y_isolation[index] = low_risk_state_pops[state]
        high_risk_state_pops = {'S': cls.high_risk_population * (1 - cls.prevelance), **cls.high_risk_infected}
        for state, index in cls._metapop_model.state_index[('high', 'negative')].items():
            if state in high_risk_state_pops:
                cls.y_isolation[index] = high_risk_state_pops[state]

        cls.y_isolation[-2] += low_risk_state_pops['H'] + high_risk_state_pops['H']


        # Seting up low risk parameters
        cls.non_subpop_parameters = {'eta': 1 / 5, 'gamma': 1 / 7, 'sigma': 1 / 3}
        cls.p_high = 0.3
        cls.p_low = 0.01
        cls.hospitalisation_probs = {'p_[high]': cls.p_high, 'p_[low]': cls.p_low}
        cls.isolation_hospitalisation_probs = {
            **{'p_[high,' + isolation_group + ']': cls.hospitalisation_probs['p_[high]'] for isolation_group in
               cls.isolation_groups},
            **{'p_[low,' + isolation_group + ']': cls.hospitalisation_probs['p_[low]'] for isolation_group in
               cls.isolation_groups}
        }
        cls.beta = (2 / 7) / cls.N
        beta_parameters = {'beta' + subpop_suffix: cls.beta for subpop_suffix in cls._metapop_model.subpop_suffixes}
        cls.rho = 1
        cls.transmission_reduction_from_isolation = 0.6
        cls.interaction_parameters = {
            **{'rho' + subpop_suffix_i + '_[' + risk_group + ',negative]': cls.rho
               for subpop_suffix_i in cls._metapop_model.subpop_suffixes
               for risk_group in cls.risk_groups},
            **{'rho' + subpop_suffix_i + '_[' + risk_group + ',positive]': cls.rho * (
                        1 - cls.transmission_reduction_from_isolation)
               for subpop_suffix_i in cls._metapop_model.subpop_suffixes
               for risk_group in cls.risk_groups}
        }
        parameters_isolation = cls.non_subpop_parameters | cls.isolation_hospitalisation_probs | beta_parameters | cls.interaction_parameters
        cls.parameters_isolation = {parameter: parameters_isolation[parameter]
                                    for parameter in cls._metapop_model.parameter_names}

        #### Setting up event que
        cls.from_index = [index
                          for risk_group in cls.risk_groups
                          for state, index in cls._metapop_model.state_index[(risk_group, 'negative')].items()
                          if state in ['E', 'I']]
        cls.to_index = [index
                        for risk_group in cls.risk_groups
                        for state, index in cls._metapop_model.state_index[(risk_group, 'positive')].items()
                        if state in ['E', 'I']]
        cls.test_every_x_days = 7
        cls.complaince = 0.8
        cls.test_sensitivity = 0.6
        cls.rapid_pathogen_test_event = TransferEvent(name='Rapid Pathogen Test',
                                                      times=range(0, cls.end_day + cls.time_step, cls.test_every_x_days),
                                                      proportion=cls.test_sensitivity * cls.complaince,
                                                      from_index=cls.from_index,
                                                      to_index=cls.to_index)
        cls._testing_eventqueue = EventQueue(cls.rapid_pathogen_test_event)

    def setUp(self):
        self.metapop_model = self._metapop_model
        self.testing_eventqueue = copy.deepcopy(self._testing_eventqueue)

    def tearDown(self):
        self.metapop_model = None
        self.testing_eventqueue = None

    def test_get_event_names(self):
        self.assertEqual(self.testing_eventqueue.get_event_names(), ['Rapid Pathogen Test'])

    def test_run_simulation(self):
        results_rapid_test, transfer_df = self.testing_eventqueue.run_simulation(model_object=self.metapop_model,
                                                                                 run_attribute='integrate',
                                                                                 parameters=self.parameters_isolation,
                                                                                 parameters_attribute='parameters',
                                                                                 y0=self.y_isolation,
                                                                                 end_time=self.end_day,
                                                                                 start_time=0,
                                                                                 simulation_step=self.time_step)
        total_hospitalisations = results_rapid_test.loc[90, ('observed_states', 'H_cumulative')]
        peak_hospitalisations = max(results_rapid_test.loc[:, ('observed_states', 'H')])
        self.assertLess(total_hospitalisations, 29297.017609)
        self.assertAlmostEqual(total_hospitalisations, 13349.153783, 3)
        self.assertLess(peak_hospitalisations, 2966.336876)
        self.assertAlmostEqual(peak_hospitalisations, 1083.910371, 3)
    def test_make_events_nullevents(self):
        self.testing_eventqueue.make_events_nullevents(['Rapid Pathogen Test'])
        results_rapid_test, transfer_df = self.testing_eventqueue.run_simulation(model_object=self.metapop_model,
                                                                                 run_attribute='integrate',
                                                                                 parameters=self.parameters_isolation,
                                                                                 parameters_attribute='parameters',
                                                                                 y0=self.y_isolation,
                                                                                 end_time=self.end_day,
                                                                                 start_time=0,
                                                                                 simulation_step=self.time_step)
        total_hospitalisations = results_rapid_test.loc[90, ('observed_states', 'H_cumulative')]
        peak_hospitalisations = max(results_rapid_test.loc[:, ('observed_states', 'H')])
        self.assertAlmostEqual(total_hospitalisations, 29297.017609, 2)
        self.assertAlmostEqual(peak_hospitalisations, 2966.336876, 3)
    def test_reset_event_queue(self):
        self.testing_eventqueue.make_events_nullevents(event_names='Rapid Pathogen Test')
        self.testing_eventqueue.reset_event_queue()
        results_rapid_test, transfer_df = self.testing_eventqueue.run_simulation(model_object=self.metapop_model,
                                                                                 run_attribute='integrate',
                                                                                 parameters=self.parameters_isolation,
                                                                                 parameters_attribute='parameters',
                                                                                 y0=self.y_isolation,
                                                                                 end_time=self.end_day,
                                                                                 start_time=0,
                                                                                 simulation_step=self.time_step)
        total_hospitalisations = results_rapid_test.loc[90, ('observed_states', 'H_cumulative')]
        peak_hospitalisations = max(results_rapid_test.loc[:, ('observed_states', 'H')])
        self.assertLess(total_hospitalisations, 29297.017609)
        self.assertAlmostEqual(total_hospitalisations, 13349.153783, 3)
        self.assertLess(peak_hospitalisations, 2966.336876)
        self.assertAlmostEqual(peak_hospitalisations, 1083.910371, 3)

if __name__ == '__main__':
    unittest.main()