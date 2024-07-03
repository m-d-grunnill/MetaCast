"""
Creation:
    Author: Martin Grunnill
    Date: 2024-03-30
Description: 
    
"""
from unittest import TestCase
from metacast import MetaCaster
import copy
import numpy as np


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


def vaccination_parameters_setup(nu_unvaccinated,
                                 nu_vaccination_lag,
                                 l_v,
                                 h_v,
                                 other_parameters,
                                 metapop_model):
    parameters = {key: value for key, value in other_parameters.items() if key in metapop_model.parameter_names}
    parameters.update({'nu_unvaccinated': nu_unvaccinated,
                       'nu_vaccination_lag': nu_vaccination_lag})

    beta = other_parameters['beta']
    parameters.update({'beta_[high,unvaccinated]': beta,
                       'beta_[high,vaccinated]': beta * (1 - l_v),
                       'beta_[high,vaccination_lag]': beta,
                       'beta_[low,unvaccinated]': beta,
                       'beta_[low,vaccinated]': beta * (1 - l_v),
                       'beta_[low,vaccination_lag]': beta})

    if h_v < l_v:
        raise ValueError('h_v must be greater than or equal to l_v. ' +
                         'Otherwise vaccine reduced severity given reduced susceptibility is negative.')

    vaccine_reduced_severity_given_reduced_susceptibility = 1 - ((1 - h_v) / (1 - l_v))
    p_high = other_parameters['p_[high]']
    p_low = other_parameters['p_[low]']
    parameters.update({'p_[high,unvaccinated]': p_high,
                       'p_[high,vaccinated]': p_high * (1 - vaccine_reduced_severity_given_reduced_susceptibility),
                       'p_[high,vaccination_lag]': p_high,
                       'p_[low,unvaccinated]': p_low,
                       'p_[low,vaccinated]': p_low * (1 - vaccine_reduced_severity_given_reduced_susceptibility),
                       'p_[low,vaccination_lag]': p_low})
    rho = other_parameters['rho']
    parameters.update({'rho' + subpop_suffix_i + subpop_suffix_j: rho
                       for subpop_suffix_i in metapop_model.subpop_suffixes
                       for subpop_suffix_j in metapop_model.subpop_suffixes})
    return parameters


def prob_over_many_days_to_prob_on_a_day(prob, many_days):
    return 1 - (1 - prob) ** (1 / many_days)


class SetUpOfTests(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.subpop_model = subpop_model

        #### Setting up 'high' 'low' risk group metapopulation
        cls.risk_groups = ['low', 'high']
        cls._metapop_model = MetaCaster(dimensions=cls.risk_groups,
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

        # Setting up y for low high risk group metapopulation
        cls.y_low_high = np.zeros(cls._metapop_model.total_states)
        cls.low_risk_state_pops = {'S': cls.low_risk_population * (1 - cls.prevelance), **cls.low_risk_infected}
        for state, index in cls._metapop_model.state_index['low'].items():
            if state in cls.low_risk_state_pops:
                cls.y_low_high[index] = cls.low_risk_state_pops[state]

        cls.high_risk_state_pops = {'S': cls.high_risk_population * (1 - cls.prevelance), **cls.high_risk_infected}
        for state, index in cls._metapop_model.state_index['high'].items():
            if state in cls.high_risk_state_pops:
                cls.y_low_high[index] = cls.high_risk_state_pops[state]

        cls.y_low_high[-2] += cls.low_risk_state_pops['H'] + cls.high_risk_state_pops['H']

        # Seting up low risk parameters
        cls.non_subpop_parameters = {'eta': 1 / 5, 'gamma': 1 / 7, 'sigma': 1 / 3}
        cls.p_high = 0.3
        cls.p_low = 0.01
        cls.hospitalisation_probs = {'p_[high]': cls.p_high, 'p_[low]': cls.p_low}
        cls.beta = (2 / 7) / cls.N
        beta_parameters = {'beta' + subpop_suffix: cls.beta for subpop_suffix in cls._metapop_model.subpop_suffixes}
        cls.rho = 1
        interaction_parameters = {'rho' + subpop_suffix_i + subpop_suffix_j: cls.rho
                                  for subpop_suffix_i in cls._metapop_model.subpop_suffixes
                                  for subpop_suffix_j in cls._metapop_model.subpop_suffixes}
        parameters_low_high = cls.non_subpop_parameters | cls.hospitalisation_probs | beta_parameters | interaction_parameters
        cls.parameters_low_high = {parameter: parameters_low_high[parameter]
                                   for parameter in cls._metapop_model.parameter_names}

        # Set up alternative dimensions
        cls.isolation_groups = ['negative', 'positive']
        cls.rapid_pathogen_test_dimensions = [cls.risk_groups, cls.isolation_groups]
        vaccination_groups = ['unvaccinated', 'vaccination_lag' 'vaccinated']
        vaccination_transfers = [{'from_coordinates': (risk_group, 'unvaccinated'),
                                  'to_coordinates': (risk_group, 'vaccination_lag'),
                                  'states': 'all', 'parameter': 'nu_unvaccinated'}
                                 for risk_group in cls.risk_groups]
        vacination_lag_transfers = [{'from_coordinates': (risk_group, 'vaccination_lag'),
                                     'to_coordinates': (risk_group, 'vaccinated'),
                                     'states': 'all', 'parameter': 'nu_vaccination_lag'}
                                    for risk_group in cls.risk_groups]
        cls.vaccination_dimensions = vaccination_transfers + vacination_lag_transfers

    def setUp(self):
        self.metapop_model = copy.deepcopy(self._metapop_model)

    def tearDown(self):
        self.metapop_model = None

    def change_to_vaccination_dimensions(self, include_parameters=True):
        metapop_model_vaccination = copy.deepcopy(self.metapop_model)
        metapop_model_vaccination.dimensions = self.vaccination_dimensions
        # Setting up y for vaccination metapopulation
        y_vaccination = np.zeros(metapop_model_vaccination.total_states)

        for state, index in metapop_model_vaccination.state_index[('high', 'unvaccinated')].items():
            if state in self.high_risk_state_pops:
                y_vaccination[index] = self.high_risk_state_pops[state]

        for state, index in metapop_model_vaccination.state_index[('low', 'unvaccinated')].items():
            if state in self.low_risk_state_pops:
                y_vaccination[index] = self.low_risk_state_pops[state]

        y_vaccination[-2] += self.low_risk_state_pops['H'] + self.high_risk_state_pops['H']
        if include_parameters:
            v_day = self.end_day
            complaince = 0.8
            prob_vaccinated_by_v_day = 0.7
            prob_vaccinated_per_day = prob_over_many_days_to_prob_on_a_day(prob_vaccinated_by_v_day, v_day)
            vaccination_lag = 1 / 14
            vaccine_reduced_susceptibility = 0.4
            vaccine_reduced_severity = 0.8
            other_parameters = {'beta': self.beta, 'rho': self.rho, **self.non_subpop_parameters,
                                **self.hospitalisation_probs}
            parameters_vaccination = vaccination_parameters_setup(nu_unvaccinated=prob_vaccinated_per_day * complaince,
                                                                  nu_vaccination_lag=vaccination_lag,
                                                                  l_v=vaccine_reduced_susceptibility,
                                                                  h_v=vaccine_reduced_severity,
                                                                  other_parameters=other_parameters,
                                                                  metapop_model=metapop_model_vaccination)
            parameters_vaccination = {parameter: parameters_vaccination[parameter]
                                      for parameter in metapop_model_vaccination.parameter_names}
            return metapop_model_vaccination, y_vaccination, parameters_vaccination
        else:
            return metapop_model_vaccination, y_vaccination

