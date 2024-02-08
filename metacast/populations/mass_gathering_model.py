"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""

from metacast.metacaster import MetaCaster

# Code section below is silenced. This section was meant for use with DOK version of the models jacobian.
# import os
# import numpy as np
# # find this files directory so as to later find jacobians saved a json files
# abspath = os.path.abspath(__file__)
# dir_name = os.path.dirname(abspath) +'/'


class MassGatheringModel(MetaCaster):
    """
    COVID-19 at mass gathering model with diagnosed at risk of being hospitalised.
        First dimension is members are referred to clusters.
        Second dimension clusters are referred to vaccination groups.

    Model structure used in:
    Grunnill, M., Arino, J., McCarthy, Z., Bragazzi, N. L., Coudeville, L., Thommes, E., Amiche, A., Ghasemi, A., Bourouiba, L., Tofighi, M., Asgary, A., Baky-Haskuee, M., & Wu, J. (2024). Modelling Disease Mitigation at Mass Gatherings: A Case Study of COVID-19 at the 2022 FIFA World Cup. In E. H. Lau (Ed.), PLoS Computational Biology: Vol. January (Issue 1, p. e1011018). Public Library of Science. https://doi.org/10.1371/JOURNAL.PCBI.1011018


    Attributes
    ----------
    states : list of strings
        States used in model.
    observed_states : list of strings
        Observed states. Useful for obtaining results or fitting (e.g. Cumulative incidence). 
    infected_states : list of strings
        Infected states (not necessarily infectious). 
    hospitalised_states : list of strings
        Hospitalised states. 
    infectious_states : list of strings
        Infectious states. These states contribute to force of infection. 
    symptomatic_states : list of strings
        Symptomatic states. NOTE any state in the list self.infectious_states but NOT in this list has its transmission
        modified by self.asymptomatic_transmission_modifier (see method calculate_fois). 
    isolating_states : list of strings
        Isolating states.   NOTE any state in this list AND self.infectious_states has its
        transmission modified by isolation_modifier (see method calculate_fois).
    asymptomatic_transmission_modifier : string
        Factor used to modify transmission from infectious but asymptomatic states. If None a factor of 1 is used in the
        asymptomatic_transmission_modifier's place when using method calculating_fois.
    non_transmission_universal_params : list of strings
        A list of all the parameters that are NOT:
        - directly to do with transmission
        - cluster specific
        - vaccine group specific
        

    transmission_cluster_specific : bool
        Default value is True. If false it is assumed that classes mix homogeneously. If true transmission
        is assumed to be different between each class interaction.
    vaccine_specific_params : list of strings
        Parameters that are vaccine group specific. 



    Methods
    ode(y, t, *parameters)
        Calculate derivative of this models state variables for time t.
    """
    states = ['S', 'E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_H', 'M_I', 'M_A', 'F_H', 'F_I', 'F_A', 'R']
    observed_states = ['Cumulative hospitalisation', 'Cumulative infections']
    infected_states = ['E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_H',  'M_I', 'M_A', 'F_H',  'F_I', 'F_A']
    hospitalised_states = ['F_H']
    infectious_states = ['P_I', 'P_A',  'M_I', 'M_A', 'M_H',  'F_I', 'F_A']
    symptomatic_states = ['M_I', 'F_I',  'M_H', 'F_H']
    non_transmission_universal_params = ['epsilon_1', 'epsilon_2', 'epsilon_3', 'epsilon_H',
                                         'p_s', 'p_h_s',
                                         'gamma_A_1', 'gamma_A_2',
                                         'gamma_I_1', 'gamma_I_2',
                                         'gamma_H',
                                         'alpha'
                                         ]
    vaccine_specific_params = ['l', 's', 'h']
    transmission_cluster_specific = True
    asymptomatic_transmission_modifier = 'theta'


    def ode(self, y, t, *parameters):
        """
        Calculate derivative of this models state variables for time t.
        This method is for use either:
            - within method integrate inherited from base parent class (see docstring of integrate method in parent
              class).
            - with scipy.integrate.odeint (see
              https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html)

        Parameters
        ----------
        y : numpy.array
            State variable values at time t.
        t : float
            Time t.
        *parameters : Numeric values
            Values of parameters must be given as same order as self.all_parameters omitting any parameters estimated
            via piecewise methods.

        Returns
        -------
        numpy.array
            Derivatives of state variables at time t.
        """
        parameters, y_deltas, fois = super().setup_child_ode_method(y, parameters)
        for cluster in self.clusters:
            foi = fois[cluster] # force of infection experienced by this specific cluster.
            for vaccine_group in self.vaccine_groups:
                ve_infection = 'l_'+vaccine_group # vaccine efficacy against infection for this vaccine group.
                ve_symptoms = 's_' + vaccine_group # vaccine efficacy against symptoms for this vaccine group.
                ve_hospitalisation = 'h_' + vaccine_group # vaccine efficacy against hospitilisation for this vaccine group.
                self.group_transfer(y, y_deltas, t, cluster, vaccine_group, parameters)
                states_index = self.state_index[cluster][vaccine_group] # Dictionary of state indexes for this cluster
                # and vaccine group.
                # Infections
                infections = (1 - parameters[ve_infection]) * foi * y[states_index['S']]
                # progression to RT-pcr sensitivity
                prog_rtpcr = parameters['epsilon_1'] * y[states_index['E']]
                p_s_v = parameters['p_s'] * (1 - parameters[ve_symptoms])
                prog_symptomatic_path = prog_rtpcr * p_s_v
                prog_asymptomatic_path = prog_rtpcr * (1-p_s_v)


                # progression to lfd/rapid antigen test sensitivity
                prog_LFD_sensitive_symptomatic_path = parameters['epsilon_2'] * y[states_index['G_I']]
                prog_LFD_sensitive_asymptomatic_path = parameters['epsilon_2'] * y[states_index['G_A']]


                # progression to mid-infection for symptomatics.
                prog_symptoms = parameters['epsilon_3'] * y[states_index['P_I']]
                p_h_v = parameters['p_h_s'] * (1 - parameters[ve_hospitalisation])
                prog_hospital_path = p_h_v*prog_symptoms
                prog_not_hospital_path = prog_symptoms-prog_hospital_path
                # progression to mid-infection for asymptomatics.
                prog_mid_asymptomatic_stage = parameters['epsilon_3'] * y[states_index['P_A']]

                # Progression to tate infection
                prog_late_asymptomatic_stage = parameters['gamma_A_1'] * y[states_index['M_A']]
                prog_late_symptomatic_stage = parameters['gamma_I_1'] * y[states_index['M_I']]
                hospitalisation = parameters['epsilon_H'] * y[states_index['M_H']]

                # Recovery
                asymptomatic_recovery = parameters['gamma_A_2'] * y[states_index['F_A']]
                symptomatic_recovery = parameters['gamma_I_2'] * y[states_index['F_I']]
                hospital_recovery = parameters['gamma_H'] * y[states_index['F_H']]

                # Natural wanning imunity
                waned_natural_immunity = parameters['alpha'] * y[states_index['R']]

                # Updating y_deltas with derivative calculations from this cluster and vaccine group.
                y_deltas[states_index['S']] += waned_natural_immunity - infections
                y_deltas[states_index['E']] += infections - prog_rtpcr
                y_deltas[states_index['G_I']] += prog_symptomatic_path - prog_LFD_sensitive_symptomatic_path
                y_deltas[states_index['G_A']] += prog_asymptomatic_path - prog_LFD_sensitive_asymptomatic_path
                y_deltas[states_index['P_I']] += prog_LFD_sensitive_symptomatic_path - prog_symptoms
                y_deltas[states_index['P_A']] += prog_LFD_sensitive_asymptomatic_path - prog_mid_asymptomatic_stage
                y_deltas[states_index['M_A']] += prog_mid_asymptomatic_stage-prog_late_asymptomatic_stage
                y_deltas[states_index['M_I']] += prog_not_hospital_path-prog_late_symptomatic_stage
                y_deltas[states_index['M_H']] += prog_hospital_path - hospitalisation
                y_deltas[states_index['F_A']] += prog_late_asymptomatic_stage-asymptomatic_recovery
                y_deltas[states_index['F_I']] += prog_late_symptomatic_stage-symptomatic_recovery
                y_deltas[states_index['F_H']] += hospitalisation - hospital_recovery
                y_deltas[states_index['R']] += hospital_recovery + asymptomatic_recovery + symptomatic_recovery - waned_natural_immunity
                y_deltas[-2] += hospitalisation
                y_deltas[-1] += infections
                # self.ode_calls_dict[key] = y_deltas

        return y_deltas
