"""
Creation:
    Author: mdgru 
    Date: 2022-12-21
Description: Diagnosed and Hospitalised Flu Metapopulations model.
    
"""

from metacast.metacaster import MetaCaster


class FluDiagHosp(MetaCaster):
    """
    Flu metapopulation model with diagnosed at risk of being hospitalised.
        First dimension is members are referred to clusters.
    Second dimension clusters are referred to vaccination groups.


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
    isolation_modifier : string
        Factor used to modify transmission from infectious but isolating states. If None a factor of 1 is used in the
        isolation_modifier's place when using method calculating_fois.
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
    states = ['S', 'E', 'I', 'D', 'U', 'H', 'R']
    observed_states = ['Cumulative hospitalisation', 'Cumulative diagnosis']
    infected_states = ['E', 'I', 'D', 'U', 'H']
    symptomatic_states = ['D', 'H']
    hospitalised_states = ['H']
    infectious_states = ['I', 'D', 'U']
    isolating_states = ['D']
    non_transmission_universal_params = ['sigma', 'p_d', 'delta', 'gamma_u',
                                         'gamma_d', 'p_hd', 'gamma_h'
                                         ]
    isolation_modifier = 'kappa'
    vaccine_specific_params = ['l', 'h']
    transmission_cluster_specific = True

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
                ve_hospitalisation = 'h_' + vaccine_group # vaccine efficacy against symptoms for this vaccine group.
                self.sub_pop_transfer(y, y_deltas, t, cluster, vaccine_group, parameters)
                states_index = self.state_index[cluster][vaccine_group] # Dictionary of state indexes for this cluster
                # and vaccine group.
                # Infections
                infections = (1 - parameters[ve_infection]) * foi * y[states_index['S']]
                # progression from latent infection
                prog_from_latent = parameters['sigma']*y[states_index['E']]
                # Progression from early symptoms
                prog_from_early_symps =  parameters['delta']*y[states_index['I']]
                # Moving to diagnosed
                prog_to_diagnosed = parameters['p_d']*prog_from_early_symps
                # Moving to undiagnosed
                prog_to_undiagnosed = (1-parameters['p_d']) * prog_from_early_symps
                # progression from diagnosed
                prog_from_diagnosed = parameters['gamma_d'] * y[states_index['D']]
                # Moving to hospital
                prob_of_hospitilisation = (1 - parameters[ve_hospitalisation]) * parameters['p_hd']
                hospitalisation = prob_of_hospitilisation * prog_from_diagnosed
                # Diagnosed moving to recovered
                diagnosed_recovery = (1-prob_of_hospitilisation)*prog_from_diagnosed
                # Undiagnosed moving to recovered
                undiagnosed_recovery = parameters['gamma_u'] * y[states_index['U']]
                # Hospitised recovery
                hospitised_recovery = parameters['gamma_h'] * y[states_index['H']]

                # Updating y_deltas with derivative calculations from this cluster and vaccine group.
                y_deltas[states_index['S']] += - infections
                y_deltas[states_index['E']] += infections - prog_from_latent
                y_deltas[states_index['I']] += prog_from_latent - prog_from_early_symps
                y_deltas[states_index['U']] += prog_to_undiagnosed - undiagnosed_recovery
                y_deltas[states_index['D']] += prog_to_diagnosed - prog_from_diagnosed
                y_deltas[states_index['H']] += hospitalisation - hospitised_recovery
                y_deltas[states_index['R']] += diagnosed_recovery+undiagnosed_recovery+hospitised_recovery
                y_deltas[-2] += hospitalisation
                y_deltas[-1] += prog_to_diagnosed

        return y_deltas




