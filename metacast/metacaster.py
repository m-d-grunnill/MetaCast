"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description: 
    Contains class for setting up and simulating two-dimensional metapopulation models.

Classes
-------
Base2DMetaPopModel
    Base class for setting up and simulating two-dimensional metapopulation models.
    It is intended that two-dimensional models using this modules API should create a class that
    inherits that is a child of this class and has the method ode.
    First dimension is members are referred to clusters.
    Second dimension clusters are referred to vaccination groups.
"""
from inspect import getargspec
from collections.abc import Iterable
import numpy as np
import pandas as pd
from numbers import Number
import scipy


def _nested_dict_values(d):
    return [index for sub_d in d.values() for index in sub_d.values()]


def _unionise_dict_of_lists(dict_of_lists):
    """
    Merge dictionary value lists into a list of unique values.

    Parameters
    ----------
    dict_of_lists : dictionay of lists

    Returns
    -------
    list
        Merged unique list values.
    """
    return list(set().union(*dict_of_lists.values()))


def _is_iterable_of_unique_elements(object):
    if not isinstance(object, Iterable):
        return False
    elif len(object) > len(set(object)):
        return False
    else:
        return True


def _is_set_like_of_strings(thing):
    if not _is_iterable_of_unique_elements(thing):
        return False
    elif isinstance(thing, (dict, str)):
        return False
    elif any(not isinstance(element, str) for element in thing):
        return False
    else:
        return True

def select_dict_items_in_list(dictionary, lst):
    return {key: value for key, value in dictionary.items() if key in lst}


class MetaCaster:
    """
    Base class for setting up and simulating two-dimensional metapopulation models.
    First dimension is members are referred to clusters.
    Second dimension clusters are referred to vaccination groups.

    Parameters
    ----------
    scaffold : dictionary, list or tuple
        If dictionary group_structure must contain the key values pairs:
            clusters: list of strings'
                Names given to clusters.
            vaccine groups: list of strings'
                Names given to vaccine groups.
        If list or tuple each entry must be a dictionary that defines a transition.
        These dictionaries must have the key values pairs:
            from_cluster: string
                Cluster from which hosts are leaving.
            to_cluster: string
                Cluster to which hosts are going.
            from_vaccine_group: string
                Vaccine group from which hosts are leaving.
            to_vaccine_group: string
                Vaccine group to which hosts are going.
            states: list of strings or string
                Host states which will transition between clusters and vaccine groups. Single entry of 'all' value
                means all the available model states transition between clusters and vaccine groups.
            parameter : string
                Name given to parameter that is responsible for flow of hosts transitions between clusters and
                vaccine groups.
        Optional key value pairs:
            piecewise targets: list, tuple, numpy.array or pandas.series
                Targets for piecewise estimation of parameter that is responsible for flow of hosts transitions
                between clusters and vaccine groups (see method group_transfer).
    
    
    Attributes
    ----------
    ode: None
        This is None in base parent class, but must be overridden with a method calculating derivative of state variables
        at t in child classes.
    states : list of strings 
        States used in model. Empty in base parent class.
    observed_states : list of strings
        Observed states. Useful for obtaining results or fitting (e.g. Cumulative incidence). Empty in base parent class.
    infected_states : list of strings
        Infected states (not necessarily infectious). Empty in base parent class.
    hospitalised_states : list of strings
        Hospitalised states. Empty in base parent class.
    infectious_states : list of strings
        Infectious states. These states contribute to force of infection. Empty in base parent class.
    symptomatic_states : list of strings
        Symptomatic states. NOTE any state in the list self.infectious_states but NOT in this list has its transmission
        modified by self.asymptomatic_transmission_modifier (see method calculate_fois). Empty in base parent class.
    isolating_states : list of strings
        Isolating states.  Empty in base parent class. NOTE any state in this list AND self.infectious_states has its
        transmission modified by isolation_modifier (see method calculate_fois).
    transmission_term : string
        General transmission term used in calculating forces of infection, default term is 'beta' (see method
        calculate_fois). If  attribute transmission_cluster_specific == False transmission terms are generate of the form
        self.transmission_term + '_' cluster_i + '_' cluster_j.
    population_term : string
        General population term used as denominator in calculating force of infection, default term is 'N' (see method
        calculate_fois). If  self.transmission_cluster_specific == False population terms are generate of the form
        self.population_term + '_' + cluster_i + '_' + cluster_j
    transmission_cluster_specific : bool
        Default value is False. If false it is assumed that classes mix homogeneously. If true transmission
        is assumed to be different between each class interaction.
    isolation_modifier : string or None
        Factor used to modify transmission from infectious but isolating states. If None a factor of 1 is used in the
        isolation_modifier's place when using method calculating_fois.
    isolation_cluster_specific : bool
        Default value is False. If True isolation is specific to a cluster and isolation_modifier are used in the
        parameters, taking the form self.isolation_modifier + '_' +cluster_j .
    asymptomatic_transmission_modifier : string or None
        Factor used to modify transmission from infectious but asymptomatic states. If None a factor of 1 is used in the
        asymptomatic_transmission_modifier's place when using method calculating_fois.
    non_transmission_universal_params : list of strings
        A list of all the parameters that are NOT:
        - directly to do with transmission
        - cluster specific
        - vaccine group specific
        Empty in base parent class.
    non_transmission_cluster_specific_params : list of strings
         A list of all the parameters that are cluster specific but NOT directly to do with transmission. Empty in base
        parent class.
    vaccine_specific_params : list of strings
        Parameters that are vaccine group specific. Empty in base parent class.
    all_states_index : dictionary
        Keys are all the states values are the associated indexes for use with numpy.arrays.
    state_index : 3 level nested dictionary.
        First level: keys are the cluster names values are another dictionary.
            Second level: keys are the vaccine_group names and values are another dictionary.
                Third level: keys are the states and values (ints) are the associated indexes for use with
                             numpy.arrays.
    infected_states_index_list : list of ints
        A list of the indexes of infected states.
    hospitalised_states_index_list : list of ints
        A list of the indexes of hospitalised states.
    infectious_symptomatic_indexes : 2 level nested dictionary.
        First level: keys are the cluster names values are another dictionary.
            Second level: keys are the vaccine_group names and values is a list of indexes for infectious and
                          symptomatic states.
    infectious_asymptomatic_indexes : 2 level nested dictionary.
        First level: keys are the cluster names values are another dictionary.
            Second level: keys are the vaccine_group names and values is a list of indexes for infectious and
                          asymptomatic states.
    isolating_symptomatic_indexes : 2 level nested dictionary.
        First level: keys are the cluster names values are another dictionary.
            Second level: keys are the vaccine_group names and values is a list of indexes for isolating symptomatic
            states.
    isolating_asymptomatic_indexes : 2 level nested dictionary.
        First level: keys are the cluster names values are another dictionary.
            Second level: keys are the vaccine_group names and values is a list of indexes for isolating asymptomatic
            states.
    infectious_and_symptomatic_states : list of stings
        A list of infectious and symptomatic states.
    infectious_and_asymptomatic_states : list of stings
        A list of infectious and asymptomatic states.
    isolating_and_symptomatic_states : list of stings
        A list of isolating and symptomatic states.
    isolating_and_asymptomatic_states : list of stings
        A list of isolating and asymptomatic states.
    num_state : int
        Total number of states in model.
    num_param : int
        Total number of parameters in model.
    all_parameters : list of strings
        A list of parameters sorted alpha numerically.

    Methods
    -------
    get_transmission_terms_between(clusters)
        Get a dictionary of transmission terms between all clusters in list provided.
    group_transfer(y, y_deltas, t, from_cluster, from_vaccine_group, parameters)
        Calculates the transfers of people from clusters and vaccination groups.
    setup_child_ode_method(y, parameters)
        Wrapper function for setting ode method in child classes.
    get_clusters_indexes(clusters)
        Returns a list of the indexes for all the states in the clusters given.
    get_vaccine_group_indexes(vaccine_groups)
        Returns a list of the indexes for all the states in the vaccine groups given.
    calculate_fois(y, parameters)
        Calculates the Forces of Infection (FOI) given variables in y and parameters.
    integrate(x0, t, full_output=False, called_in_fitting=False, **kwargs_to_pass_to_odeint)
        Simulate model via integration using initial values x0 of time range t.
        A wrapper on top of :mod:`odeint <scipy.integrate.odeint>`
        Modified method from the pygom method `DeterministicOde <pygom.model.DeterministicOde>`.
    """
    states = None
    infected_states = None
    infectious_states = infected_states
    symptomatic_states = infected_states
    observed_states = infected_states
    hospitalised_states = None
    isolating_states = None
    transmission_term = 'beta'
    population_term = 'N'
    transmission_cluster_specific = False
    isolation_modifier = None
    isolation_cluster_specific = False
    asymptomatic_transmission_modifier = None
    non_transmission_universal_params = None
    non_transmission_cluster_specific_params = None  # does not include transmission term beta.
    vaccine_specific_params = None
    _sub_pop_model = None

    def _check_model_attributes(self):
        if self.states is None:
            raise AssertionError('Model attribute "states" needs to be defined at initialisation of MetaCaster,' +
                                 ' or given as an attribute of a child class of MetaCaster.')
        if not _is_set_like_of_strings(self.states):
            raise TypeError('Model attribute "states" should be a collection of unique strings.')
        for model_attribute in ['infected_states',
                                'infectious_states',
                                'symptomatic_states',
                                'observed_states',
                                'hospitalised_states',
                                'isolating_states',
                                'non_transmission_universal_params',
                                'non_transmission_cluster_specific_params',
                                'vaccine_specific_params']:
            if eval('self.' + model_attribute) is not None:
                if not _is_set_like_of_strings(eval('self.' + model_attribute)):
                    raise TypeError('Model attribute "' +
                                    model_attribute +
                                    '" should be a collection of unique strings.')
        for model_attribute in ['transmission_cluster_specific',
                                'isolation_cluster_specific']:
            if not isinstance('self.' + model_attribute, bool):
                raise TypeError('Model attribute "' +
                                model_attribute +
                                '" should be a bool value.')

    def _set_model_attributes(self,
                              model_attributes,
                              ):
        for name, value in model_attributes.items():
            if not hasattr(self, name):
                raise AssertionError(name + 'is not a model attribute.')
            setattr(self, name, value)

    def __init__(self, scaffold, model_attributes=None, sub_pop_model=None):
        if model_attributes is not None:
            self._set_model_attributes(model_attributes)

        self._check_model_attributes()
        if sub_pop_model is not None:
            self.sub_pop_model = sub_pop_model

        if type(self) == MetaCaster:
            raise TypeError('Base2DMetaPopModel is not meant to run models, only its children.' +
                            '\nIf unfamiliar with class inheritance look  up:\n' +
                            ' https://www.w3schools.com/python/python_inheritance.asp.')
        self.all_parameters = set(self.non_transmission_universal_params)
        for transmission_modifier in [self.isolation_modifier,
                                      self.asymptomatic_transmission_modifier]:
            if transmission_modifier is not None:
                self.all_parameters.add(transmission_modifier)
        if self.asymptomatic_transmission_modifier is not None:
            self.all_parameters.add(self.asymptomatic_transmission_modifier)
        self._gen_structure(scaffold)
        self.all_parameters.update([param + '_' + vaccine_group
                                    for vaccine_group in self.vaccine_groups
                                    for param in self.vaccine_specific_params
                                    if param not in self.non_transmission_cluster_specific_params])
        self.all_parameters.update([param + '_' + cluster
                                    for cluster in self.clusters
                                    for param in self.non_transmission_cluster_specific_params
                                    if param not in self.vaccine_specific_params])
        self.all_parameters.update([param + '_' + cluster + '_' + vaccine_group
                                    for cluster in self.clusters
                                    for vaccine_group in self.vaccine_groups
                                    for param in self.vaccine_specific_params
                                    if param in self.non_transmission_cluster_specific_params])
        if self.transmission_cluster_specific:
            self.transmission_to_terms = {cluster_i: {self.transmission_term: [],
                                                      self.population_term: []
                                                      } for cluster_i in self.clusters}

            self.transmission_from_terms = {cluster_j: {self.transmission_term: [],
                                                        self.population_term: []
                                                        } for cluster_j in self.clusters}
            for cluster_i in self.clusters:
                term = self.population_term + '_' + cluster_i
                self.all_parameters.add(term)
                self.transmission_to_terms[cluster_i][self.population_term].append(term)
                for cluster_j in self.clusters:
                    term = self.transmission_term + '_' + cluster_i + '_' + cluster_j
                    self.all_parameters.add(term)
                    self.transmission_to_terms[cluster_i][self.transmission_term].append(term)
                    self.transmission_from_terms[cluster_j][self.transmission_term].append(term)
        if self.isolation_cluster_specific:
            if self.isolation_modifier is None:
                raise AssertionError('isolation_modifier must be specified to be considered cluster specific')
            if not self.transmission_cluster_specific:
                raise AssertionError('isolation being cluster specific is only supported when ' +
                                     'transmission is cluster specific.')
            self.all_parameters.update([self.isolation_modifier + '_' + cluster
                                        for cluster in self.clusters])
        self.all_parameters = sorted(self.all_parameters)
        non_piece_wise_params_names = set(self.all_parameters) - set(self.params_estimated_via_piecewise_method)
        self.non_piece_wise_params_names = sorted(list(non_piece_wise_params_names))
        self._sorting_states()
        self._parameters = None
        self.num_param = len(self.all_parameters)
        self.piecewise_est_param_values = None


    def get_transmission_terms_between(self, clusters):
        """
        Get a dictionary of transmission terms between all clusters in list provided.

        Parameters
        ----------
        clusters : list of strings

        Returns
        -------
        transmission_terms_dict: dictionary
            The first entry of which the key is self.transmission_term the value (default this is beta) is a list of the
            transmission terms (factors used for multiplying with infectious states in a cluster).
            The second entry of which the key is self.population_term the value (default this is N)is a list of the
             population terms (the denominators used when calculating the forces of infections).


        """
        population_terms = []
        transmission_terms = []
        for cluster_i in clusters:
            for cluster_j in clusters:
                transmission_term = self.transmission_term + '_' + cluster_i + '_' + cluster_j
                population_term = self.population_term + '_' + cluster_i + '_' + cluster_j
                if transmission_term not in transmission_terms:
                    transmission_terms.append(transmission_term)
                    population_terms.append(population_term)
        transmission_terms_dict = {self.transmission_term: transmission_terms, self.population_term: population_terms}
        return transmission_terms_dict

    def _gen_structure(self, scaffold):
        """
        Sets up structure of flow between sub-population models.

        Parameters
        ----------
        scaffold : dictionary, list or tuple
            If dictionary group_structure must contain the key values pairs:
                clusters: list of strings'
                    Names given to clusters.
                vaccine groups: list of strings'
                    Names given to vaccine groups.
            If list or tuple each entry must be a dictionary that defines a transition.
            These dictionaries must have the key values pairs:
                from_cluster: string
                    Cluster from which hosts are leaving.
                to_cluster: string
                    Cluster to which hosts are going.
                from_vaccine_group: string
                    Vaccine group from which hosts are leaving.
                to_vaccine_group: string
                    Vaccine group to which hosts are going.
                states: list of strings or string
                    Host states which will transition between clusters and vaccine groups. Single entry of 'all' value
                    means all the available model states transition between clusters and vaccine groups.
                parameter : string
                    Name given to parameter that is responsible for flow of hosts transitions between clusters and
                    vaccine groups.
            Optional key value pairs:
                piecewise targets: list, tuple, numpy.array or pandas.series
                    Targets for piecewise estimation of parameter that is responsible for flow of hosts transitions
                    between clusters and vaccine groups (see method group_transfer).


        Returns
        -------
        Nothing.
        """
        self.params_estimated_via_piecewise_method = []
        self.subpop_transfer_dict = {}
        self.sub_pop_transition_params_dict = {}
        self.vaccine_groups = set()
        self.clusters = set()
        self.one_dimensional_metapopulation = True


        if isinstance(scaffold, dict):
            self.clusters.update(scaffold['clusters'])
            if 'vaccine_group' in scaffold:
                self.vaccine_groups.update(scaffold['vaccine groups'])
        elif isinstance(scaffold, (list, tuple)):
            for count, group_transfer in enumerate(scaffold):
                cluster = group_transfer['from_cluster']
                self.clusters.add(cluster)
                if cluster not in self.subpop_transfer_dict:
                    self.subpop_transfer_dict[cluster] = {}
                if 'from_vaccine_group' in group_transfer:
                    if 'to_vaccine_group' not in group_transfer:
                        raise AssertionError("If 'from_vaccine_group' is in scaffold 'to_vaccine_group' must be given.")
                    if count == 0:
                        self.one_dimensional_metapopulation = False
                    else:
                        raise AssertionError('Either all scaffold entries should have a' +
                                             ' "from_vaccine_group" entry of none should.')
                    vaccine_group = group_transfer['from_vaccine_group']
                    self.vaccine_groups.add(vaccine_group)
                    if vaccine_group not in self.subpop_transfer_dict[cluster]:
                        self.subpop_transfer_dict[cluster][vaccine_group] = []

                to_cluster = group_transfer['to_cluster']
                self.clusters.add(to_cluster)
                if 'to_vaccine_group' in group_transfer:
                    to_vaccine_group = group_transfer['to_vaccine_group']
                    self.vaccine_groups.add(to_vaccine_group)

                if group_transfer['states'] == 'all':
                    group_transfer['states'] = self.states
                else:
                    if not isinstance(group_transfer['states'], (list, tuple)):
                        group_transfer['states'] = list(group_transfer['states'])

                    for state in group_transfer['states']:
                        self._check_string_in_list_strings(state, 'states')
                parameter = group_transfer['parameter']
                if not isinstance(parameter, str):
                    raise TypeError(str(parameter) + ' should be of type string.')
                if parameter not in self.sub_pop_transition_params_dict:
                    self.sub_pop_transition_params_dict[parameter] = []
                entry = {key: value for key, value in
                         group_transfer.items()
                         if key != 'parameter'}
                self.sub_pop_transition_params_dict[parameter].append(entry)
                self.all_parameters.add(parameter)
                if 'piecewise targets' in group_transfer:
                    self.params_estimated_via_piecewise_method.append(parameter)
                    if isinstance(group_transfer['piecewise targets'], pd.Series):
                        group_transfer['piecewise targets'] = group_transfer['piecewise targets'].tolist()

                if self.one_dimensional_metapopulation:
                    entry = {key: value
                             for key, value in group_transfer.items()
                             if key != 'from_cluster'}
                    self.subpop_transfer_dict[cluster].append(entry)
                else:
                    entry = {key: value
                             for key, value in group_transfer.items()
                             if key not in ['from_cluster', 'from_vaccine_group']}
                    self.subpop_transfer_dict[cluster][vaccine_group].append(entry)
        else:
            raise TypeError('group_structure must be a dictionary, list or tuple.')

    def sub_pop_transfer(self, y, y_deltas, t,
                         from_cluster,
                         from_vaccine_group,
                         parameters
                         ):
        """
        Calculates the transfers of people between subpopulation.

        Parameters
        ----------
        y : numpy.array
            Values of variables at time t.
        y_deltas : numpy.array
            Store of delta (derivative) of variables in y which this method adds/subtracts to.
        t : float
            Time t for which derivative is being calculated.
        from_cluster : string
            Cluster from which transfers are being made.
        from_vaccine_group : string
            Vaccine group from which transfers are being made.
        parameters : dictionary {keys are strings: values are numeric}
            Dictionary of parameter values used in calculating derivative.

        Returns
        -------
        y_deltas : numpy.array
            Store of delta (derivative) of variables in y which this method adds/subtracts to.

        """
        if from_cluster in self.subpop_transfer_dict:
            if from_vaccine_group in self.subpop_transfer_dict[from_cluster]:
                group_transfers = self.subpop_transfer_dict[from_cluster][from_vaccine_group]
                from_index_dict = self.state_index[from_cluster][from_vaccine_group]
                for group_transfer in group_transfers:
                    parameter = group_transfer['parameter']
                    if 'piecewise targets' in group_transfer:
                        # This section allows for the piecewise estimation of a people being transferred between groups.
                        # For example say from_vaccine_group=unvaccinated, t=0 and 15 people of this cluster
                        # got vaccinated on t=1. The code within this if statement calculate rate of change at t
                        # to get 15 people being transfered to the vaccinated group.
                        if t in self.piecewise_est_param_values[parameter]:
                            param_val = self.piecewise_est_param_values[parameter][t]
                        else:
                            index_of_t = int(t) + 1
                            total_being_tranfered = group_transfer['piecewise targets'][index_of_t]
                            if total_being_tranfered == 0:  # No point in calculations if no one is being vaccinated.
                                param_val = 0
                            else:
                                from_states_index = [from_index_dict[state] for state in group_transfer['states']]
                                total_avialable = y[from_states_index].sum()
                                param_val = self._instantaneous_transfer(total_being_tranfered,
                                                                         total_avialable, t)

                            self.piecewise_est_param_values[parameter][t] = param_val
                    else:
                        param_val = parameters[parameter]

                    to_cluster = group_transfer['to_cluster']
                    to_vaccine_group = group_transfer['to_vaccine_group']
                    to_index_dict = self.state_index[to_cluster][to_vaccine_group]
                    for state in group_transfer['states']:
                        from_index = from_index_dict[state]
                        to_index = to_index_dict[state]
                        transferring = param_val * y[from_index]
                        y_deltas[from_index] -= transferring
                        y_deltas[to_index] += transferring

        return y_deltas

    def ode(self, y, t, parameters):
        """
        Evaluate the ODE given states (y), time (t) and parameters


        Parameters
        ----------
        y : numpy.array
            State variables.
        t : float
            Time.
        parameters : dict
            Parameter values.

        Returns
        -------
        y_delta: `numpy.ndarray`
            Deltas of state variables at time point t.
        """
        if self.sub_pop_model is None:
            raise AssertionError('sub_pop_model needs to set before simulations can run.')

        sub_pop_model = self.sub_pop_model
        sub_pop_model_arg_names = getargspec(sub_pop_model)[0]
        sub_pop_model_args = {'y': y,
                              'parameters': self._sorting_params(parameters)}
        if self.infectious_states is not None:
            fois = self.calculate_fois(**sub_pop_model_args)

        sub_pop_model_args['t'] = t
        sub_pop_model_args['y_deltas'] = np.zeros(self.num_state)
        for cluster in self.clusters:
            sub_pop_model_args['cluster'] = cluster
            if self.infectious_states is not None:
                sub_pop_model_args['foi'] = fois[cluster]  # force of infection experienced by this specific cluster.

            if len(self.vaccine_groups)==0:
                sub_pop_model_args['y_deltas'] = self.sub_pop_transfer(**sub_pop_model_args)
                sub_pop_model_args['states_index'] = self.state_index[cluster]  # Dictionary of state indexes for this cluster
                sub_pop_model_args = select_dict_items_in_list(sub_pop_model_args, sub_pop_model_arg_names)
                y_deltas = sub_pop_model(**sub_pop_model_args)
            else:
                for vaccine_group in self.vaccine_groups:
                    sub_pop_model_args['vaccine_group'] = vaccine_group
                    sub_pop_model_args['y_deltas'] = self.sub_pop_transfer(**sub_pop_model_args)
                    sub_pop_model_args['states_index'] = self.state_index[cluster][vaccine_group]  # Dictionary of state indexes for this cluster
                    sub_pop_model_args = select_dict_items_in_list(sub_pop_model_args, sub_pop_model_arg_names)
                    y_deltas = sub_pop_model(**sub_pop_model_args)

        return y_deltas

    def _check_string_in_list_strings(self, string, list_strings):
        if not isinstance(string, str):
            raise TypeError(str(string) + ' should be of type string.')

        check_list = eval('self.' + list_strings)
        if string not in check_list:
            raise ValueError(string + ' is not one of the predefined model ' + list_strings + ': ' +
                             ','.join(check_list[:-1]) + ' and ' + check_list[:-1] + '.')

    def _sorting_states(self):
        """
        Creates many instance attributes for dealing with the states when class is initialised.

        Attributes Created
        ------------------
        all_states_index : dictionary
            Keys are all the states values are the associated indexes for use with numpy.arrays.
        state_index : 3 level nested dictionary.
            First level: keys are the cluster names values are another dictionary.
                Second level: keys are the vaccine_group names and values are another dictionary.
                    Third level: keys are the states and values (ints) are the associated indexes for use with
                                 numpy.arrays.
        infected_states_index_list : list of ints
            A list of the indexes of infected states.
        hospitalised_states_index_list : list of ints
            A list of the indexes of hospitalised states.
        infectious_symptomatic_indexes : 2 level nested dictionary.
            First level: keys are the cluster names values are another dictionary.
                Second level: keys are the vaccine_group names and values is a list of indexes for infectious and
                              symptomatic states.
        infectious_asymptomatic_indexes : : 2 level nested dictionary.
            First level: keys are the cluster names values are another dictionary.
                Second level: keys are the vaccine_group names and values is a list of indexes for infectious and
                              asymptomatic states.

        infectious_and_symptomatic_states : list of stings
            A list of infectious and symptomatic states.
        infectious_and_asymptomatic_states : list of stings
            A list of infectious and asymptomatic states.
        isolating_and_symptomatic_states : list of stings
            A list of isolating and symptomatic states.
        isolating_and_asymptomatic_states : list of stings
            A list of isolating and asymptomatic states.
        num_state : int
            Total number of states in model.

        Returns
        -------
        Nothing
        """
        self.infectious_and_symptomatic_states = [state for state in self.infectious_states
                                                  if state in self.symptomatic_states and
                                                  state not in self.isolating_states]
        self.infectious_and_asymptomatic_states = [state for state in self.infectious_states
                                                   if state not in self.symptomatic_states and
                                                   state not in self.isolating_states]
        self.isolating_and_symptomatic_states = [state for state in self.infectious_states
                                                 if state in self.symptomatic_states and
                                                 state in self.isolating_states]
        self.isolating_and_asymptomatic_states = [state for state in self.infectious_states
                                                  if state not in self.symptomatic_states and
                                                  state in self.isolating_states]
        self.all_states_index = {}
        self.state_index = {}
        self.infectious_symptomatic_indexes = {}
        self.infectious_asymptomatic_indexes = {}
        self.isolating_symptomatic_indexes = {}
        self.isolating_asymptomatic_indexes = {}
        self.infected_states_index_list = []
        self.hospitalised_states_index_list = []
        # populating index dictionaries
        index = 0
        for cluster in self.clusters:
            self.state_index[cluster] = {}
            self.infectious_symptomatic_indexes[cluster] = []
            self.infectious_asymptomatic_indexes[cluster] = []
            self.isolating_symptomatic_indexes[cluster] = []
            self.isolating_asymptomatic_indexes[cluster] = []
            for vaccine_group in self.vaccine_groups:
                self.state_index[cluster][vaccine_group] = {}
                for state in self.states:
                    self.all_states_index[state + '_' + cluster + '_' + vaccine_group] = index
                    self.state_index[cluster][vaccine_group][state] = index
                    if state in self.infectious_and_symptomatic_states:
                        self.infectious_symptomatic_indexes[cluster].append(index)
                    if state in self.infectious_and_asymptomatic_states:
                        self.infectious_asymptomatic_indexes[cluster].append(index)
                    if state in self.isolating_and_symptomatic_states:
                        self.isolating_symptomatic_indexes[cluster].append(index)
                    if state in self.isolating_and_asymptomatic_states:
                        self.isolating_asymptomatic_indexes[cluster].append(index)
                    if state in self.infected_states:
                        self.infected_states_index_list.append(index)
                    if state in self.hospitalised_states:
                        self.hospitalised_states_index_list.append(index)
                    index += 1

        self.state_index['observed_states'] = {}
        for state in self.observed_states:
            self.all_states_index[state] = index
            self.state_index['observed_states'][state] = index
            index += 1

        self.num_state = index
        for transfer_info in self.sub_pop_transition_params_dict.values():
            for transfer_info_entry in transfer_info:
                cluster = transfer_info_entry['from_cluster']
                vaccine_group = transfer_info_entry['from_vaccine_group']
                states_dict = self.state_index[cluster][vaccine_group]
                to_cluster = transfer_info_entry['to_cluster']
                to_vaccine_group = transfer_info_entry['to_vaccine_group']
                to_states_dict = self.state_index[to_cluster][to_vaccine_group]
                state_selection = transfer_info_entry['states']
                if state_selection == 'all':
                    transfer_info_entry['from_index'] = [states_dict.values()]
                    transfer_info_entry['to_index'] = [to_states_dict.values()]
                else:
                    transfer_info_entry['from_index'] = [states_dict[state] for state in state_selection]
                    transfer_info_entry['to_index'] = [to_states_dict[state] for state in state_selection]

    def get_clusters_indexes(self, clusters):
        """
        Returns a list of the indexes for all the states in the clusters given.

        Parameters
        ----------
        clusters : list of strings or single string
            A list of clusters.

        Returns
        -------
        indexes : list of ints
            A list of indexes.
        """
        indexes = []
        for cluster in clusters:
            indexes += _nested_dict_values(self.state_index[cluster])
        return indexes

    def get_vaccine_group_indexes(self, vaccine_groups):
        """
        Returns a list of the indexes for all the states in the vaccine groups given.

        Parameters
        ----------
        vaccine_groups : list of strings or single string
            A list of vaccination groups.

        Returns
        -------
        indexes : list of ints
            A list of indexes.
        """
        if isinstance(vaccine_groups, str):
            vaccine_groups = [vaccine_groups]
        indexes = []
        for cluster in self.clusters:
            for vaccine_group, sub_dict in self.state_index[cluster].items():
                if vaccine_group in vaccine_groups:
                    indexes += [sub_dict.values()]
        return indexes

    def calculate_fois(self, y, parameters):
        """
        Calculates the Forces of Infection (FOI) given variables in y and parameters.

        Parameters
        ----------
        y : numpy.array
            Values of state variables at current time.
        parameters : dictionary {keys are strings: values are numeric}
            Dictionary of parameter values used in calculating derivative.

        Returns
        -------
        If transmission is cluster specific:
            fois : dictionary {keys are strings: values are numeric}
                Dictionary of the FOIs experienced by each cluster.
        Else:
            foi : Numeric
                FOI experienced by all the population.

        """
        if self.asymptomatic_transmission_modifier is not None:
            asymptomatic_transmission_modifier = parameters[self.asymptomatic_transmission_modifier]
        else:
            asymptomatic_transmission_modifier = 1

        if self.transmission_cluster_specific:
            fois = {}
            for cluster_i in self.clusters:
                foi = 0
                contactable_population = parameters[self.population_term + '_' + cluster_i]
                for cluster_j in self.clusters:
                    if self.isolation_modifier is not None:
                        if self.isolation_cluster_specific:
                            isolation_modifier = parameters[self.isolation_modifier + '_' + cluster_j]
                        else:
                            isolation_modifier = parameters[self.isolation_modifier]
                    else:
                        isolation_modifier = 1
                    beta = parameters[self.transmission_term + '_' + cluster_i + '_' + cluster_j]

                    if beta > 0:
                        total_asymptomatic = (asymptomatic_transmission_modifier *
                                              y[self.infectious_asymptomatic_indexes[cluster_j]].sum())
                        total_symptomatic = y[self.infectious_symptomatic_indexes[cluster_j]].sum()
                        total_isolating_asymptomatic = (isolation_modifier * asymptomatic_transmission_modifier *
                                                        y[self.isolating_asymptomatic_indexes[cluster_j]].sum())
                        total_isolating_symptomatic = (isolation_modifier *
                                                       y[self.isolating_symptomatic_indexes[cluster_j]].sum())
                        full_contribution = sum([total_asymptomatic, total_symptomatic,
                                                 total_isolating_asymptomatic, total_isolating_symptomatic])

                        foi += beta * full_contribution / contactable_population

                fois[cluster_i] = foi
            return fois
        else:
            if self.isolation_modifier is not None:
                isolation_modifier = parameters[self.isolation_modifier]
            else:
                isolation_modifier = 1
            infectious_symptomatic_indexes = _unionise_dict_of_lists(self.infectious_symptomatic_indexes)
            infectious_and_asymptomatic_indexes = _unionise_dict_of_lists(self.infectious_asymptomatic_indexes)
            isolating_asymptomatic_indexes = _unionise_dict_of_lists(self.isolating_asymptomatic_indexes)
            isolating_symptomatic_indexes = _unionise_dict_of_lists(self.isolating_symptomatic_indexes)
            total_asymptomatic = asymptomatic_transmission_modifier * y[infectious_and_asymptomatic_indexes].sum()
            total_symptomatic = y[infectious_symptomatic_indexes].sum()
            total_isolating_asymptomatic = (isolation_modifier * asymptomatic_transmission_modifier
                                            * y[isolating_asymptomatic_indexes].sum())
            total_isolating_symptomatic = isolation_modifier * y[isolating_symptomatic_indexes].sum()
            full_contribution = sum([total_asymptomatic, total_symptomatic,
                                     total_isolating_asymptomatic, total_isolating_symptomatic])
            foi = parameters[self.transmission_term] * full_contribution / parameters[self.population_term]
            return foi

    def _instantaneous_transfer(self, population_transitioning, population, t=None):
        """
        Calculate instantaneous rate needed to reduce population by population transitioning to another compartment.

        Parameters
        ----------
        population_transitioning : float
            Population moving between compartments.
        population : float
            Population from which transition is taking place.
        t : float
            Time at t. Used in generating an error specific to simulating models using this class.

        Returns
        -------
        float
            Instantaneous rate of change for the time t.
        """
        if population_transitioning > population:
            error_msg = "population_transitioning (" + str(
                population_transitioning) + ") is greater than population (" + str(population)
            if t is None:
                error_msg += ').'
            else:
                error_msg += '), at time ' + str(t) + ').'
            raise ValueError(error_msg)
        if population_transitioning == 0 and population == 0:
            return 0
        else:
            proportion_by_t = population_transitioning / population
            return -np.log(1 - proportion_by_t)

    @property
    def sub_pop_model(self):
        return self._sub_pop_model

    @sub_pop_model.setter
    def sub_pop_model(self, sub_pop_model):
        if not callable(sub_pop_model):
            raise TypeError('sub_pop_model should be a callable function.')
        self._sub_pop_model = sub_pop_model

    @property
    def parameters(self):
        """
        Returns
        -------
        Dictionary
            A dictionary of parameter values.

        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        """
        Set parameter values for simulations.

        Parameters
        ----------
        parameters : dictionary
            Parameter values.

        Returns
        -------
        Nothing
        """
        if not isinstance(parameters, dict):
            raise TypeError('Currently non non_piecewise_params must be entered as a dict.')
        # we assume that the key of the dictionary is a string and
        # the value can be a single value or a distribution

        for param_name, value in parameters.items():
            if param_name not in self.all_parameters:
                raise ValueError(param_name + ' is not a name given to a parameter for this model.')
            if param_name in self.params_estimated_via_piecewise_method:
                raise AssertionError(param_name + ' was set as a parameter to be estimated via piecewise estimiation ' +
                                     'at the initialization of this model.')
            if not isinstance(value, Number):
                raise TypeError(param_name + ' is not a number type.')
        params_not_given = [param for param in self.all_parameters
                            if param not in
                            list(parameters.keys()) + self.params_estimated_via_piecewise_method]
        if params_not_given:
            raise Exception(', '.join(params_not_given) +
                            " are/is missing from parameters for model (see self.all_parameters).")
        # this must be sorted alphanumerically.
        self._parameters = {key: value for key, value in sorted(parameters.items())}

    def _check_all_params_represented(self):
        """
        Checks all parameters have been given. Gives error if not.

        Returns
        -------
        Nothing
        """
        check_list = (list(self.parameters.keys()) +
                      self.params_estimated_via_piecewise_method)
        for param in self.all_parameters:
            if param not in check_list:
                raise AssertionError(param +
                                     'has not been assigned a value or set up for piecewise estimation.')

    def _sorting_params(self, parameters):
        """
        Sorts parameters values given as list into a dictionary.

        Parameters
        ----------
        parameters : list of Numeric types
            List of parameter values.

        Returns
        -------
        Dictionary keys are strings used in naming parameters values are the corresponding values.
        """
        return dict(zip(self.non_piece_wise_params_names, parameters))

    def integrate(self, x0, t, full_output=False, called_in_fitting=False, **kwargs_to_pass_to_odeint):
        """
        Simulate model via integration using initial values x0 of time range t.

        A wrapper on top of :mod:`odeint <scipy.integrate.odeint>`
        Modified method from the pygom method `DeterministicOde <pygom.model.DeterministicOde>`.
        
        Parameters
        ----------
        x0 : array like
            Initial values of states.
        t : array like
            Timeframe over which model is to be simulated.
        full_output : bool, optional
            If additional information from the integration is required
        called_in_fitting : bool, optional
            If method is being called in fitting.
        kwargs_to_pass_to_odeint : dictionary
            Key word arguments to pass to scipy.integrate.odeint.

        Returns
        -------
        solution: pandas.DataFrame
            Multi-index columns are  clusters by vaccine groups by states.
        """
        if not called_in_fitting:  # If fitting model these checks should be done in fitting method.
            # This would avoid unnecessary error checks.
            self._check_all_params_represented()
        self.piecewise_est_param_values = {param: {} for param in self.params_estimated_via_piecewise_method}
        # INTEGRATE!!! (shout it out loud, in Dalek voice)
        # determine the number of output we want
        args = tuple(self.parameters.values())
        # The if else statement below is for use with DOK matrix version of the Jacobian see sileneced section below.
        # if self.dok_jacobian is None: # May or may not of defined the models Jacobian
        #     solution, output = scipy.integrate.odeint(self.ode,
        #                                               x0, t, args=args,
        #                                               full_output=True,
        #                                               **kwargs_to_pass_to_odeint)
        # else:
        #     solution, output = scipy.integrate.odeint(self.ode,
        #                                               x0, t, args=args,
        #                                               Dfun=self.jacobian,
        #                                               full_output=True,
        #                                               **kwargs_to_pass_to_odeint)
        solution, output = scipy.integrate.odeint(self.ode,
                                                  x0, t, args=args,
                                                  full_output=True,
                                                  **kwargs_to_pass_to_odeint)
        solution = self.results_array_to_df(solution, t)
        if full_output:
            # have both
            return solution, output
        else:
            return solution

    def results_array_to_df(self, results, t):
        """
        Converts array of results into a dataframe with multi-index columns reflecting meta-population structure.

        Parameters
        ----------
        results : np.array
            Results from simulations of model.
        t : np.array
            Time over which model was simulated.

        Returns
        -------
        results_df : pandas.DataFrame
            Results with multi-index columns reflecting meta-population structure.

        """
        state_index = self.state_index
        multi_columns = []
        for cluster, sub_dict in state_index.items():
            if cluster != 'observed_states':
                for vaccine_group, state_dict in sub_dict.items():
                    for state in state_dict.keys():
                        multi_columns.append((cluster, vaccine_group, state))
            else:
                for state in sub_dict.keys():
                    multi_columns.append((cluster, None, state))
        results_df = pd.DataFrame(results, index=t)
        results_df.columns = pd.MultiIndex.from_tuples(multi_columns)
        return results_df
