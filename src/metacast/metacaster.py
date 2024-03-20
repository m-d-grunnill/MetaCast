"""
Creation:
    Author: Martin Grunnill
    Date: 2024/03/08
Description: 
    Contains class for setting up and simulating multidimensional metapopulation models.

Classes
-------
MetaCaster
    Class for setting up and simulating multidimensional metapopulation models.

    Can simulate models on its own if given model_attributes and subpop_model arguments at initialisation.
    Alternatively these can be defined as attributes of MetaCaster subclass.

Notes
-----
I have tried to use notation find in:
    Keeling, M. J., & Rohani, P. (2008). Metapopulations. In Modeling Infectious Diseases in Humans and Animals
    (pp. 237–240). Princeton University Press.

"""
import copy
from inspect import getfullargspec
from collections.abc import Iterable
import itertools
import numpy as np
import pandas as pd
from numbers import Number
import scipy


def _check_string_in_list_strings(string, list_strings):
    if not isinstance(string, str):
        raise TypeError(str(string) + ' should be of type string.')

    check_list = eval('self.' + list_strings)
    if string not in check_list:
        raise ValueError(string + ' is not one of the predefined model ' + list_strings + ': ' +
                         ','.join(check_list[:-1]) + ' and ' + check_list[:-1] + '.')


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


def _collection_of_strings(object):
    if isinstance(object, (dict, str)):
        return False
    elif any(not isinstance(element, str) for element in object):
        return False
    else:
        return True


def _is_set_like_of_strings(object):
    if not _is_iterable_of_unique_elements(object):
        return False
    else:
        return _collection_of_strings(object)


def select_dict_items_in_list(dictionary, lst):
    return {key: value for key, value in dictionary.items() if key in lst}


class MetaCaster:
    """
    Class for setting up and simulating multidimensional metapopulation models.

    Can simulate models on its own if given model_attributes and subpop_model arguments at initialisation.
    Alternatively these can be defined as attributes of MetaCaster subclass.

    Parameters & Attributes
    -----------------------
    dimensions : int, collection of unique strings, list/tuple of ints, list/tuple of collections of unique strings OR list/tuple of dictionaries (transfer dictionaries)
        If int :
            This creates a one dimension metapopulation structure with range(scaffold) used to label subpopulations
            in dimensions attribute.
        If list/tuple/set of unique strings:
            This creates a one dimension metapopulation structure with entries used to label subpopulations
             in dimensions attribute.
        If list/tuple of ints:
             This creates a multidimensional metapopulation structure with range(each int entry) used to generate
              labels on an axis of the subpopulations in dimensions attribute.
        If list/tuple of list/tuple/set of unique strings:
         This creates a multidimensional metapopulation structure with each sub-list/tuple/set entries used as
          labels in am axis of dimensions attribute.
        If  list/tuple of dictionaries (transfer dictionaries):
            Transfer dictionaries outlines the transfer of one subpopulation to another subpopulation.
            Each transfer dictionary must have the key values pairs:
                from_coordinates: string/int or list/tuple of strings/ints
                    Subpopulation coordinates from which hosts are leaving. All of these entries should be of the same
                     length.
                to_coordinates: string/int or list/tuple of strings/ints
                    Subpopulation coordinates from which hosts are leaving.All of these entries should be of the same
                     length and the same length as the from_coordinates entries.
                states: list of strings or string
                    Host states which will transition between subpopulations. Single entry of 'all' value
                    means all the available model states transition between subpopulations. Alternatively a list of
                    specific states can be given.
                parameter : string
                    Name given to parameter that is responsible for flow of hosts transferring between subpopulations.

    subpop_model : callable function/class method
        Method used to model subpopulation:
            Required arguments: 'y', 'y_deltas', 'parameters' and 'states_index'.
            Possible additional arguments: 'coordinates', 'subpop_suffix', 'foi' or 't'.
    states : list of strings
        States used in model.
    observed_states : string or list of strings , optional
        Observed states. Useful for obtaining results or fitting (e.g. Cumulative incidence).
    infected_states : list of strings , optional
        Infected states (not necessarily infectious).
    infectious_states : string or list of strings , optional if not given value(s) for infected_states used
        Infectious states. These states contribute to force of infection. If None no force of infection is
        caluculated when running model (within ode method).
    symptomatic_states : string or list of strings , optional if not given value(s) for infected_states used
        Symptomatic states. NOTE any state in the list self.infectious_states but NOT in this list has its transmission
        modified by self.asymptomatic_transmission_modifier (see method calculate_fois).
    transmission_term_prefix : string, default 'beta'
        Prefix of subpopulation term used in calculating forces of infection, default term is 'beta' (see method
        calculate_fois).
    population_term_prefix : string, default 'N'
        Prefix of subpopulation term used as denominator in calculating force of infection (see method calculate_fois).
    subpop_interaction_prefix : string default is 'rho'
        Prefix of subpopulation term used to denote level of interaction between subpopulations when calculating force
         of infection (see method calculate_fois).
    asymptomatic_transmission_modifier : string or None
        Factor used to modify transmission from infectious but asymptomatic states. If None a factor of 1 is used in the
        asymptomatic_transmission_modifier's place when using method calculating_fois.
    universal_params : list of strings
        A list of all the parameters that are NOT:
        - directly to do with transmission
        - subpopulation specific (sub_pop_params).
    subpop_params : list of strings, or if axis/dimensions > 1 a dictionary {int: lists of strings}
         A list of all the parameters that axis/dimensions specific but NOT directly to do with transmission.
    foi_population_focus : string or None (default None)
        If None a subpopulations force of infection is not divided by  the total population of a subpopulation.
        If 'i' a subpopulations force of infection is divided by the total population of subpopulation 'i',
        i.e. its own population (the population that is being transmitted to).
        If 'j' a subpopulations force of infection for each interacting subpopulation is divided by the total
        population of subpopulation 'j', i.e. the subpopulation it is interacting with (the population that is being
        transmitted from).

    Attributes
    ----------
    parameter_names : list of strings
        The names of all the parameters.
    population_terms : list of strings
        Subpopulation population terms used in calculating force of infection (see method calculate_fois).
    transmission_terms :  list of strings
        The transmission terms for each subpopulation (see method calculate_fois).
    subpop_interaction_terms : list of strings
        The terms for each interaction between subpopulations used when calculating force of infection
        (see method calculate_fois).
    total_subpops : int
        Total number of subpopulations.
    total_parameters : int
        Total number of parameters in model.
    dimensions : list of sets of ints/strings
        Dimensions of metapopulation.
    subpop_transfer_dict : dictionary {tuple of ints or strings: list of dictionaries}
        Each entry outlines all the outflows from a subpopulation to other subpopulation. For use with
        subpop_transfer method.
    subpop_transition_params_dict : dictionary {string: list of dictionaries}
        Each entry outlines parameters responsible for flows between subpopulations.
    all_states_index : dictionary
        Keys are all the states values are the associated indexes for use with numpy.arrays.
    state_index : dictionary {tuple of strings or ints: {string : int}}
        First level: keys are the subpopulation coordinates are another dictionary.
            Second level: keys are the states and values (ints) are the associated indexes for use with
                          numpy.arrays.
    infected_states_indexes : dictionary {tuple of strings or ints: [int]}
        Keys are the subpopulation coordinates values are a list of indexes for infected states.
    exposed_states_indexes : dictionary {tuple of strings or ints: [int]}
        Keys are the subpopulation coordinates values are a list of indexes for infected but not infectious states.
    infectious_symptomatic_indexes : dictionary {tuple of strings or ints: [int]}
        Keys are the subpopulation coordinates values are a list of indexes for infectious and symptomatic states.
    infectious_asymptomatic_indexes : dictionary {tuple of strings or ints: [int]}
        Keys are the subpopulation coordinates values are a list of indexes for infectious and asymptomatic states.
    infectious_and_symptomatic_states : list of stings
        A list of infectious and symptomatic states.
    infectious_and_asymptomatic_states : list of stings
        A list of infectious and asymptomatic states.
    total_states : int
        Total number of states in model.
    subpop_coordinates : list [tuples of ints/strings]
        Coordinates of all subpopulations.
    subpop_suffixes : list of stings
        Suffixes to appended to parameters specific to subpopulations.

    Methods
    -------
    set_structure(self, scaffold, foi_population_focus=None)
        Set metapopulation structure using scaffold.
    ode(self, y, t, *parameters):
        Evaluate the ODE given states (y), time (t) and parameters
    integrate(x0, t, full_output=False, **kwargs_to_pass_to_odeint)
        Simulate model via integration using initial values x0 of time range t.
        A wrapper on top of :mod:`odeint <scipy.integrate.odeint>` giving ode method to odeint.
    get_state_index_dict_of_coordinate(self, coordinate, axis=0):
        Fetch state index dictionaries for given coordinate on an axis.
    get_indexes_of_coordinate(self, coordinate, axis=0):
        Fetch a list of indices for all states for given coordinate on axis.

    Notes
    -----
    I have tried to use notation find in:
        Keeling, M. J., & Rohani, P. (2008). Metapopulations. In Modeling Infectious Diseases in Humans and Animals
        (pp. 237–240). Princeton University Press.
    """
    _dimensions = []
    _subpop_model = None
    _foi_population_focus = None

    def __init__(self,
                 dimensions,
                 subpop_model,
                 states,
                 infected_states=None,
                 infectious_states=None,
                 symptomatic_states=None,
                 observed_states=None,
                 universal_params=None,
                 subpop_params=None,
                 foi_population_focus=None,
                 transmission_term_prefix='beta',
                 subpop_interaction_prefix='rho',
                 population_term_prefix='N',
                 asymptomatic_transmission_modifier=None
                 ):
        self.subpop_model = subpop_model
        self.foi_population_focus = foi_population_focus
        if not _is_set_like_of_strings(states):
            raise TypeError("states should be a collection of unique strings.")
        self.states = states

        if isinstance(infected_states,str):
            self.infected_states = [infected_states]
        elif infected_states is None:
            self.infected_states = []
        elif _is_set_like_of_strings(infected_states):
            self.infected_states = infected_states
        else:
            raise TypeError("If not None infected_states should be a collection of unique strings.")

        if isinstance(infectious_states, str):
            self.infectious_states = [infectious_states]
        elif infectious_states is None:
            self.infectious_states = infected_states
        elif _is_set_like_of_strings(infectious_states):
            self.infectious_states = infectious_states
        else:
            raise TypeError("If not None infectious_states should be a collection of unique strings.")

        if isinstance(symptomatic_states, str):
            self.symptomatic_states = [symptomatic_states]
        elif symptomatic_states is None:
            self.symptomatic_states = infected_states
        elif _is_set_like_of_strings(symptomatic_states):
            self.symptomatic_states = symptomatic_states
        else:
            raise TypeError("If not None symptomatic_states should be a collection of unique strings.")

        if isinstance(observed_states, str):
            self.observed_states = [observed_states]
        elif observed_states is None:
            self.observed_states = []
        elif _is_set_like_of_strings(observed_states):
            self.observed_states = observed_states
        else:
            raise TypeError("If not None observed_states should be a collection of unique strings.")

        if self.states is None:
            raise AssertionError('Model attribute "states" needs to be defined at initialisation of MetaCaster,' +
                                 ' or given as an attribute of a child class of MetaCaster.')

        if universal_params is None and subpop_params is None:
            raise AssertionError('At least one of the model attributes "universal_params" or "subpop_params" ' +
                                 'needs to be defined at initialisation of MetaCaster,' +
                                 ' or given as an attribute of a child class of MetaCaster.')

        if universal_params is None:
            self.universal_params = []
        elif _is_set_like_of_strings(universal_params):
            self.universal_params = universal_params
        else:
            raise TypeError("If not None universal_params should be a collection of unique strings.")
        if subpop_params is None:
            self.subpop_params = []
        elif _is_set_like_of_strings(subpop_params):
            self.subpop_params = subpop_params
        else:
            raise TypeError("If not None subpop_params should be a collection of unique strings.")

        if not isinstance(transmission_term_prefix, str):
            raise TypeError('transmission_term_prefix should be a string.')
        if transmission_term_prefix in self.subpop_params:
            raise ValueError('transmission_term_prefix should not be one of the values in subpop_params.')
        if transmission_term_prefix in self.universal_params:
            raise ValueError('transmission_term_prefix should not be one of the values in universal_params.')
        self.transmission_term_prefix = transmission_term_prefix

        if not isinstance(subpop_interaction_prefix, str):
            raise TypeError('subpop_interaction_prefix should be a string.')
        if subpop_interaction_prefix in self.subpop_params:
            raise ValueError('subpop_interaction_prefix should not be one of the values in subpop_params.')
        if subpop_interaction_prefix in self.universal_params:
            raise ValueError('subpop_interaction_prefix should not be one of the values in universal_params.')
        self.subpop_interaction_prefix = subpop_interaction_prefix

        if not isinstance(population_term_prefix, str):
            raise TypeError('population_term_prefix should be a string.')
        if population_term_prefix in self.subpop_params:
            raise ValueError('population_term_prefix should not be one of the values in subpop_params.')
        if population_term_prefix in self.universal_params:
            raise ValueError('population_term_prefix should not be one of the values in universal_params.')
        self.population_term_prefix = population_term_prefix

        if asymptomatic_transmission_modifier is not None:
            if not isinstance(asymptomatic_transmission_modifier, str):
                raise TypeError('asymptomatic_transmission_modifier should be a string or None.')
            if asymptomatic_transmission_modifier in self.subpop_params:
                raise ValueError(
                    'If not None asymptomatic_transmission_modifier should not be one of the values in subpop_params.')
            if asymptomatic_transmission_modifier in self.universal_params:
                raise ValueError(
                    'If not None asymptomatic_transmission_modifier should not be one of the values in universal_params.')
        self.asymptomatic_transmission_modifier = asymptomatic_transmission_modifier
        self.dimensions = dimensions

    def subpop_transfer(self, y, y_deltas, t, from_coordinates, parameters):
        """
        Calculates the transfers of people from a subpopulation to all other subpopulations.

        Parameters
        ----------
        y : numpy.array
            Values of variables at time t.
        y_deltas : numpy.Array
            Store of delta (derivative at t) of variables in y which this method adds/subtracts to.
        t : float
            Time t for which derivative is being calculated.
        from_coordinates : tuple or list of strings
            Coordinates for subpopulation being transferred.
        parameters : dictionary {keys are strings: values are numeric}
            Dictionary of parameter values used in calculating derivative.

        Returns
        -------
        y_deltas : numpy.array
            Store of delta (derivative) of variables in y which this method adds/subtracts to.

        """
        if from_coordinates in self.subpop_transfer_dict:
            from_state_index_dict = self.state_index[from_coordinates]
            for group_transfer in self.subpop_transfer_dict[from_coordinates]:
                to_coordinates = group_transfer['to_coordinates']
                to_state_index_dict = self.state_index[to_coordinates]
                for state in group_transfer['states']:
                    from_index = from_state_index_dict[state]
                    to_index = to_state_index_dict[state]
                    transferring = parameters[group_transfer['parameter']] * y[from_index]
                    y_deltas[from_index] -= transferring
                    y_deltas[to_index] += transferring

        return y_deltas

    def ode(self, y, t, *parameters):
        """
        Evaluate the ODE given states (y), time (t) and parameters

        Parameters
        ----------
        y : numpy.array
            State variables.
        t : float
            Time.
        parameters : floats
            Parameter values. **NOTE* these values must be given in the same order as the instance attribute
            parameter_names.

        Returns
        -------
        y_delta: `numpy.ndarray`
            Deltas of state variables at time point t.
        """
        if self.subpop_model is None:
            raise AssertionError('subpop_model needs to set before simulations can run.')

        subpop_model = self.subpop_model
        subpop_model_arg_names = getfullargspec(subpop_model)[0]
        parameters = dict(zip(self.parameter_names, parameters))
        if self.infectious_states:
            fois = self.calculate_fois(y, parameters, t)
        y_deltas = np.zeros(self.total_states)
        for coordinates in self.subpop_coordinates:
            y_deltas = self.subpop_transfer(y, y_deltas, t, coordinates, parameters)
            if self.infectious_states is not None:
                foi = fois[coordinates]  # force of infection experienced by this specific cluster.

            subpop_suffix = self.coordinates_to_subpop_suffix(coordinates)
            subpop_model_args = {'y': y,
                                 'y_deltas': y_deltas,
                                 'coordinates': coordinates,
                                 'subpop_suffix': subpop_suffix,
                                 'parameters': parameters,
                                 'states_index': self.state_index[coordinates],
                                 't': t,
                                 'foi': foi
                                 }
            subpop_model_args = select_dict_items_in_list(subpop_model_args, subpop_model_arg_names)
            y_deltas = subpop_model(**subpop_model_args)

        return y_deltas

    def calculate_fois(self, y, parameters, t):
        """
        Calculates the Forces of Infection (FOIs) experienced by each subpopulation.

        Parameters
        ----------
        y : numpy.array
            Values of state variables at current time.
        parameters : dictionary {strings: numeric}
            Dictionary of parameter values used in calculating derivative.
        t : float
            Time.

        Returns
        -------
        fois : dictionary {tupple of strings or ints: float}
            Dictionary of the FOIs experienced at each coordinate.

        """
        if self.asymptomatic_transmission_modifier is not None:
            asymptomatic_transmission_modifier = parameters[self.asymptomatic_transmission_modifier]
        else:
            asymptomatic_transmission_modifier = 1

        fois = {coordinates: 0 for coordinates in self.subpop_coordinates}
        for coordinates_i in self.subpop_coordinates:
            subpop_prefix_i = self.coordinates_to_subpop_suffix(coordinates_i)
            beta = parameters[self.transmission_term_prefix + subpop_prefix_i]
            interactions_with_infectious = [self._subpop_infected_interaction(y,
                                                                              coordinates_i,
                                                                              coordinates_j,
                                                                              parameters,
                                                                              asymptomatic_transmission_modifier,
                                                                              t)
                                            for coordinates_j in self.subpop_coordinates]

            interactions_with_infectious = sum(interactions_with_infectious)
            fois[coordinates_i] += beta * interactions_with_infectious

        return fois

    def _subpop_infected_interaction(self, y, coordinates_i, coordinates_j, parameters,
                                     asymptomatic_transmission_modifier, t):
        total_asymptomatic = (asymptomatic_transmission_modifier *
                              y[self.infectious_asymptomatic_indexes[coordinates_j]].sum())
        total_symptomatic = y[self.infectious_symptomatic_indexes[coordinates_j]].sum()
        subpop_prefix_i = self.coordinates_to_subpop_suffix(coordinates_i)
        subpop_prefix_j = self.coordinates_to_subpop_suffix(coordinates_j)
        contribution = parameters[self.subpop_interaction_prefix + subpop_prefix_i + subpop_prefix_j] \
                       * sum([total_asymptomatic, total_symptomatic])
        if self._foi_population_focus == 'i':
            contactable_population = parameters[self.population_term_prefix + '_[' + coordinates_i + ']']
            if callable(contactable_population):
                contactable_population = contactable_population(model=self,
                                                                y=y,
                                                                coordinates=coordinates_i,
                                                                parameters=parameters,
                                                                t=t)
            contribution = contribution / contactable_population

        if self._foi_population_focus == 'j':
            contactable_population = parameters[self.population_term_prefix + '_[' + coordinates_j + ']']
            if callable(contactable_population):
                contactable_population = contactable_population(model=self,
                                                                y=y,
                                                                coordinates=coordinates_j,
                                                                parameters=parameters,
                                                                t=t)
            contribution = contribution / contactable_population

        return contribution

    def get_state_index_dict_of_coordinate(self, coordinate, axis=0):
        """
        Fetch state index dictionaries for given coordinate on an axis.

        Parameters
        ----------
        coordinate : int or str
            Coordinate of
        axis : int default = 0
            Axis on which coordinate is found.

        Returns
        -------
        nested dict: {tuple of int/str: {str: int}}
        """
        selected_coordinates = [coordinates
                                for coordinates in self.subpop_coordinates
                                if coordinates[axis] == coordinate]
        return {coordinates: sub_dict
                for coordinates, sub_dict in self.state_index.items()
                if coordinates in selected_coordinates}

    def get_indexes_of_coordinate(self, coordinate, axis=0):
        """
        Fetch a list of indices for all states for given coordinate on axis.

        Parameters
        ----------
        coordinate : int or str
            Coordinate of
        axis : int default = 0
            Axis on which coordinate is found.

        Returns
        -------
        list of ints
        """
        selected_state_indexes = self.get_state_index_dict_of_coordinate(self, coordinate, axis=axis)
        return _nested_dict_values(selected_state_indexes)

    @staticmethod
    def coordinates_to_subpop_suffix(coordinates):
        """
        Transform coordinates to subpopulation suffix appended to subpopulation specific parameters.

        Parameters
        ----------
        coordinates : list/tuple of ints or strings
            Coordinate of subpopulation

        Returns
        -------
        string
        """
        if isinstance(coordinates, (list, tuple)):
            if all(isinstance(coordinate, int) for coordinate in coordinates):
                coordinates = (str(coordinate) for coordinate in coordinates)
            if any(not isinstance(coordinate, str) for coordinate in coordinates):
                raise TypeError('All coordinates must be either string or integers, or lists/tuples of those types.')
            return '_[' + ','.join(coordinates) + ']'
        elif isinstance(coordinates, str):
            return '_[' + coordinates + ']'
        elif isinstance(coordinates, int):
            return '_[' + str(coordinates) + ']'
        else:
            raise TypeError('All coordinates must be either string or integers, or lists/tuples of those types.')

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

    def integrate(self, x0, t, full_output=False, **kwargs_to_pass_to_odeint):
        """
        Simulate model via integration using initial values x0 of time range t.

        A wrapper on top of :mod:`odeint <scipy.integrate.odeint>`, giving ode method to odeint.
        Modified method from the pygom method `DeterministicOde <pygom.model.DeterministicOde>`.
        
        Parameters
        ----------
        x0 : array like
            Initial values of states.
        t : array like
            Timeframe over which model is to be simulated.
        full_output : bool, optional
            If additional information from the integration is required
        kwargs_to_pass_to_odeint : dictionary
            Key word arguments to pass to scipy.integrate.odeint.

        Returns
        -------
        solution: pandas.DataFrame
            Multi-index columns are  clusters by vaccine groups by states.
        """
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
        if len(self) == 1:
            multi_columns = [(coordinates, state)
                             for coordinates, sub_dict in self.state_index.items()
                             for state in sub_dict.keys()
                             ]
        else:
            multi_columns = [(','.join(coordinates), state)
                             for coordinates, sub_dict in self.state_index.items()
                             for state in sub_dict.keys()
                             if coordinates != 'observed_states'
                             ]
            multi_columns += [('observed_states', state)
                              for state in self.state_index['observed_states'].keys()
                              ]
        results_df = pd.DataFrame(results, index=t)
        results_df.columns = pd.MultiIndex.from_tuples(multi_columns)
        return results_df

    def __len__(self):
        """
        Number of axis/dimensions of the metapopulation model.

        Returns
        -------
        int
        """
        return len(self._dimensions)

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions):
        """
        Set metapopulation structure using scaffold.

        Parameters
        ----------
        dimensions : int, collection of unique strings, list/tuple of ints, list/tuple of collections of unique strings OR list/tuple of dictionaries (transfer dictionaries)
            If int :
                This creates a one dimension metapopulation structure with range(scaffold) used to label subpopulations
                in dimensions attribute.
            If list/tuple/set of unique strings:
                This creates a one dimension metapopulation structure with entries used to label subpopulations
                 in dimensions attribute.
            If list/tuple of ints:
                 This creates a multidimensional metapopulation structure with range(each int entry) used to generate
                  labels on an axis of the subpopulations in dimensions attribute.
            If list/tuple of list/tuple/set of unique strings:
             This creates a multidimensional metapopulation structure with each sub-list/tuple/set entries used as
              labels in am axis of dimensions attribute.
            If  list/tuple of dictionaries (transfer dictionaries):
                Transfer dictionaries outlines the transfer of one subpopulation to another subpopulation.
                Each transfer dictionary must have the key values pairs:
                    from_coordinates: string/int or list/tuple of strings/ints
                        Subpopulation coordinates from which hosts are leaving. All of these entries should be of the same
                         length.
                    to_coordinates: string/int or list/tuple of strings/ints
                        Subpopulation coordinates from which hosts are leaving.All of these entries should be of the same
                         length and the same length as the from_coordinates entries.
                    states: list of strings or string
                        Host states which will transition between subpopulations. Single entry of 'all' value
                        means all the available model states transition between subpopulations. Alternatively a list of
                        specific states can be given.
                    parameter : string
                        Name given to parameter that is responsible for flow of hosts transferring between subpopulations.

        Returns
        -------
        None

        Notes: Attributes Altered
        -------------------------
        parameter_names : list of strings
            The names of all the parameters.
        population_terms : list of strings
            Subpopulation population terms used in calculating force of infection (see method calculate_fois).
        transmission_terms :  list of strings
            The transmission terms for each subpopulation (see method calculate_fois).
        subpop_interaction_terms : list of strings
            The terms for each interaction between subpopulations used when calculating force of infection
            (see method calculate_fois).
        total_subpops : int
            Total number of subpopulations.
        total_parameters : int
            Total number of parameters in model.
        dimensions : list of sets of ints/strings
            Dimensions of metapopulation.
        subpop_transfer_dict : dictionary {tuple of ints or strings: list of dictionaries}
            Each entry outlines all the outflows from a subpopulation to other subpopulation. For use with
            subpop_transfer method.
        subpop_transition_params_dict : dictionary {string: list of dictionaries}
            Each entry outlines parameters responsible for flows between subpopulations.
        all_states_index : dictionary
            Keys are all the states values are the associated indexes for use with numpy.arrays.
        state_index : dictionary {tuple of strings or ints: {string : int}}
            First level: keys are the subpopulation coordinates are another dictionary.
                Second level: keys are the states and values (ints) are the associated indexes for use with
                              numpy.arrays.
        infected_states_indexes : dictionary {tuple of strings or ints: [int]}
            Keys are the subpopulation coordinates values are a list of indexes for infected states.
        exposed_states_indexes : dictionary {tuple of strings or ints: [int]}
            Keys are the subpopulation coordinates values are a list of indexes for infected but not infectious states.
        infectious_symptomatic_indexes : dictionary {tuple of strings or ints: [int]}
            Keys are the subpopulation coordinates values are a list of indexes for infectious and symptomatic states.
        infectious_asymptomatic_indexes : dictionary {tuple of strings or ints: [int]}
            Keys are the subpopulation coordinates values are a list of indexes for infectious and asymptomatic states.
        infectious_and_symptomatic_states : list of stings
            A list of infectious and symptomatic states.
        infectious_and_asymptomatic_states : list of stings
            A list of infectious and asymptomatic states.
        total_states : int
            Total number of states in model.
        subpop_coordinates : list [tuples of ints/strings]
            Coordinates of all subpopulations.
        subpop_suffixes : list of stings
            Suffixes to appended to parameters specific to subpopulations.
        """
        self.parameter_names = set(self.universal_params)
        if self.asymptomatic_transmission_modifier is not None:
            self.parameter_names.add(self.asymptomatic_transmission_modifier)

        #### Using Scaffold to define dimensions ####
        self.subpop_transfer_dict = {}
        self.subpop_transition_params_dict = {}
        if isinstance(dimensions, (list, tuple)) and all(isinstance(item, dict) for item in dimensions):
            for count, group_transfer in enumerate(dimensions):
                if 'from_coordinates' not in group_transfer:
                    raise ValueError(
                        'If scaffold is a list of subpopulation transfer dictionaries it must have a "from_coordinates"' +
                        ' entry in every subpopulation transfer dictionary,' +
                        ' check subpopulation transfer dictionary [' +
                        str(count) +
                        '] of scaffold.')
                from_coordinates = group_transfer['from_coordinates']
                if not isinstance(from_coordinates, (list, tuple)):
                    raise TypeError(
                        'If scaffold is a list of subpopulation transfer dictionaries values for "from_coordinates"' +
                        '  entries should be a list or a tuple,' +
                        ' check subpopulation transfer dictionary [' +
                        str(count) +
                        '] of scaffold.')
                if not (all(isinstance(coordinate, int) for coordinate in from_coordinates) or
                        all(isinstance(coordinate, str) for coordinate in from_coordinates)):
                    raise TypeError(
                        'If scaffold is a list of subpopulation transfer dictionaries values for "from_coordinates"' +
                        '  entries should be a list or a tuple of only strings or integers,' +
                        ' check subpopulation transfer dictionary [' +
                        str(count) +
                        '] of scaffold.')
                if count == 0:
                    self._dimensions = [{coordinate} for coordinate in from_coordinates]
                elif len(from_coordinates) != len(self):
                    raise ValueError('If scaffold is a list of dictionaries all "from_coordinates"' +
                                     ' entries in subpopulation transfer dictionaries should be of the same length' +
                                     ', check subpopulation transfer dictionary [' +
                                     str(count) +
                                     '] of scaffold.')
                else:
                    for axis, coordinate in enumerate(from_coordinates):
                        self._dimensions[axis].add(coordinate)

                if from_coordinates not in self.subpop_transfer_dict:
                    self.subpop_transfer_dict[from_coordinates] = []

                entry = {}
                if 'to_coordinates' not in group_transfer:
                    raise ValueError(
                        'If scaffold is a list of subpopulation transfer dictionaries it must have a "to_coordinates"' +
                        ' entry in every subpopulation transfer dictionary,' +
                        ' check subpopulation transfer dictionary [' +
                        str(count) +
                        '] of scaffold.')
                to_coordinates = group_transfer['to_coordinates']
                if not isinstance(to_coordinates, (list, tuple)):
                    raise TypeError(
                        'If scaffold is a list of subpopulation transfer dictionaries values for "to_coordinates"' +
                        '  entries should be a list or a tuple,' +
                        ' check subpopulation transfer dictionary [' +
                        str(count) +
                        '] of scaffold.')
                if not (all(isinstance(coordinate, int) for coordinate in to_coordinates) or
                        all(isinstance(coordinate, str) for coordinate in to_coordinates)):
                    raise TypeError(
                        'If scaffold is a list of subpopulation transfer dictionaries values for "to_coordinates"' +
                        '  entries should be a list or a tuple of only strings or integers,' +
                        ' check subpopulation transfer dictionary [' +
                        str(count) +
                        '] of scaffold.')
                if len(to_coordinates) != len(self):
                    raise ValueError('If scaffold is a list of dictionaries all "to_coordinates" and' +
                                     ' "from_coordinates" entries in subpopulation transfer dictionaries should be of the' +
                                     ' same length' +
                                     ', check "to_coordinates" subpopulation transfer dictionary [' +
                                     str(count) +
                                     '] of scaffold.')
                for axis, coordinate in enumerate(to_coordinates):
                    self._dimensions[axis].add(coordinate)

                entry['to_coordinates'] = to_coordinates
                if group_transfer['states'] == 'all':
                    entry['states'] = self.states
                else:
                    states = copy.deepcopy(group_transfer['states'])
                    if isinstance(states, str):
                        states = list(states)
                    if not _is_set_like_of_strings(states):
                        raise ValueError('"states" entry in Subpopulation transfer dictionary [' +
                                         str(count) +
                                         '] of scaffold is not a collection of unique items.')
                    for state in states:
                        if state not in self.states:
                            raise ValueError('Subpopulation transfer dictionary [' +
                                             str(count) +
                                             '] of scaffold has an unrecognised state (' + state + ').' +
                                             'All states in listed in a scaffold should be one of those listed in the' +
                                             ' subpopulation model.')
                    entry['states'] = states

                parameter = group_transfer['parameter']
                if not isinstance(parameter, str):
                    raise TypeError('In subpopulation transfer dictionary [' + str(count) +
                                    '] of scaffold ' + str(parameter) + ' should be of type string.')
                entry['parameter'] = parameter
                if parameter not in self.subpop_transition_params_dict:
                    self.subpop_transition_params_dict[parameter] = []
                self.subpop_transition_params_dict[parameter].append({key: value for key, value in
                                                                      group_transfer.items()
                                                                      if key != 'parameter'})
                self.parameter_names.add(parameter)

                accepted_keys = list(entry.keys()) + ['from_coordinates']
                for key in group_transfer.keys():
                    if key not in accepted_keys:
                        raise ValueError('Subpopulation transfer dictionary [' +
                                         str(count) +
                                         '] of scaffold has an unrecognised entry (' + key + ').')

                self.subpop_transfer_dict[from_coordinates].append(entry)
        elif (isinstance(dimensions, (list, tuple)) and
              all(_is_set_like_of_strings(item) for item in dimensions)):
            self._dimensions = [set(item) for item in dimensions]
        elif (isinstance(dimensions, (list, tuple)) and
              all(isinstance(item, int) for item in dimensions)):
            self._dimensions = [set(*range(num)) for num in dimensions]
        elif isinstance(dimensions, (list, tuple)) and _is_set_like_of_strings(dimensions):
            self._dimensions = [set(dimensions)]
        elif isinstance(dimensions, set) and all(isinstance(item, str) for item in dimensions):
            self._dimensions = [dimensions]
        elif isinstance(dimensions, int):
            self._dimensions = [set(*range(int))]
        else:
            raise TypeError('scaffold is not supported.')

        #### Definining States based on new Dimensions ####
        self.infectious_and_symptomatic_states = [state for state in self.infectious_states
                                                  if state in self.symptomatic_states]
        self.infectious_and_asymptomatic_states = [state for state in self.infectious_states
                                                   if state not in self.symptomatic_states]
        self.all_states_index = {}
        self.state_index = {}
        self.infectious_symptomatic_indexes = {}
        self.infectious_asymptomatic_indexes = {}
        self.exposed_states_indexes = {}
        self.infected_states_indexes = {}
        # populating index dictionaries
        index = 0
        self.subpop_coordinates = []
        self.subpop_suffixes = []
        if len(self) == 1:
            axis_equals_1 = True
        else:
            axis_equals_1 = False
        for coordinates in itertools.product(*self._dimensions):
            subpop_suffix = self.coordinates_to_subpop_suffix(coordinates)
            self.subpop_suffixes.append(subpop_suffix)
            if axis_equals_1:
                coordinates = coordinates[0]
            self.subpop_coordinates.append(coordinates)
            self.state_index[coordinates] = {}
            self.infectious_symptomatic_indexes[coordinates] = []
            self.infectious_asymptomatic_indexes[coordinates] = []
            self.exposed_states_indexes[coordinates] = []
            self.infected_states_indexes[coordinates] = []
            for state in self.states:
                all_state_index_key = state + subpop_suffix
                self.all_states_index[all_state_index_key] = index
                self.state_index[coordinates][state] = index
                if state in self.infectious_and_symptomatic_states:
                    self.infectious_symptomatic_indexes[coordinates].append(index)
                if state in self.infectious_and_asymptomatic_states:
                    self.infectious_asymptomatic_indexes[coordinates].append(index)
                if state in self.infected_states:
                    self.infected_states_indexes[coordinates].append(index)
                if state in self.infected_states and state not in self.infectious_states:
                    self.exposed_states_indexes[coordinates].append(index)

                index += 1

        self.state_index['observed_states'] = {}
        for state in self.observed_states:
            self.all_states_index[state] = index
            self.state_index['observed_states'][state] = index
            index += 1

        self.total_states = index
        for transfer_info in self.subpop_transition_params_dict.values():
            for transfer_info_entry in transfer_info:
                from_coordinates = transfer_info_entry['from_coordinates']
                from_states_dict = self.state_index[from_coordinates]
                to_coordinates = transfer_info_entry['to_coordinates']
                to_states_dict = self.state_index[to_coordinates]
                state_selection = transfer_info_entry['states']
                if state_selection == 'all':
                    transfer_info_entry['from_index'] = [from_states_dict.values()]
                    transfer_info_entry['to_index'] = [to_states_dict.values()]
                else:
                    transfer_info_entry['from_index'] = [from_states_dict[state] for state in state_selection]
                    transfer_info_entry['to_index'] = [to_states_dict[state] for state in state_selection]

        #### Defining Parameters based on new Dimensions ####
        self.parameter_names.update([param + subpop_suffix
                                     for param in self.subpop_params
                                     for subpop_suffix in self.subpop_suffixes
                                     ])

        self.transmission_terms = [
            self.transmission_term_prefix + subpop_suffix
            for subpop_suffix in self.subpop_suffixes]
        self.subpop_interaction_terms = [
            self.subpop_interaction_prefix + subpop_suffix_i + subpop_suffix_j
            for subpop_suffix_i in self.subpop_suffixes
            for subpop_suffix_j in self.subpop_suffixes
        ]
        self.parameter_names.update(self.subpop_interaction_terms)
        if self._foi_population_focus is not None:
            self.population_terms = [self.population_term_prefix + subpop_suffix
                                     for subpop_suffix in self.subpop_suffixes]
            self.parameter_names.update(self.population_terms)

        self.parameter_names.update(self.transmission_terms)
        self.total_subpops = len(self.subpop_coordinates)
        self.parameter_names = sorted(self.parameter_names)
        self._parameters = None
        self.total_parameters = len(self.parameter_names)

    @property
    def foi_population_focus(self):
        return self._foi_population_focus

    @foi_population_focus.setter
    def foi_population_focus(self, foi_population_focus):
        if foi_population_focus is not None:
            if foi_population_focus not in ['i', 'j']:
                raise ('population_denominator_in_foi can only be change to "i" or "j" or None.')

        self._foi_population_focus = foi_population_focus

    @property
    def subpop_model(self):
        return self._subpop_model

    @subpop_model.setter
    def subpop_model(self, subpop_model):
        if not callable(subpop_model):
            raise TypeError('subpop_model should be a callable function.')
        subpop_model_arg_names = getfullargspec(subpop_model)[0]
        required_args = ['y', 'y_deltas', 'parameters', 'states_index']
        for arg in required_args:
            if arg not in subpop_model_arg_names:
                raise ValueError(arg + ' is missing from subpop_model function.')

        expected_args = ['y', 'y_deltas', 'parameters', 'states_index', 'coordinates', 'subpop_suffix', 'foi', 't']
        for arg in subpop_model_arg_names:
            if arg not in expected_args:
                raise ValueError(arg + ' in not a supported argument for the subpop_model function. ' +
                                 'Supported arguments include ' + ', '.join(expected_args))
        self._subpop_model = subpop_model

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
        parameters : dictionary {key: Numeric or callable}
            Parameter values. In the case of population_terms these can be callable function/class methods
            which calculate populations at a given time point. If so population_terms must have the arguments 'model',
            'y', 'parameters', 'coordinates', 't'.

        Returns
        -------
        Nothing
        """
        if not isinstance(parameters, dict):
            raise TypeError('Currently parameters must be entered as a dict.')
        # we assume that the key of the dictionary is a string and
        # the value can be a single value or a distribution

        for param_name, value in parameters.items():
            if param_name not in self.parameter_names:
                raise ValueError(param_name + ' is not a name given to a parameter for this model.')
            if callable(value):
                function_arg_names = getfullargspec(value)[0]
                if any(args not in ['model', 'y', 'parameters', 'coordinates', 't'] for args in function_arg_names):
                    raise ValueError(param_name +
                                     " is a function but not a population_term but does not have all of the arguments" +
                                     " 'model', 'y', 'parameters', 'coordinates' or 't'.")
            elif not isinstance(value, Number):
                raise TypeError(param_name + ' should be a number type.')
        params_not_given = [param for param in self.parameter_names
                            if param not in parameters]
        if params_not_given:
            raise Exception(', '.join(params_not_given) +
                            " are/is missing from parameters for model (see self.all_parameters).")
        # this must be sorted in the same order as parameter_names (i.e. alphanumerically).
        self._parameters = {parameter: parameters[parameter] for parameter in self.parameter_names}

    @property
    def shape(self):
        """
        Number of entries in each axis of the metapopulation model.

        Returns
        -------
        tuple of ints
        """
        return (len(axis) for axis in self._dimensions)


if __name__ == "__main__":
    pass
