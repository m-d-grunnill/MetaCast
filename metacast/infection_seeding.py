"""
Creation:
    Author: Martin Grunnill
    Date: 13/09/2022
Description: Classes for Multnomial random draw seeding of infections.

Classes
-------
MultnomialSeeder
    Makes multinomial draws selecting an infectious hosts branch and then state.
    
"""
from numbers import Number
import numpy as np
import math

class _InfectionBranch:
    """
    Makes multinomial draws for selecting which stage of an infection pathway to place infected hosts.
    Calculates normalised weighting of an infection branch's states, based on inverse outflow for states.

    Parameters & Attributes
    -----------------------
    name : string
        Name of branch.
    outflows: dictionary {str or ints: string}
        Keys are name or number given to state. Values are name given to parameter.

    Methods
    -------
    calculate_weighting(parameters)
        Calculate normalised weighting for each state.
    seed_infections(self, n, parameters)
        Make multinomial draw to select infectious stages of this branch to seed infection into.

    """
    def __init__(self, name, outflows):
        if not isinstance(name, str):
            raise TypeError('name argument should be a string.')
        self.name = name
        outflows_err_msg = ('outflows argument should be a dictionary,'+
                            ' with keys being strings or integers and values being string.')
        if not isinstance(outflows,dict):
            raise TypeError(outflows_err_msg)
        if any(not isinstance(key,(int,str)) for key in outflows.keys()):
            raise TypeError(outflows_err_msg)
        if any(not isinstance(value,str) for value in outflows.values()):
            raise TypeError(outflows_err_msg)
        self.outflows = outflows

    def calculate_weighting(self, parameters):
        """
        Calculate normalised weighting for each state.

        Parameters
        ----------
        parameters : dict {str: Number}
            Dictionary of parameter values.

        Returns
        -------
        noramlised_weightings : dict {str: float}
            Dictionary normalised weighting for each state.
        """
        parameters_error = ('parameters argument should be a dictionary,' +
                            ' with keys being strings and values being numbers.')
        if not isinstance(parameters, dict):
            raise TypeError(parameters_error)
        if any(not isinstance(value, Number) for value in parameters.values()):
            raise TypeError(parameters_error)
        if any(not isinstance(key,str) for key in parameters.keys()):
            raise TypeError(parameters_error)

        weightings = {}
        total = 0
        for state, outflow in self.outflows.items():
            weighting = parameters[outflow] ** -1
            weightings[state] = weighting
            total += weighting

        noramlised_weightings = {state: weight/total for state, weight in weightings.items()}

        return noramlised_weightings

    def seed_infections(self, n, parameters, rng=None):
        """
        Make multinomial draw to select infectious stages of this branch to seed infection into.

        Parameters
        ----------
        n : int
            Number of infections to seed.
        parameters : dict {str: Number}
            Dictionary of parameter values.
        rng : numpy random number generator, optional.
            Random number generator to use.

        Returns
        -------
        draw_dict : dict {str: int}
            Keys are states values are number of infections in state.
        """
        if rng is None:
            rng = np.random.default_rng()
        weighting = self.calculate_weighting(parameters)
        pvals = list(weighting.values())
        states = list(weighting.keys())
        draw = rng.multinomial(n=n, pvals=pvals, size=1)
        draw = draw[0]
        draw_dict = {state: draw[index] for index, state in enumerate(states)}
        return draw_dict


class MultnomialSeeder:
    """
    Makes multinomial draws selecting an infectious hosts branch and then state.

    Parameters
    ----------
    branch_info : nested dict
        First level keys are branches (str).
            Second level keys are states (str or ints) and values are names of outflows for states (str).

    Attributes
    ----------
    branches : dict {str: InfectionBranch}
        Infection branches that a host can be placed upon.
    parameters : set of strings
        Parameters (outflows) given in branch_info.

    Methods
    -------
    seed_infections(n, branch_probability, parameters)
        Draw selection of states to place infected hosts.

    """

    def __init__(self, branch_info):
        if not isinstance(branch_info,dict):
            raise TypeError('branch_info should be a dictionary.')
        self.branches = {}
        self.parameters = set()
        self.rng = None
        for branch_name, outflows in branch_info.items():
            self.parameters.update(list(outflows.values()))
            if not isinstance(branch_info, dict):
                raise TypeError('branch_info should be a dictionary of dictionaries.')
            self.branches[branch_name] = _InfectionBranch(branch_name, outflows)

    def set_seed(self,seed):
        """
        Sets random number generator seed.

        Parameters
        ----------
        seed : int (>0)

        """
        self.rng = np.random.default_rng(seed)

    def _seed_branches(self, n, branch_probability):
        """
        Make multinomial draw for which infection branch to place a host.

        Parameters
        ----------
        n : int
            Number of infections to seed.
        branch_probability : dict {string, float}
            Probability of being on each infection branch.

        Returns
        -------
        draw_dict : dict {str: int}
            Keys are branches values are number of infections on branch.
        """
        if self.rng is None:
            rng = np.random.default_rng()
        else:
            rng = self.rng
        pvals = list(branch_probability.values())
        branches = list(branch_probability.keys())
        draw = rng.multinomial(n=n, pvals=pvals, size=1)
        draw = draw[0]
        draw_dict = {branch: draw[index] for index, branch in enumerate(branches)}
        return draw_dict
    def seed_infections(self, n, branch_probability, parameters):
        """
        Draw selection of states to place infected hosts.

        Parameters
        ----------
        n : int
            Number of infections to seed.
        branch_probability : dict {string, float}
            Probability of being on each infection branch.
        parameters : dict {str: Number}
            Dictionary of parameter values.

        Returns
        -------
        infections_draw : dict {str: int}
             Keys are infected states values are number of hosts in state.
        """
        prob_error = ', all proportion argument should be a number <=1 and >=0.'
        for key, value in branch_probability.items():
            if not isinstance(value, Number):
                raise TypeError(key+' not a Number type'+ prob_error)
            if value > 1 or value < 0:
                raise ValueError(key+' is of value '+ str(value) + prob_error)
        proportions_total = sum(branch_probability.values())
        if not math.isclose(1, proportions_total, abs_tol=0.000001):
            raise ValueError('The sum of dictionary values in proportions should equal 1, it is equal to ' +
                             str(proportions_total)+'.')
        branch_draw = self._seed_branches(n, branch_probability)
        infections_draw = {}
        for branch_name, branch_seed in branch_draw.items():
            branch = self.branches[branch_name]
            branch_infection_draw = branch.seed_infections(branch_seed, parameters, self.rng)
            states_already_drawn = set(infections_draw.keys()).union(set(branch_infection_draw.keys()))
            updated_infection_draws = {state: branch_infection_draw.get(state, 0) + infections_draw .get(state, 0)
                                       for state in states_already_drawn}
            infections_draw = updated_infection_draws
        return infections_draw





