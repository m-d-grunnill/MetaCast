"""
Creation:
    Author: mdgru 
    Date: 2023-01-03
Description: Function for generating scaffolds  meta_pop_transfer list for clusters and vaccine groups.
    
"""

import json
from varname import nameof
from .error_checks import list_of_strs, list_of_strs_or_all

def age_and_vaccine(age_groups, vaccine_groups, vaccinable_states, json_file_name=None):
    """
    Create age group and vaccination group transfer list outlining population flows for child classes of MetaCaster.

    Parameters
    ----------
    age_groups : list of strings
        Names of age groups in population. NOTE order matters it will be assumed that the population from the first age
        group transfers into the second age group, and so on.
    vaccine_groups : list of strings
        Names of vaccine groups in population. NOTE order matters it will be assumed that the population from the first
        vaccine group transfers into the second vaccine group, and so on.
    vaccinable_states : list of strings or string 'all'
    json_file_name : string, optional
        If given generated transfer list is saved into json file using this string.


    Returns
    -------
    transfer_list : list of dicts
        transfer_list acts as a scafold used in initialising child classes of MetaCaster. transfer_list instructs child
        classes of MetaCaster on what classes flow between subpopulations.
        If json_file_name is None the transfer_list is returned. If json_file_name is not None the transfer_list is
        saved as a json file.

    """
    # Error checks
    list_of_strs(age_groups, nameof(age_groups))
    list_of_strs(vaccine_groups, nameof(vaccine_groups))
    list_of_strs_or_all(vaccinable_states, nameof(vaccinable_states))

    for argument in [age_groups, vaccine_groups, vaccinable_states]:
        if not isinstance(argument,(list,tuple)):
            raise TypeError('First three arguments to this function should be a list or tuple.')
        if not all(isinstance(item, str) for item in argument):
            raise TypeError('First three arguments to this function should be a list or tuple of strings.')
    if not len(age_groups) >= 1:
        raise ValueError('There must be one or more age_groups.')
    if not len(vaccine_groups) >= 2:
        raise ValueError('There must be two or more vaccine_groups.')
    if not len(vaccinable_states) >= 1:
        raise ValueError('There must be one or more vaccinable_states.')

    transfer_list = []
    # Dealing with vaccination group transferes
    for age_group in age_groups:
        to_and_from_age_group = {'from_cluster': age_group, 'to_cluster': age_group}
        for index, vaccine_group in enumerate(vaccine_groups[:-1]):
            next_vaccine_group = vaccine_groups[index+1]
            transfer_list.append({**to_and_from_age_group,
                                   'from_vaccine_group': vaccine_group, 'to_vaccine_group': next_vaccine_group,
                                   'parameter': 'nu_'+ age_group + '_' + vaccine_group, 'states': vaccinable_states})

    # Dealing with age group transfers:
    if len(age_groups) > 1:
        for vaccine_group in vaccine_groups:
            to_and_from_vaccine_group = {'from_vaccine_group': vaccine_group, 'to_vaccine_group': vaccine_group}
            for index, age_group in enumerate(age_groups[:-1]):
                next_age_group = age_groups[index+1]
                transfer_list.append({**to_and_from_vaccine_group,
                                       'from_cluster': age_group, 'to_cluster': next_age_group,
                                       'parameter': 'omega_' + age_group, 'states': 'all'})

    if json_file_name is not None:
        with open(json_file_name, "w") as outfile:
            json.dump(transfer_list, outfile)
    else:
        return transfer_list



