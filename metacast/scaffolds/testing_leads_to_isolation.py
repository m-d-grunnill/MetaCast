"""
Creation:
    Author: Martin Grunnill
    Date: 8/8/2022
Description: Functions for setting up metapopulation scafolds were testing leads to isolation.
    
"""
# %%
# import packages
import json
from .error_checks import list_of_strs, list_of_strs_or_all
from varname import nameof

# %%
# Setting up states that can be transfered between metapopulations
def wanning_booster_with_ra_pcr(clusters,
                                vaccinable_states=['S', 'E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_A', 'F_A', 'R'],
                                waning_vaccine_efficacy_states='all',
                                ra_true_positive_states=['P_I', 'P_A', 'M_A', 'M_I', 'M_H', 'F_A', 'F_I'],
                                pcr_true_positive_states=['G_I', 'G_A', 'P_I', 'P_A', 'M_A', 'M_I', 'M_H', 'F_A',
                                                          'F_I'],
                                pcr_delay='all',
                                json_file_name=None):
    """
    Creates transfer list for vaccination leading to circling between waning and boosting with RA and RT-PCR testing.

    This scafold was used in the simulations in [1].

    Parameters
    ----------
    clusters : list of strings
        A list of cluster names to use.
    vaccinable_states : list of strings or the string 'all'
        A list of vaccinable states to use. If 'all' given all states used.
    waning_vaccine_efficacy_states : list of strings or the string 'all'
        A list of states were vaccination wanes from. If 'all' given all states used.
    ra_true_positive_states : list of strings or the string 'all'
        A list of states that are RA positive. If 'all' given all states used.
    pcr_true_positive_states : list of strings or the string 'all'
        A list of states that are RT-PCR positive. If 'all' given all states used.
    pcr_delay : list of strings or the string 'all'
        A list of states . If 'all' given all states used.
    json_file_name : string, optional
        If given generated transfer list is saved into json file using this string.

    Returns
    -------
    transfer_list : list of dicts
        transfer_list acts as a scafold used in initialising child classes of MetaCaster. transfer_list instructs child
        classes of MetaCaster on what classes flow between subpopulations.
        If json_file_name is None the transfer_list is returned. If json_file_name is not None the transfer_list is
        saved as a json file.

    References
    ----------
    [1] Grunnill, M., Arino, J., McCarthy, Z., Bragazzi, N. L., Coudeville, L., Thommes, E., Amiche, A., Ghasemi, A.,
        Bourouiba, L., Tofighi, M., Asgary, A., Baky-Haskuee, M., & Wu, J. (2024). Modelling Disease Mitigation at Mass
        Gatherings: A Case Study of COVID-19 at the 2022 FIFA World Cup. In E. H. Lau (Ed.), PLoS Computational Biology:
        Vol. January (Issue 1, p. e1011018). Public Library of Science. https://doi.org/10.1371/JOURNAL.PCBI.1011018
    """
    # Error checks
    list_of_strs(clusters, nameof(clusters))
    list_of_strs_or_all(vaccinable_states, nameof(vaccinable_states))
    list_of_strs_or_all(waning_vaccine_efficacy_states, nameof(waning_vaccine_efficacy_states))
    list_of_strs_or_all(ra_true_positive_states, nameof(ra_true_positive_states))
    list_of_strs_or_all(pcr_true_positive_states, nameof(pcr_true_positive_states))
    list_of_strs_or_all(pcr_delay, nameof(pcr_delay))

    transfer_list = []
    for cluster in clusters:
        # Lets start off by a general population cluster
        to_and_from_cluster = {'from_cluster': cluster, 'to_cluster': cluster}
        first_vaccination = {**to_and_from_cluster,
                             'from_vaccine_group': 'unvaccinated', 'to_vaccine_group': 'effective',
                             'parameter': 'nu_e', 'states': vaccinable_states}

        waning_vaccination = {**to_and_from_cluster,
                              'from_vaccine_group': 'effective', 'to_vaccine_group': 'waned',
                              'parameter': 'nu_w', 'states': waning_vaccine_efficacy_states}
        booster_vaccination = {**to_and_from_cluster,
                               'from_vaccine_group': 'waned', 'to_vaccine_group': 'effective',
                               'parameter': 'nu_b', 'states': vaccinable_states}

        transfer_list += [first_vaccination,
                          waning_vaccination,
                          booster_vaccination]
        # now for isolation
        vaccine_groups = ['unvaccinated', 'effective', 'waned']
        # Testing
        # LFD test information
        LFD_positive = cluster + '_LFD_positive'
        to_and_from_cluster = {'from_cluster': cluster, 'to_cluster': LFD_positive}
        for vaccine_group in vaccine_groups:
            entry = {**to_and_from_cluster,
                     'from_vaccine_group': vaccine_group, 'to_vaccine_group': vaccine_group,
                     'parameter': 'iota_{RA}', 'states': ra_true_positive_states}
            transfer_list.append(entry)

        # RT_PCR test information
        PCR_positive_waiting = cluster + '_PCR_waiting'
        to_and_from_cluster = {'from_cluster': cluster, 'to_cluster': PCR_positive_waiting}
        for vaccine_group in vaccine_groups:
            entry = {**to_and_from_cluster,
                     'from_vaccine_group': vaccine_group, 'to_vaccine_group': vaccine_group,
                     'parameter': 'iota_{RTPCR}', 'states': pcr_true_positive_states}
            transfer_list.append(entry)

        PCR_positive = cluster + '_PCR_positive'
        to_and_from_cluster = {'from_cluster': PCR_positive_waiting, 'to_cluster': PCR_positive}
        for vaccine_group in vaccine_groups:
            entry = {**to_and_from_cluster,
                     'from_vaccine_group': vaccine_group, 'to_vaccine_group': vaccine_group,
                     'parameter': 'omega_{RTPCR}', 'states': pcr_delay}
            transfer_list.append(entry)

    if json_file_name is not None:
        with open(json_file_name, "w") as outfile:
            json.dump(transfer_list, outfile)
    else:
        return transfer_list
