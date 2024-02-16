"""
Creation:
    Author: Martin Grunnill
    Date: 2024-02-09
Description: 
    
"""
import math
import json
import numpy as np
import inspect
import itertools
import pandas as pd

from metacast import MassGatheringModel
# import seeding methods
from metacast import MultnomialSeeder
from metacast.event_handling.event_que import EventQueue
from .setup_utils import (
    list_to_and_from_cluster_param,
    list_cluster_param,
    update_params_with_to_from_cluster_param,
    update_params_with_cluster_param
    )


def _load_parameters(file='Parameters values in LHS Sports Match Sims.xlsx',
                     directory='parameters'):
    parameters_df = pd.read_excel(directory+'/'+file, sheet_name='Sheet1',index_col='Parameter')
    fixed_params = parameters_df['Fixed Value'].dropna()
    fixed_params = fixed_params.to_dict()
    parameters_df = parameters_df[parameters_df['Fixed Value'].isnull()]
    # No waning immunity or flows of people between clusters and vaccination groups
    # during international spots matches. This is due to the simulations time of such
    # matches being short in comparison.
    parameters_held_at_0 = {param: 0
                            for param in
                            ['alpha', 'iota_{RA}', 'iota_{RTPCR}', 'nu_e', 'nu_b', 'nu_w']}
    fixed_params.update(parameters_held_at_0)
    return parameters_df, fixed_params



def _gen_host_sub_popultion(host_population, host_tickets, host_staff,
                            hosts_unvaccinated, hosts_effectively_vaccinated, hosts_waned_vaccinated,
                            infection_prevalence):
    host_sub_population = {'hosts': {'unvaccinated': hosts_unvaccinated,
                                     'effective': hosts_effectively_vaccinated,
                                     'waned': hosts_waned_vaccinated},
                           'host_spectators': {},
                           'host_staff': {}}

    for sub_pop in ['unvaccinated', 'effective', 'waned']:
        spectator_sub_pop = round((host_sub_population['hosts'][sub_pop] / host_population) * host_tickets)
        staff_sub_pop = round((host_sub_population['hosts'][sub_pop] / host_population) * host_staff)
        host_sub_population['hosts'][sub_pop] -= (spectator_sub_pop+staff_sub_pop)
        host_sub_population['host_spectators'][sub_pop] = spectator_sub_pop
        host_sub_population['host_staff'][sub_pop] = staff_sub_pop


    for cluster, vac_group_dict in host_sub_population.items():
        for vac_group, population in vac_group_dict.items():
            sub_pop_infections = round(infection_prevalence * population)
            host_sub_population[cluster][vac_group] = {'infections': sub_pop_infections,
                                                       'S': population - sub_pop_infections}

    return host_sub_population

def _gen_visitor_sub_population(supportes_pop,
                                supportes_vaccine_prop,
                                infection_prevalence):
    proportions_total = sum(supportes_vaccine_prop.values())
    if not math.isclose(1, proportions_total, abs_tol=0.000001):
        raise ValueError('The sum of dictionary values in supportes_vaccine_prop should equal 1, it is equal to ' +
                         str(proportions_total) + '.')

    sub_populations = {}
    for vaccine_group, proportion in supportes_vaccine_prop.items():
        sub_population_total = round(supportes_pop*proportion)
        sub_pop_infections = round(sub_population_total*infection_prevalence)
        sub_populations[vaccine_group] = {'S':sub_population_total-sub_pop_infections,
                                          'infections': sub_pop_infections}

    return sub_populations


class SportMatchMGESimulation:
    """
    This class simulate the world cup model. It brings together the metapopulation modelling class MassGatheringModel
    with structure outlined in  'Model meta population structures/Sports match MGE.json', MultnomialSeeder and the
    EventQueue class.

    Parameters & Attributes
    -----------------------
    scaffold : list of dicts
        Scafold instructing MassGatheringModel on the flows of classes between metapopulations.
    fixed_parameters : dict {str: Numeric}
        Parameters that remain fixed between model simulations (i.e. not sampled).

    Attributes
    ----------
    model : MassGatheringModel
        MassGatheringModel initialised with metapopulation structure outlined in
        'Model meta population structures/Sports match MGE.json'. For metapopulation details see manuscripts sections
        on vaccine groups and clusters. For disease stages see manuscripts section on Disease Stages.
    event_queue : EventQueue
        Event queue used for running model through events. See manuscripts section Event Queue.
    hosts_main : list of strings
        Main host clusters.
    host_not_positive : list of strings
        Host clusters that are not positive for COVID (includes those that are waiting for a positive PCR.
    visitors_main : list of strings
        Main visitor clusters.
    visitors_not_positive : list of strings
        Visitor clusters that are not positive for COVID (includes those that are waiting for a positive PCR.
    isolating_clusters : list of strings
        Clusters that are isolating.
    not_isolating_clusters : list of strings
        Clusters that are not isolating.
    match_attendees_main : list of strings
        Clusters that attend the match.
    match_attendees_not_positive : list of strings
        Clusters that attend the match but are not positive.
    not_attending_match : list of strings
        Clusters that do not attend the match.
    start_time : int
        Simulation start time.
    end_time : int
        Simulation end time.
    time_step : float
        Simulation time step.

    Methods
    -------
    run_simulation(sampled_parameters, return_full_results=False, save_dir=None)
        Runs simulations using event_queue and model attributes.

    """

    def __init__(self, scaffold,
                 fixed_parameters={}):
        self.model = MassGatheringModel(scaffold)
        self.fixed_parameters = fixed_parameters
        self._setup_seeder()
        self._setup_event_queue()

    def _setup_seeder(self):
        """
        Initialises multinomial seeder class see manuscript section Simulation of a FIFA 2022 World Cup Match.
        Returns
        -------
        Nothing
        """
        infection_branches = {'asymptomatic': {'E': 'epsilon_1',
                                               'G_A': 'epsilon_2',
                                               'P_A': 'epsilon_3',
                                               'M_A': 'gamma_A_1',
                                               'F_A': 'gamma_A_2'},
                              'symptomatic': {'E': 'epsilon_1',
                                              'G_I': 'epsilon_2',
                                              'P_I': 'epsilon_3',
                                              'M_I': 'gamma_I_1',
                                              'F_I': 'gamma_I_2'},
                              'hospitalised': {'E': 'epsilon_1',
                                               'G_I': 'epsilon_2',
                                               'P_I': 'epsilon_3',
                                               'M_H': 'epsilon_H',
                                               'F_H': 'gamma_H'}}
        # setting up seeding class
        self.seeder = MultnomialSeeder(infection_branches)

    def _create_transfer_index(self, transfer_info, groups):
        from_index = []
        to_index = []
        for vaccine_group_transfer_info in transfer_info:
            # for now we are only interested in pre-travel screen so removing from team_a and team_b supporters, not tranfering.
            if vaccine_group_transfer_info['from_cluster'] in groups:
                if vaccine_group_transfer_info['from_index'] not in from_index:
                    from_index.append(vaccine_group_transfer_info['from_index'])
                    to_index.append(vaccine_group_transfer_info['to_index'])
        from_index = list(itertools.chain.from_iterable(from_index))
        to_index = list(itertools.chain.from_iterable(to_index))
        return from_index, to_index

    def _setup_event_queue(self):
        """
        Sets up event queue as outlined manuscripts section Event Queue.
        Returns
        -------
        Nothing
        """
        # useful lists of clusters
        self.hosts_main = ['hosts', 'host_spectators', 'host_staff']
        self.host_not_positive = (self.hosts_main +
                                  [cluster + '_PCR_waiting' for cluster in self.hosts_main])
        self.visitors_main = ['team_A_supporters', 'team_B_supporters']
        self.visitors_not_positive = (self.visitors_main +
                                      [cluster + '_PCR_waiting' for cluster in self.visitors_main])
        self.isolating_clusters = [cluster for cluster in self.model.clusters
                                   if cluster.endswith('positive') or cluster.endswith('self_isolating')]
        self.not_isolating_clusters = [cluster for cluster in self.model.clusters if
                                       cluster not in self.isolating_clusters]

        self.match_attendees_main = self.visitors_main + ['host_spectators', 'host_staff']
        self.match_attendees_not_positive = (self.match_attendees_main +
                                             [cluster + '_PCR_waiting' for cluster in self.match_attendees_main])
        self.not_attending_match = ['hosts', 'hosts_PCR_waiting'] + self.isolating_clusters

        # Setting up event_queue.
        self.start_time = -2
        self.end_time = 100
        self.time_step = 0.5  # https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-021-06528-3#:~:text=This%20systematic%20review%20has%20identified,one%20single%20centred%20trial%20(BD

        event_info_dict = {}

        # Setup Tests events
        ra_transfer_info = self.model.sub_pop_transition_params_dict['iota_{RA}']
        visitor_ra_from_index, visitor_ra_to_index = self._create_transfer_index(ra_transfer_info,
                                                                                 self.visitors_main)
        event_info_dict['Pre-travel RA'] = {'from_index': visitor_ra_from_index,
                                            # 'to_index': visitor_ra_to_index,
                                            'times': -0.5,
                                            'type': 'transfer'}
        attendee_ra_from_index, attendee_ra_to_index = self._create_transfer_index(ra_transfer_info,
                                                                                   self.match_attendees_main)
        event_info_dict['Pre-match RA'] = {'from_index': attendee_ra_from_index,
                                           'to_index': attendee_ra_to_index,
                                           'times': 2.5,
                                           'type': 'transfer'}

        rtpcr_transfer_info = self.model.sub_pop_transition_params_dict['iota_{RTPCR}']
        visitor_rtpcr_from_index, visitor_rtpcr_to_index = self._create_transfer_index(rtpcr_transfer_info,
                                                                                       self.visitors_main)
        event_info_dict['Pre-travel RTPCR'] = {'from_index': visitor_rtpcr_from_index,
                                               # 'to_index': visitor_rtpcr_to_index,
                                               'times': -1.5,
                                               'type': 'transfer'}
        attendee_rtpcr_from_index, attendee_rtpcr_to_index = self._create_transfer_index(rtpcr_transfer_info,
                                                                                         self.match_attendees_main)
        event_info_dict['Pre-match RTPCR'] = {'from_index': attendee_rtpcr_from_index,
                                              'to_index': attendee_rtpcr_to_index,
                                              'times': 1.5,
                                              'type': 'transfer'}

        # Those about to be hospitalised or Hospitalised should be removed before arriving in host nation.
        visitor_pre_host_and_hosp = []
        for cluster in ['team_A_supporters', 'team_B_supporters']:
            vaccine_group_dict = self.model.state_index[cluster]
            for states_group_index in vaccine_group_dict.values():
                visitor_pre_host_and_hosp += [states_group_index[state] for state in ['M_H', 'F_H']]
        event_info_dict['Removing pre-hospitalised and hospitalised visitors'] = {
            'from_index': visitor_pre_host_and_hosp,
            'proportion': 1,
            'times': 0,
            'type': 'transfer'}
        pop_visitors_arrive = list_cluster_param('N',
                                                 clusters=self.host_not_positive + self.visitors_not_positive)
        all_clusters_index = self.model.get_clusters_indexes(self.model.clusters)
        event_info_dict['visitor arrival pop changes'] = {'type': 'parameter equals subpopulation',
                                                          'changing_parameters': pop_visitors_arrive,
                                                          'times': 0,
                                                          'subpopulation_index': all_clusters_index}
        beta_visitors_arrive = list_to_and_from_cluster_param('beta',
                                                              to_clusters=self.host_not_positive + self.visitors_not_positive,
                                                              from_clusters=self.host_not_positive + self.visitors_not_positive)
        event_info_dict['visitor arrival beta changes'] = {'type': 'change parameter',
                                                           'changing_parameters': beta_visitors_arrive,
                                                           'times': 0}

        event_info_dict['Restart Cumulative counts'] = {'from_index': [-1, -2],
                                                        'proportion': 1,
                                                        'times': 0,
                                                        'type': 'transfer'}

        pop_match_day_begins = list_cluster_param('N',
                                                  clusters=self.match_attendees_not_positive)
        attendee_index = self.model.get_clusters_indexes(self.match_attendees_not_positive)
        event_info_dict['match day begins pop changes attendees'] = {'type': 'parameter equals subpopulation',
                                                                     'changing_parameters': pop_match_day_begins,
                                                                     'times': 3,
                                                                     'subpopulation_index': attendee_index}

        pop_match_day_begins_non_attendees = list_cluster_param('N',
                                                                clusters=self.not_attending_match)
        non_attendee_index = self.model.get_clusters_indexes(self.not_attending_match)
        event_info_dict['match day begins pop changes non-attendees'] = {'type': 'parameter equals subpopulation',
                                                                         'changing_parameters': pop_match_day_begins_non_attendees,
                                                                         'times': 3,
                                                                         'subpopulation_index': non_attendee_index}
        beta_match_day = list_to_and_from_cluster_param('beta',
                                                        to_clusters=self.match_attendees_not_positive,
                                                        from_clusters=self.match_attendees_not_positive)
        event_info_dict['match day begins beta changes'] = {'type': 'change parameter',
                                                            'changing_parameters': beta_match_day,
                                                            'times': 3}

        event_info_dict['match day ends pop changes'] = {'type': 'parameter equals subpopulation',
                                                         'changing_parameters': pop_visitors_arrive,
                                                         'times': 4,
                                                         'subpopulation_index': all_clusters_index}
        event_info_dict['match day ends beta changes'] = {'type': 'change parameter',
                                                          'changing_parameters': beta_match_day,
                                                          'times': 4}
        transmision_terms = [param for param in self.model.all_parameters if param.startswith('beta')]
        event_info_dict['MGE ends'] = {'type': 'change parameter',
                                       'changing_parameters': transmision_terms,
                                       'value': 0,
                                       'times': 7}
        self.event_queue = EventQueue(event_info_dict)

    def _calculate_certain_parameters(self, parameters):
        gamma_I_1 = 2 / (parameters['gamma_{IT}^{-1}'] - parameters['epsilon_1'] ** -1 - parameters['epsilon_2'] ** -1 -
                         parameters['epsilon_3'] ** -1)
        parameters['gamma_I_1'] = gamma_I_1
        parameters['gamma_I_2'] = gamma_I_1
        parameters['gamma_A_1'] = gamma_I_1
        parameters['gamma_A_2'] = gamma_I_1
        parameters['p_h_s'] = parameters['p_h'] / parameters['p_s']
        parameters['h_effective'] = 1 - ((1 - parameters['VE_{hos}']) / (1 - parameters['l_effective']))
        parameters['h_waned'] = 1 - ((1 - parameters['VW_{hos}']) / (1 - parameters['l_waned']))

    def run_simulation(self, sampled_parameters, return_full_results=False, save_dir=None):
        """
        Runs simulations using event_queue and model attributes.

        Parameters
        ----------
        sampled_parameters : dict {string: Numeric}
            Parameters sampled for running this iteration of model. This is dictionary is merged with
            self.fixed_parameters.
        return_full_results : bool, default is False
            If true solution and transfers_df Dataframes are returned.
        save_dir : str, optional
            Directory for saving results. If given solution and transfers_df Dataframes are saved to directory.

        Returns
        -------
        focused_ouputs_and_sample : dictionary
            Focused outputs ('peak infected', 'total infections', 'peak hospitalised' and 'total hospitalisations')
            merged with sampled parameters.

        Optional Returns
        ----------------
        solution : pandas.DataFrame
            Model simulations results. Multi-index columns are clusters by vaccine groups by states.
        transfers_df : pandas.DataFrame
            Outlines values transferred between states.
        """
        # reset any changes in event queue caused by previously running self.run_simulation
        self.event_queue.reset_event_queue()
        parameters = {**self.fixed_parameters, **sampled_parameters}
        self._calculate_certain_parameters(parameters)
        # Setup population
        attendees = round(parameters['N_{A}'])
        host_attendees = round(attendees * parameters['eta_{spectators}'])
        visitor_attendees = attendees - host_attendees
        host_and_visitor_population = parameters['N_{hosts}'] + visitor_attendees
        staff = round(parameters['N_{staff}'])

        # Setup starting transmission and changes in transmission
        beta_derive_func_args = list(inspect.signature(MGE_beta_no_vaccine_1_cluster).parameters.keys())
        params_for_deriving_beta = {param: parameters[param]
                                    for param in beta_derive_func_args}
        beta = MassGatheringModel.beta_one_sub_pop(**params_for_deriving_beta)

        # starting transmission terms
        # all isolating clusters will have reduced transmission
        isolation_beta = beta * parameters['kappa']
        update_params_with_to_from_cluster_param(parameters,
                                                 self.model.clusters,
                                                 self.isolating_clusters,
                                                 'beta',
                                                 isolation_beta)

        # There positive population only gets added to once pre match testing begins so ..
        update_params_with_cluster_param(parameters,
                                         self.isolating_clusters,
                                         'N',
                                         host_and_visitor_population)
        # isolating clusters only include infecteds or recovereds in this model (they cannot be infected again so):
        update_params_with_to_from_cluster_param(parameters,
                                                 self.isolating_clusters,
                                                 self.model.clusters,
                                                 'beta',
                                                 0)

        # beta between host clusters starts at default and stays same until 'MGE ends' event
        update_params_with_to_from_cluster_param(parameters,
                                                 self.host_not_positive,
                                                 self.host_not_positive,
                                                 'beta', beta)
        # all transmission to hosts happens in there population to begin with.
        update_params_with_cluster_param(parameters,
                                         self.host_not_positive,
                                         'N',
                                         parameters['N_{hosts}'])  # starts off just Qatari population

        # all visitor clusters do not transmit to begin with.
        update_params_with_to_from_cluster_param(parameters,
                                                 self.model.clusters, self.visitors_not_positive,
                                                 'beta', 0)
        # or get transmited to begin with.
        update_params_with_to_from_cluster_param(parameters,
                                                 self.visitors_not_positive, self.model.clusters,
                                                 'beta', 0)
        # all vistor population are half the number of visitor attendees to begin with
        update_params_with_cluster_param(parameters,
                                         self.visitors_not_positive,
                                         'N', round(visitor_attendees / 2))

        # visitors arrive transmission terms changes
        self.event_queue.change_event_value('visitor arrival beta changes', beta)

        # match day begins
        self.event_queue.change_event_value('match day begins beta changes',
                                            beta * parameters['b'])

        # match day ends
        self.event_queue.change_event_value('match day ends beta changes', beta)

        # change test_event values
        if parameters['Pre-travel RTPCR']:
            self.event_queue.change_event_proportion('Pre-travel RTPCR',
                                                     parameters['tau_{RTPCR}'])
        if parameters['Pre-match RTPCR']:
            self.event_queue.change_event_proportion('Pre-match RTPCR',
                                                     parameters['tau_{RTPCR}'])
        if parameters['Pre-travel RA']:
            if parameters['Pre-travel RA'] == 'low':
                self.event_queue.change_event_proportion('Pre-travel RA',
                                                         parameters['tau_{RA} low'])
            elif parameters['Pre-travel RA'] == 'mid':
                self.event_queue.change_event_proportion('Pre-travel RA',
                                                         parameters['tau_{RA} mid'])
            elif parameters['Pre-travel RA'] == 'high':
                self.event_queue.change_event_proportion('Pre-travel RA',
                                                         parameters['tau_{RA} high'])
            else:
                raise ValueError('"Pre-travel RA" entry in parameters should be False or ' +
                                 'a string value of "low", "mid" or "high".')

        if parameters['Pre-match RA']:
            if parameters['Pre-match RA'] == 'low':
                self.event_queue.change_event_proportion('Pre-match RA',
                                                         parameters['tau_{RA} low'])
            elif parameters['Pre-match RA'] == 'mid':
                self.event_queue.change_event_proportion('Pre-match RA',
                                                         parameters['tau_{RA} mid'])
            elif parameters['Pre-match RA'] == 'high':
                self.event_queue.change_event_proportion('Pre-match RA',
                                                         parameters['tau_{RA} high'])
            else:
                raise ValueError('"Pre-match RA" entry in parameters should be False or ' +
                                 'a string value of "low", "mid" or "high".')

        # setting up submodels
        # Setting up host population
        hosts_unvaccinated = parameters['N_{hosts}'] - parameters['N_{hosts,full}']
        hosts_waned_vaccinated = parameters['N_{hosts,full}'] - parameters['N_{hosts,eff}']
        host_sub_population = _gen_host_sub_popultion(parameters['N_{hosts}'], host_attendees, staff,
                                                      hosts_unvaccinated,
                                                      parameters['N_{hosts,eff}'],
                                                      hosts_waned_vaccinated,
                                                      parameters['sigma_H'])

        # Sorting visitor submodels
        team_A_vacc_status_proportion = {'unvaccinated': 0,
                                         'effective': parameters['v_A'],
                                         'waned': 1 - parameters['v_A']}
        team_B_vacc_status_proportion = {'unvaccinated': 0,
                                         'effective': parameters['v_B'],
                                         'waned': 1 - parameters['v_B']}
        attendees_per_team = round(0.5 * visitor_attendees)
        visitor_sub_pop = {'team_A_supporters': _gen_visitor_sub_population(attendees_per_team,
                                                                            team_A_vacc_status_proportion,
                                                                            parameters['sigma_A']),
                           'team_B_supporters': _gen_visitor_sub_population(attendees_per_team,
                                                                            team_B_vacc_status_proportion,
                                                                            parameters['sigma_B'])
                           }

        # Seeding infections
        self.seeder.set_seed(int(sampled_parameters['seed']))  # takes seed from LH sample ensuring same results are
        # obtained if LH sample is run.
        all_sub_pops = {**host_sub_population, **visitor_sub_pop}
        y0 = np.zeros(self.model.num_state)
        state_index = self.model.state_index
        for cluster, vaccine_group_dict in all_sub_pops.items():
            for vaccine_group, s_vs_infections in vaccine_group_dict.items():
                state_index_sub_pop = state_index[cluster][vaccine_group]
                y0[state_index_sub_pop['S']] = s_vs_infections['S']
                p_s_v = parameters['p_s'] * (1 - parameters['s_' + vaccine_group])
                p_h_v = parameters['p_h_s'] * (1 - parameters['h_' + vaccine_group])
                asymptomatic_prob = 1 - p_s_v
                hospitalised_prob = p_s_v * p_h_v
                symptomatic_prob = p_s_v * (1 - p_h_v)
                infection_branch_proportions = {'asymptomatic': asymptomatic_prob,
                                                'symptomatic': symptomatic_prob,
                                                'hospitalised': hospitalised_prob}
                params_for_seeding = {key: value for key, value in parameters.items()
                                      if key in self.seeder.parameters}
                seeds = self.seeder.seed_infections(s_vs_infections['infections'],
                                                    infection_branch_proportions,
                                                    params_for_seeding)
                for state, population in seeds.items():
                    y0[state_index_sub_pop[state]] = population

        # Runninng mg_model
        parameters_needed_for_model = {key: value for key, value in parameters.items()
                                       if key in self.model.all_parameters}
        self.model.parameters = parameters_needed_for_model
        solution, transfers_df = self.event_queue.run_simulation(model_object=self.model,
                                                                 run_attribute='integrate',
                                                                 parameters=parameters_needed_for_model,
                                                                 parameters_attribute='parameters',
                                                                 y0=y0,
                                                                 end_time=self.end_time, start_time=self.start_time,
                                                                 simulation_step=self.time_step)
        test_events = ['Pre-travel RTPCR', 'Pre-travel RA',
                       'Pre-match RTPCR', 'Pre-match RA']
        tested_positives = transfers_df[transfers_df.event.isin(test_events)]
        tested_positive_total = tested_positives.transfered.sum()
        run_time = np.arange(self.start_time, self.end_time, self.time_step)
        MGE_run_time_start = np.where(run_time == 0)[0][0]
        infection_prevelances = solution.iloc[MGE_run_time_start:, self.model.infected_states_index_list]
        all_infection_prevelances = infection_prevelances.sum(axis=1)
        peak_infected = all_infection_prevelances.max()
        total_infections = solution.iloc[-1, -1]
        hospital_prevelances = solution.iloc[MGE_run_time_start:, self.model.hospitalised_states_index_list]
        all_hospitalisation_prevelances = hospital_prevelances.sum(axis=1)
        peak_hospitalised = all_hospitalisation_prevelances.max()
        total_hospitalisations = solution.iloc[-1, -2]
        focused_ouputs_and_sample = {'peak infected': peak_infected,
                                     'total infections': total_infections,
                                     'peak hospitalised': peak_hospitalised,
                                     'total hospitalisations': total_hospitalisations,
                                     # 'total positive tests': tested_positive_total,
                                     **sampled_parameters}
        if return_full_results:
            if save_dir is None:
                return focused_ouputs_and_sample, solution, transfers_df
            else:
                if 'Sample Number' in parameters:
                    solution.to_csv(save_dir + '/Solution ' + str(parameters['Sample Number']) + '.csv', index=False,
                                    header=True)
                    transfers_df.to_csv(save_dir + '/Event que transfers ' + str(parameters['Sample Number']) + '.csv',
                                        index=False, header=True)
                    focused_ouputs_df = pd.DataFrame([focused_ouputs_and_sample])
                    focused_ouputs_df.to_csv(
                        save_dir + '/Focused Outputs and Sample ' + str(parameters['Sample Number']) + '.csv',
                        index=False, header=True)
                else:
                    solution.to_csv(save_dir + '/Solution.csv')
                    transfers_df.to_csv(save_dir + '/Event que transfers.csv')
                return focused_ouputs_and_sample
        else:
            return focused_ouputs_and_sample