"""
Creation:
    Author: Martin Grunnill
    Date: 8/8/2022
Description: Sets up metapopulation structure as outlined in https://www.overleaf.com/read/bsrfhxqhtzkk.
 Metapopulation is then saved as json.
    
"""
#%%
# import packages
import json


#%%
# Setting up states that can be transfered between metapopulations
def sports_event(clusters, json_file_name=None):
    vaccinable_states = ['S', 'E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_A', 'F_A', 'R']
    wanning_vaccine_efficacy_states = 'all'
    LFD_true_positive_states = ['P_I', 'P_A', 'M_A', 'M_I', 'M_H', 'F_A', 'F_I']
    PCR_true_positive_states = ['G_I', 'G_A', 'P_I', 'P_A', 'M_A', 'M_I', 'M_H', 'F_A', 'F_I']
    PCR_delay = 'all'
    transfer_list = []
    for cluster in clusters:
        # Lets start off by a general population cluster
        to_and_from_cluster = {'from_cluster': cluster, 'to_cluster': cluster}
        first_vaccination = {**to_and_from_cluster,
                             'from_vaccine_group': 'unvaccinated', 'to_vaccine_group': 'effective',
                             'parameter': 'nu_e', 'states': vaccinable_states}

        waning_vaccination = {**to_and_from_cluster,
                              'from_vaccine_group': 'effective', 'to_vaccine_group': 'waned',
                              'parameter': 'nu_w', 'states': wanning_vaccine_efficacy_states}
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
                     'parameter': 'iota_{RA}', 'states': LFD_true_positive_states}
            transfer_list.append(entry)

        # RT_PCR test information
        PCR_positive_waiting = cluster +'_PCR_waiting'
        to_and_from_cluster = {'from_cluster': cluster, 'to_cluster': PCR_positive_waiting}
        for vaccine_group in vaccine_groups:
            entry = {**to_and_from_cluster,
                     'from_vaccine_group': vaccine_group, 'to_vaccine_group': vaccine_group,
                     'parameter': 'iota_{RTPCR}', 'states': PCR_true_positive_states}
            transfer_list.append(entry)
        
        PCR_positive = cluster +'_PCR_positive'
        to_and_from_cluster = {'from_cluster': PCR_positive_waiting, 'to_cluster': PCR_positive}
        for vaccine_group in vaccine_groups:
            entry = {**to_and_from_cluster,
                     'from_vaccine_group': vaccine_group, 'to_vaccine_group': vaccine_group,
                     'parameter': 'omega_{RTPCR}', 'states': PCR_delay}
            transfer_list.append(entry)

    if json_file_name is not None:
        with open(json_file_name, "w") as outfile:
            json.dump(transfer_list, outfile)
    else:
        return transfer_list

#%%
clusters = ['hosts','host_staff','host_spectators','team_A_supporters','team_B_supporters']
group_info = sports_event(clusters)


#%%
# Save structure as json file
dir_name = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/'+
            'Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/'+
            'CVM_models/Model meta population structures/')
with open(dir_name + "Sports match MGE.json", "w") as outfile:
    json.dump(group_info, outfile)