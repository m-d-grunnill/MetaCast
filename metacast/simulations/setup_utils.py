"""
Creation:
    Author: Martin Grunnill
    Date: 2022-09-26
Description: Functions for setting up cluster specific params .
    
"""
def list_to_and_from_cluster_param(param, to_clusters, from_clusters):
    """
    Creates a list of params with the suffixes '_' + cluster_i + '_' + cluster_j based on to_clusters and from_clusters.

    Parameters
    ----------
    param : string
    to_clusters : list of strings
    from_clusters : list of strings

    Returns
    -------
    list of strings
        A list of params with the suffixes '_' + cluster_i + '_' + cluster_j based on to_clusters and from_clusters.
    """
    return [param + '_' + cluster_i + '_' + cluster_j for cluster_i in to_clusters for cluster_j in from_clusters]

def list_cluster_param(param, clusters):
    """
    Creates a list  of params with the suffixes '_' + cluster based on clusters.

    Parameters
    ----------
    param : string
    clusters : list of strings

    Returns
    -------
    list of strings
        A list of params with the suffixes '_' + cluster based on clusters.
    """
    return [param + '_' + cluster for cluster in clusters]

def update_params_with_to_from_cluster_param(params_dict, to_clusters, from_clusters, param, value, return_list=False):
    """
    Updates a dict with to and from cluster versions of param with the value in arg value.

    See Also list_to_and_from_cluster_param.

    Parameters
    ----------
    params_dict : dict
        Dictionary being updated with params.
    to_clusters : list of strings
    from_clusters : list of strings
    param : string
    value : Number
    return_list : bool, default is False

    Returns
    -------
    If return_list==False
        Nothing
    else: list of strings that where added to dict.
    """
    pararms_to_add = list_to_and_from_cluster_param(param, to_clusters, from_clusters)
    update_dict = {term: value
                   for term in pararms_to_add}
    params_dict.update(update_dict)
    if return_list:
        return pararms_to_add

def update_params_with_cluster_param(params_dict, clusters, param, value, return_list=False):
    """
    Updates a dict with cluster versions of param with the value in arg value.

    See Also list_cluster_param

    Parameters
    ----------
    params_dict : dict
        Dictionary being updated with params.
    clusters : list of strings
    param : string
    value : Number
    return_list : bool, default is False

    Returns
    -------
    If return_list==False
        Nothing
    else: list of strings that where added to dict.
    """
    pararms_to_add = list_cluster_param(param, clusters)
    update_dict = {term: value
                   for term in pararms_to_add}
    params_dict.update(update_dict)
    if return_list:
        return list(update_dict.keys())