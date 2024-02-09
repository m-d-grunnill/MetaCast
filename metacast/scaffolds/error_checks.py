"""
Creation:
    Author: Martin Grunnill
    Date: 2024-02-09
Description: 
    
"""
def list_of_strs(var, var_name):
    if isinstance(var,list):
        if not all(isinstance(item, str) for item in var):
            raise TypeError(var_name + ' is a list but not all items are strings.')
    else:
        raise TypeError(var_name + ' is not a list.')

def list_of_strs_or_all(var, var_name):
    if isinstance(var,list):
        if not all(isinstance(item, str) for item in var):
            raise TypeError(var_name + ' is a list but not all items are strings.')
    elif var != 'all':
        raise TypeError(var_name + ' must be a list of strings, or the string "all".')