"""
Creation:
    Author: Martin Grunnill
    Date: 2022-10-05
Description: Assess if a Latin-Hypercube sample size is reasonable.
"""

import pandas as pd
import os
from scipy.stats import qmc
import timeit
from tqdm import tqdm


def determine_lh_sample_size(parameters_df,
                             model_run_method,
                             start_n,
                             repeats_per_n,
                             std_aim,
                             LHS_PRCC_method,
                             save_dir,
                             attempts_to_make=float('inf'),
                             n_increase_addition = None,
                             n_increase_multiple = None,
                             y0=None,
                             other_samples_to_repeat=None,
                             max_workers=None):
    """
    Assess if a Latin-Hypercube (LH) sample size is reasonable.

    Method is as follows:
        1.  For a given LH sample size draw and simulate a model through repeats_per_n number of LHSs.
        2.  Determine standard deviation in PRCCs. If any PRCC is greater than std_aim
            increase LH sample size by either addittion (n_increase_addition)
            or multiplication (n_increase_multiple) and return to 1.

    Parameters
    ----------
    parameters_df : pd.DataFrame
        DataFrame outlining the boundaries for each parameter. Must contain fields 'Lower Bound' and
        'Upper Bound'. Name of the parameters is assumed to be in the index.
    model_run_method : function
        Method for simulating model.
    start_n : int
        Suggested sample size to start from.
    repeats_per_n : int
        Number of repeats per sample size.
    std_aim : float
        Standard deviation goal.
    LHS_PRCC_method : function
        Method for drawing LHs simulating models through LHs and determining PRCC.
    save_dir : string
        Directory for saving results.
    attempts_to_make : int
        Number of attempts to make.
    n_increase_addition : int, mutually exclusive with n_increase_multiple
        Increase in sample size, by addition, to be made if standard deviation in any paramters PRCC
        is greater than std_aim.
    n_increase_multiple : int, mutually exclusive with n_increase_addition
        Increase in sample size, by multiplication, to be made if standard deviation in any paramters PRCC
        is greater than std_aim.
    y0 : numpy.array, optional
        Intial values of state variables.
    other_samples_to_repeat : pandas.DataFrame
        Samples to resampled and merged with LH samples.
    max_workers : int, optional
        Number workers if LHS_PRCC method is parallelized.

    Returns
    -------
    int
        Sample size assessed as being reasonable.
    """
    if n_increase_addition is not None:
        if not isinstance(n_increase_addition,int):
            if isinstance(n_increase_addition, float):
                if not n_increase_addition.is_integer():
                    raise TypeError('n_increase_addition must be an interger > 0.')
        if n_increase_addition <0:
            raise ValueError('n_increase_addition must be > 0.')
    elif n_increase_multiple is not None:
        if not isinstance(n_increase_multiple, (int, float)):
            raise TypeError('n_increase_multiple must be an int or float > 1.')
        if n_increase_multiple <1:
            raise ValueError('n_increase_multiple must be > 1.')
    else:
        raise AssertionError('Either n_increase_addition or n_increase_multiple must be given.')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    LHS_obj = qmc.LatinHypercube(len(parameters_df))
    sample_size = start_n
    aim_reached = False
    attempts_made = 0

    while attempts_made < attempts_to_make and not aim_reached:
        prcc_measures = []
        run_times = []
        save_prcc_stats_file = (save_dir +
                                '/PRCC descriptive stats at sample size ' +
                                str(sample_size) + '.csv')
        run_time_stats_file = (save_dir +
                                '/Run time descriptive stats at sample size ' +
                                str(sample_size) + '.csv')
        temporary_folder = (save_dir +
                            '/PRCC results from LH sample size ' +
                            str(sample_size))
        run_time_record_file = (save_dir +
                                '/Run times at sample size ' +
                                str(sample_size) + '.csv')
        if not os.path.exists(temporary_folder):
            os.makedirs(temporary_folder)
        if not os.path.isfile(save_prcc_stats_file):
            range_of_repeats = range(repeats_per_n)
            for repeat_num in tqdm(range_of_repeats, leave=False, position=0, colour='blue',
                                   desc='LHS resample sample size of '+str(sample_size)):
                temporary_results_file = temporary_folder + '/PRCC results from run ' + str(repeat_num)+'.csv'
                if os.path.exists(temporary_results_file):
                    prcc_measure_enty = pd.read_csv(temporary_results_file, index_col=0)
                    run_times_df = pd.read_csv(run_time_record_file, index_col=0)
                    run_times = run_times_df.Times.to_list()
                else:
                    start = timeit.default_timer()
                    if other_samples_to_repeat is not None:
                        prcc_measure_enty = LHS_PRCC_method(parameters_df, sample_size, model_run_method,
                                                            LHS_obj=LHS_obj, y0=y0,
                                                            other_samples_to_repeat=other_samples_to_repeat,
                                                            max_workers=max_workers)
                    else:
                        prcc_measure_enty = LHS_PRCC_method(parameters_df, sample_size, model_run_method,
                                                            LHS_obj=LHS_obj, y0=y0, max_workers=max_workers)
                    end = timeit.default_timer()
                    run_times.append(end - start)
                    run_times_df = pd.DataFrame({'Times': run_times})
                    run_times_df.to_csv(run_time_record_file)
                    prcc_measure_enty.to_csv(temporary_results_file)
                prcc_measures.append(prcc_measure_enty['r'])
            prcc_measures_df = pd.concat(prcc_measures, axis=1)
            prcc_measures_df.columns = range_of_repeats
            prcc_measures_df = prcc_measures_df.transpose(copy=True)
            prcc_decriptive_stats = prcc_measures_df.describe()
            prcc_decriptive_stats.to_csv(save_prcc_stats_file)
            times_decriptive_stats = run_times_df.Times.describe()
            times_decriptive_stats.to_csv(run_time_stats_file)
        else:
            prcc_decriptive_stats = pd.read_csv(save_prcc_stats_file, index_col=0)

        max_std = prcc_decriptive_stats.loc['std', :].max()
        if max_std > std_aim:
            if n_increase_addition is not None:
                sample_size += n_increase_addition
            elif n_increase_multiple is not None:
                sample_size *= n_increase_multiple

        else:
           aim_reached = True
        attempts_made += 1

    return sample_size

if __name__ == "__main__":
    pass