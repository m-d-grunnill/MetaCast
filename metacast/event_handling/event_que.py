"""
Creation:
    Author: Martin Grunnill
    Date: 10/08/2022
Description: Event que for simulating models between events.
    
"""
from .events import (
    BaseEvent,
    TransferEvent,
    ChangeParametersEvent,
    ParametersEqualSubPopEvent,
    )
from collections import OrderedDict, deque
from numbers import Number
import numpy as np
import pandas as pd
from warnings import warn
import copy
import inspect



class EventQueue:
    """
    An event queue for processing events from lowest to the highest time. User defined method simulates model between events.

    See method run_simulation on how to customise simulating of model between events.

    Parameters
    ----------
    event_info_dict : Nested dictionary
        Details information about events to go into event queue.
        First level: keys are the event names (stirngs) values are another dictionary of key value paris:
            times : single float/int or list of floats/ints.
                Times that event occurs.
            type : string
                Type of event. Must be either
                    - 'transfer' for events.TransferEvent
                    - 'change parameter' for ChangeParametersEvent
                    - 'parameter equals subpopulation' for ParametersEqualSubPopEvent.
            All other key values pairs are passed to selected event type as kwargs when the event is initialised.

    Methods
    -------
    reset_event_queue()
        Resets event queue to state at initialisation.
    change_event_proportion(event_names, proportion)
        Change proportion attribute value of events in event_names.
    change_event_factor(event_names, factor)
        Change factor attribute value of events in event_names.
    change_event_value(event_names, value)
        Change value attribute value of events in event_names.
    make_events_nullevents(event_names)
        Turns events into nullevents (do nothing events).
    get_event_names()
        Return list of event names.
    events_at_same_time()
        Returns a dictionary of events occurring at same time. Keys are time values are lists of event names.
    run_simulation(model_object, run_attribute, y0, end_time, parameters_attribute, parameters,
                       start_time=0, simulation_step=1,
                       full_output=False, return_param_changes=False,
                       **kwargs_to_pass_to_func)
        Carries out events in queue, running simulations using user defined method between events.


    """

    def __init__(self, event_list):
        self._event_queue = _EventQueue(event_list)
        self._master_event_queue = copy.deepcopy(self._event_queue)
        self._events = self._event_queue._events

    def reset_event_queue(self):
        """
        Resets event queue to state at initialisation.

        Returns
        -------
        Nothing
        """
        self._event_queue = copy.deepcopy(self._master_event_queue)
        self._events = self._event_queue._events

    def _event_names_checker(self, event_names):
        if event_names == 'all':
            event_names = self.get_event_names()
        else:
            if not pd.api.types.is_list_like(event_names):
                event_names = [event_names]
            if any(not isinstance(item, str) for item in event_names):
                raise TypeError('All event_names entries must be a string.')
            available_event_names = self.get_event_names()
            for event_name in event_names:
                if event_name not in available_event_names:
                    raise TypeError(event_name + ' not listed in names give to available evednt: ' +
                                    ','.join(available_event_names) +  '.')
        return event_names

    def change_event_proportion(self, event_names, proportion):
        """
        Change proportion attribute value of events in event_names.

        Parameters
        ----------
        event_names : list-like of strings or single string
            Name of event(s). If 'all' is entered attribute is changed for all events.
        proportion : float/int between 0 and 1
            Value for which proportion attribute is changed to.

        Returns
        -------
        Nothing
        """
        event_names = self._event_names_checker(event_names)
        for event_name in event_names:
            event = self._events[event_name]
            event.proportion = proportion

    def change_event_factor(self, event_names, factor):
        """
        Change factor attribute value of events in event_names.

        Parameters
        ----------
        event_names : list-like of strings or single string
            Name of event(s). If 'all' is entered attribute is changed for all events.
        factor : int/float
            Value for which factor attribute is changed to.

        Returns
        -------
        Nothing
        """
        event_names = self._event_names_checker(event_names)
        for event_name in event_names:
            event = self._events[event_name]
            event.factor = factor

    def change_event_value(self, event_names, value):
        """
        Change value attribute value of events in event_names.

        Parameters
        ----------
        event_names : list-like of strings or single string
            Name of event(s). If 'all' is entered attribute is changed for all events.
        value : int/float
            Value for which value attribute is changed to.

        Returns
        -------
        Nothing
        """
        event_names = self._event_names_checker(event_names)
        for event_name in event_names:
            event = self._events[event_name]
            event.value = value

    def make_events_nullevents(self, event_names):
        """
        Turns events into nullevents (do nothing events).

        Parameters
        ----------
        event_names : list-like of strings or single string
            Name of event(s). If 'all' is entered all events are made nullevents.

        Returns
        -------
        Nothing
        """
        event_names = self._event_names_checker(event_names)
        for event_name in event_names:
            event = self._events[event_name]
            event.make_event_a_nullevent()

    def get_event_names(self):
        """
        Return list of event names.
        Returns
        -------
        List of strings
            Event names.
        """
        return list(self._events.keys())

    def events_at_same_time(self):
        """
        Returns a dictionary of events occurring at same time. Keys are time values are lists of event names.

        Returns
        -------
        dictionary
             Keys are floats/ints values are lists of strings.
        """
        return self._event_queue.events_at_same_time

    def run_simulation(self, model_object, run_attribute, y0, end_time, parameters_attribute, parameters,
                       start_time=0, simulation_step=1,
                       full_output=False, return_param_changes=False,
                       **kwargs_to_pass_to_func):
        """
        Carries out events in queue, running simulations using user defined method between events.

        Parameters
        ----------
        model_object : object
            Object used to define and simulate model.
        run_attribute : string
            Name of model_objects method that simulates model. Must return either a numpy array of pandas.DataFrame.
        y0 : numpy.array
            Intial values of state varibles.
        end_time : float/int
            End time of simulations.
        parameters_attribute : string
            Attribute of model_object that sets parameters (must accept dictionary where keys are strings and values are
            floats/ints).
        parameters : dict {str : floats/ints}
            Parameters being used by model.
        start_time : float/int, default 0
            Start time of simulations.
        simulation_step : float/int, default 1
            Time steps used in simulations.
        full_output : bool, default False
            If true an info_dict is returned outlining full_output information given from running of model between
            events.
        return_param_changes : bool, default is False
            If true dictionary outlining parameters value changes made by events.
        kwargs_to_pass_to_func : **kwargs
            Key word arguments to pass to run_attribute method.

        Returns
        -------
        y : numpy.array or pandas.Dataframe
            Solution from simulating model and processing events.
        transfers_df : pandas.Dataframe
            Details of values being transferred by events between state variables.
        """
        if not all(time % simulation_step == 0
                   for time in self._event_queue.times):
            raise ValueError('All time points for events must be divisible by simulation_step, leaving no remainder.')
        setattr(model_object, parameters_attribute, parameters)
        param_changes = {'Starting value, time: '+str(start_time): copy.deepcopy(parameters)}
        sim_event_queue = copy.deepcopy(self._event_queue)
        sim_event_queue.prep_queue_for_sim_time(start_time, end_time)
        tranfers_list = []
        y = []
        current_time = start_time
        solution_at_t = copy.deepcopy(y0)
        if full_output:
            run_method = getattr(model_object, run_attribute)
            function_args_inspection = inspect.getfullargspec(run_method)
            full_output_in_func_args = 'full_output' in function_args_inspection.args
            if full_output_in_func_args:
                info_dict = {}
            else:
                warn('Full output unavailable as full_output is not an argument in function given to "func".')

            def _func_with_full_output(func, solution_at_t, current_t_range, info_dict, **kwargs_to_pass_to_func):
                # Not sure if this should be a private function outside of class or private class method.
                y_over_current_t_range, info_sub_dict = func(solution_at_t, current_t_range, full_output=True,
                                                             **kwargs_to_pass_to_func)
                range_as_list = current_t_range.tolist()
                time_points = (range_as_list[0], range_as_list[-1])
                info_dict[time_points] = info_sub_dict
                return y_over_current_t_range


        earliest_event_time = self._event_queue.earliest_event_time
        while sim_event_queue.not_empty():
            next_time, event = sim_event_queue.poptop()
            if next_time > end_time: # if the next event is after simulations time break out of loop.
                break

            if current_time != next_time:
                # run until current time is next time
                current_t_range = np.arange(current_time, next_time+simulation_step, simulation_step)
                run_method = getattr(model_object, run_attribute)
                if full_output and full_output_in_func_args:
                    y_over_current_t_range = _func_with_full_output(run_method, solution_at_t, current_t_range,
                                                                    info_dict, **kwargs_to_pass_to_func)
                else:
                    y_over_current_t_range = run_method(solution_at_t, current_t_range, **kwargs_to_pass_to_func)
                if next_time == earliest_event_time:
                    if not isinstance(y_over_current_t_range, (np.ndarray, np.generic, pd.DataFrame)):
                        raise TypeError('run_attribute should return numpy array of pandas dataframe.')
                if isinstance(y_over_current_t_range, (np.ndarray, np.generic)):
                    solution_at_t = y_over_current_t_range[-1, :]
                    y.append(y_over_current_t_range[:-1, :])
                elif isinstance(y_over_current_t_range, pd.DataFrame):
                    solution_at_t = y_over_current_t_range.iloc[-1].to_numpy()
                    y.append(y_over_current_t_range.iloc[:-1])
                current_time = next_time
            # then do event
            if isinstance(event, TransferEvent):
                transfered = event.process(solution_at_t, current_time)
                transfers_entry = {'time':current_time,
                                   'transfered':transfered,
                                   'event':event.name}
                tranfers_list.append(transfers_entry)
            elif isinstance(event, ChangeParametersEvent):
                parameters = event.process(model_object, parameters_attribute, parameters)
                param_changes[event.name + ', time: ' + str(current_time)] = copy.deepcopy(parameters)

        if current_time != end_time:
            current_t_range = np.arange(current_time, end_time+simulation_step, simulation_step)
            run_method = getattr(model_object, run_attribute)
            if full_output and full_output_in_func_args:
                y_over_current_t_range = _func_with_full_output(run_method, solution_at_t, current_t_range, info_dict, **kwargs_to_pass_to_func)
            else:
                y_over_current_t_range = run_method(solution_at_t, current_t_range, **kwargs_to_pass_to_func)
            y.append(y_over_current_t_range)

        if isinstance(y[0], pd.DataFrame):
            y = pd.concat(y)
        else:
            y = np.vstack(y)
        transfers_df = pd.DataFrame(tranfers_list)
        if full_output and full_output_in_func_args:
            if return_param_changes:
                return y, transfers_df, param_changes, info_dict
            else:
                return y, transfers_df, info_dict
        else:
            if return_param_changes:
                return y, transfers_df, param_changes
            else:
                return y, transfers_df


class _EventQueue:
    """
    Event queue internal to EventQueue class above.
    This is kept separate from EventQueue so that a master copy can be kept for reverting changes back to.

    Parameters
    ----------
    event_information : Nested dictionary
        Details information about events to go into event queue.
        First level: keys are the event names (stirngs) values are another dictionary of key value paris:
            times : single float/int or list of floats/ints.
                Times that event occurs.
            type : string
                Type of event. Must be either
                    - 'transfer' for events.TransferEvent
                    - 'change parameter' for ChangeParametersEvent
                    - 'parameter equals subpopulation' for ParametersEqualSubPopEvent.
            All other key values pairs are passed to selected event type as kwargs when the event is initialised.

    Attributes
    ----------
    queue : OrderedDict
        Queue events ordered by time (keys of ordered dict). When more then one event occurs at a time a deque object
        used to stack events. Events occurring at same time are processed under first one in first one out order.
    events_at_same_time : dict
        A dictionary of events occurring at same time. Keys are time values are lissts of event names.
    times : list of floats or int values.
        Times of events.
    earliest_event_time(self):
        Get the earliest event's time.
    len : int
        Number of events in queue.

    Methods
    -------
    poptop()
        Pops item from top of queue going by order of the lowest event time first.
    prep_queue_for_sim_time(start_time, end_time)
        Edits event queue removing events before start_time or after end_time.
    not_empty()
        Checks that event queue is not empty.

    """

    def __init__(self, events):
        unordered_que = {}
        self.events_at_same_time = {}
        self._events = {}
        if isinstance(events, BaseEvent):
            events = [events]
        if not isinstance(events,(list,tuple,set)):
            raise TypeError('Events should be either a single BaseEvent (or a subclass of BaseEvent) '+
                            'or a list/tuple/set of BaseEvents (or a subclasses of BaseEvent).')
        for event in events:
            self._events[event.name] = event
            times_already_in_queue = set(unordered_que.keys()) & event.times
            if times_already_in_queue:
                for time in times_already_in_queue:
                    if isinstance(unordered_que[time], BaseEvent):
                        self.events_at_same_time[time] = [unordered_que[time].name, event.name]
                        unordered_que[time] = deque([unordered_que[time], event])
                    else:
                        self.events_at_same_time[time].append(event.name)
                        unordered_que[time].append(event)

            times = event.times - times_already_in_queue
            unordered_que.update({time: event for time in times})
        self.queue = OrderedDict(sorted(unordered_que.items()))
        # if self.events_at_same_time:
        #     warn('Concuring events in event queue. To view use method "events_at_same_time".')

    def poptop(self):
        """
        Pops item from top of queue going by order of the lowest event time first.

        Returns
        -------
        event_time : float
            Time of event.
        event : child of event_handling.event.BaseEvent
            An event.
        """
        if self.queue:  # OrderedDicts if not empty are seen as True in bool statements.
            next_item = next(iter(self.queue.values()))
            if isinstance(next_item, deque):
                event_time = next(iter(self.queue.keys()))
                event = next_item.popleft()
                if not next_item:
                    self.queue.popitem(last=False)
                return event_time, event
            else:
                return self.queue.popitem(last=False)
        else:
            raise KeyError("Empty event queue")

    def prep_queue_for_sim_time(self, start_time, end_time):
        """
        Edits event queue removing events before start_time or after end_time.

        Parameters
        ----------
        start_time : float or int
            Start time of simulations.
        end_time : float or int
            End time of simulations.
        Returns
        -------
        Nothing
        """
        first_event_time = next(iter(self.queue.keys()))
        last_event_time = next(reversed(self.queue.keys()))
        if first_event_time < start_time and last_event_time > last_event_time:
            self.queue = OrderedDict({time: item for time, item
                                      in self.queue.items()
                                      if time >= start_time and time <= end_time})
        elif first_event_time < start_time:
            self.queue = OrderedDict({time: item for time, item
                                      in self.queue.items()
                                      if time >= start_time})
        elif last_event_time > last_event_time:
            self.queue = OrderedDict({time: item for time, item
                                      in self.queue.items()
                                      if time <= end_time})

    def not_empty(self):
        """
        Checks that event queue is not empty.
        Returns
        -------
        bool
        """
        return bool(self.queue)

    @property
    def times(self):
        """
        Return a list of the times of events.
        Returns
        -------
        list of floats or int values
            A list of the times of events.
        """
        return list(self.queue.keys())
    @property
    def earliest_event_time(self):
        """
        Get the earliest event's time.

        Returns
        -------
        int or float.
        """
        return self.times[0]

    def __len__(self):
        """
        Returns number of event in queue.

        Returns
        -------
        int
        """
        return len(self.queue)

    def __repr__(self):
        return f"Queue({self.data.items()})"
