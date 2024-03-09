"""
Creation:
    Author: Martin Grunnill
    Date: 2024/03/08
Description: Events for use in class EventQueue.

"""
from numbers import Number
from warnings import warn
import numpy as np

class BaseEvent:
    """
    Base parent event.

    Parameters & Attributes
    -----------------------
    name : str
        Name given to event.
    times : floats/ints, ranges or list/tuple/set of floats/ints
        Times at which event occurs.

    Methods
    -------
    process()
        Method for carrying out event.
    make_event_a_nullevent()
        Changes method into a do null event (do nothing event). Child classes process method should start with
        'if self._do_nothing:
            super().process()
        else:'
        or
        'if self._do_nothing:
            pass
        else:'

    undo_make_event_a_nullevent()
        Changes method from a do null event (do nothing event). Child classes process method should start with
        'if self._do_nothing:
            super().process()
        else:'
        or
        'if self._do_nothing:
            pass
        else:'

    """

    def __init__(self, name, times):
        if not isinstance(name, str):
            raise TypeError('A name given to an event should be a string.')
        self.name = name
        if isinstance(times, (float, int)):
            self.times = set([times])
        elif isinstance(times, (list, tuple, set)):
            if any(not isinstance(entry, (int, float)) for entry in times):
                raise TypeError('All entries in event_information "times" should be ints or floats.')
            self.times = set(times)
        elif isinstance(times, range):
            self.times = set(times)
        elif isinstance(times, (np.ndarray, np.generic)):
            self.times = set(times)
        else:
            raise TypeError('"times" type is not supported. ' +
                            'Accepted types are ranges (including numpy), single ints/floats, ' +
                            'or np.arrays, lists, tupples and sets of ints/floats.')
        self._do_nothing = False

    def process(self):
        """
        Method for carrying out event.

        Returns
        -------
        Nothing
        """
        pass

    def make_event_a_nullevent(self):
        self._do_nothing = True

    def undo_make_event_a_nullevent(self):
        self._do_nothing = False



class ValueFactorProportionChangeEvent(BaseEvent):
    """
    Parent event for events that involve changing a value or values.

    Parameters & Attributes
    -----------------------
    name : str
        Name given to event.
    times : floats/ints, ranges or list/tuple/set of floats/ints
        Times at which event occurs.
    value : foat/int, mutually exclusive with factor and proportion.
        Value that overrides original value(s).
    proportion : foat/int, mutually exclusive with value and factor.
        Proportion the multiplies original value(s). Must be between 0 and 1.
    factor : foat/int, mutually exclusive with value and proportion.
        Factor the multiplies original value(s).

    Methods
    -------
    process()
        Method for carrying out event.
    make_event_a_nullevent()
        Changes method into a do null event (do nothing event).
    undo_make_event_a_nullevent()
        Changes method from a do null event (do nothing event).

    """
    def __init__(self, name, times, value=None, factor=None, proportion=None):
        self._value = None
        self._factor = None
        self._proportion = None
        error_msg = 'Only one out of factor, value and proportion can be given.'
        if factor is not None and value is not None and proportion is not None:
            raise AssertionError(error_msg)
        if factor is not None and value is not None:
            raise AssertionError(error_msg)
        if factor is not None and proportion is not None:
            raise AssertionError(error_msg)
        if value is not None and proportion is not None:
            raise AssertionError(error_msg)
        if factor is not None:
            self.factor = factor
        if value is not None:
            self.value = value
        if proportion is not None:
            self.proportion = proportion
        super().__init__(name=name, times=times)
    @property
    def proportion(self):
        return self._proportion
    @proportion.setter
    def proportion(self, proportion):
        if not isinstance(proportion, Number):
            raise TypeError('Value for proportion must be a numeric type.')
        if proportion <= 0:
            raise ValueError('Value of ' + str(proportion) + ' entered for proportion must be greater than 0.' +
                             'To turn event to a nullevent (an event that transfers nothing use method "make_event_a_nullevent".')
        if proportion > 1:
            raise ValueError('Value of ' + str(proportion) +
                             ' entered for proportion must be less than or equal to 1.')
        if self.value is not None:
            warn(self.name + ' was set to value, it will now be set to a proportion.')
            self._value = None
        if self.factor is not None:
            warn(self.name + ' was set to factor, it will now be set to a proportion.')
            self._factor = None
        self._proportion = proportion
    @property
    def factor(self):
        return self._factor
    @factor.setter
    def factor(self, factor):
        if not isinstance(factor, Number):
            raise TypeError('Value for factor must be a numeric type.')
        if self.value is not None:
            warn(self.name + ' was set to value, it will now be set to a factor.')
            self._value = None
        if self.proportion is not None:
            warn(self.name + ' was set to proportion, it will now be set to a factor.')
            self._proportion = None
        self._factor = factor
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, value):
        if not isinstance(value, Number):
            raise TypeError('Value for value must be a numeric type.')
        if self.factor is not None:
            warn(self.name + ' was set to factor, it will now be set to a value.')
            self._factor = None
        if self.proportion is not None:
            warn(self.name + ' was set to proportion, it will now be set to a value.')
            self._proportion = None
        self._value = value


class TransferEvent(ValueFactorProportionChangeEvent):
    """
    Transfers values between state variables.

    Note if value, factor and proportion are all None event is a null event (do nothing event).

    Parameters & Attributes
    -----------------------
    name : str
        Name given to event.
    times : floats/ints, ranges or list/tuple/set of floats/ints
        Times at which event occurs.
    value : foat/int, optional but mutually exclusive with factor and proportion.
        Value that overrides original parameter value(s).
    proportion : foat/int, optional but mutually exclusive with value and factor.
        Proportion the multiplies original parameter value(s). Must be between 0 and 1.
    factor : foat/int, optional but mutually exclusive with value and proportion.
        Factor the multiplies original parameter value(s).
    from_index: list-like of ints
        Indexes of stat variables from which transfers take place.
    to_index: list-like of ints
        Indexes of stat variables to which transfers take place.


    Methods
    -------
    process()
        Process event by transferring amount between state variables
    make_event_a_nullevent()
        Changes method into a do null event (do nothing event).

    """
    def __init__(self, name, times, value=None, factor=None,  proportion=None,
                 from_index=None, to_index=None):
        super().__init__(name=name, times=times, value=value, factor=factor, proportion=proportion)
        if from_index is None and to_index is None:
            raise AssertionError('A container of ints must be given for from_index or to_index or both.')
        if from_index is not None:
            if not all(isinstance(index, int) for index in from_index):
                raise TypeError('All values in from_index must be ints.')
            if from_index == to_index:
                raise AssertionError('from_index and to_index must not be equivelant.')
            self.number_of_elements = len(from_index)
        if to_index is not None:
            if not all(isinstance(index, int) for index in from_index):
                raise TypeError('All values in to_index must be ints.')
            self.number_of_elements = len(from_index)
        if from_index is not None and to_index is not None and len(from_index)!=len(to_index):
            raise AssertionError('If both from_index and to_index are given they must be of the same length.')
        self.from_index = from_index
        self.to_index = to_index

    def process(self, solution_at_t, time, return_total_effected=True):
        """
        Process event by transferring amount between state variables.
        Note if value, factor and proportion are all None event is a null event (do nothing event).

        Parameters
        ----------
        solution_at_t : numpy.array
            State variable values at time t.
        time : float
            Time at t.
        return_total_effected : bool, default is True
            If true total number transferred is returned.

        Returns
        -------
        If return_total_effected is True, total number transferred is returned.
        """
        if self._do_nothing:
            super().process()
        elif self.value is None and self.factor is None and self.proportion is None:
            pass
        else:
            if self.value is not None:
                transfers = np.repeat(self.value, self.number_of_elements)
                if self.from_index is not None:
                    less_than_array = solution_at_t < transfers
                    if any(less_than_array):
                        warn('The total in one or more states was less than default value being deducted'+
                             ','+str(self.value)+', at time ' + str(time) + '.'+
                             ' Removed total population of effected state or states instead.')
                        transfers[less_than_array] = solution_at_t[self.from_index[less_than_array]]
            if self.proportion is not None:
                transfers = solution_at_t[self.from_index] * self.proportion
            if self.factor is not None:
                transfers = solution_at_t[self.from_index] * self.factor
                if self.factor > 1:
                    warn('More people are being transferred than in population.')
            if self.to_index is not None:
                solution_at_t[self.to_index] += transfers
            if self.from_index is not None:
                solution_at_t[self.from_index] -= transfers
            if return_total_effected:
                return transfers.sum()


class ChangeParametersEvent(ValueFactorProportionChangeEvent):
    """
    Event for changing parameter values.
    Note if value, factor and proportion are all None event is a null event (do nothing event).

    Parameters & Attributes
    -----------------------
    name : str
        Name given to event.
    times : floats/ints, ranges or list/tuple/set of floats/ints
        Times at which event occurs.
    changing_parameters : list-like of strings
        Parameters whose values will be changed.
    value : foat/int, optional but mutually exclusive with factor and proportion.
        Value that overrides original parameter value(s).
    proportion : foat/int, optional but mutually exclusive with value and factor.
        Proportion the multiplies original parameter value(s). Must be between 0 and 1.
    factor : foat/int, optional but mutually exclusive with value and proportion.
        Factor the multiplies original parameter value(s).

    Methods
    -------
    process()
        Process event by change parameter values for parameters listed in self.changing_parameters.
    make_event_a_nullevent()
        Changes method into a do null event (do nothing event).

    """
    def __init__(self, name, times, changing_parameters, value=None, factor=None, proportion=None):
        super().__init__(name=name, times=times, value=value, factor=factor, proportion=proportion)
        self.changing_parameters = changing_parameters

    def process(self, model_object, parameters_attribute, parameters):
        """
        Process event by change parameter values for parameters listed in self.changing_parameters.
        Note if value, factor and proportion are all None event is a null event (do nothing event).

        Parameters
        ----------
        model_object : object
            Object used to define and simulate model.
        parameters_attribute : string
            Attribute of model_object that sets parameters (must accept dictionary where keys are strings and values are
            floats/ints).
        parameters : dict {str : floats/ints}
            Parameters being used by model.


        Returns
        -------
        parameters : dict {str : floats/ints}
            Parameters being used by model after the values of those is list self.changing_parameters is changed.
        """
        if self._do_nothing:
            super().process()
        elif self.value is None and self.factor is None and self.proportion is None:
            pass
        else:
            for parameter in self.changing_parameters:
                if self.value is not None:
                    parameters[parameter] = self.value
                if self.factor is not None:
                    parameters[parameter] *= self.factor
                if self.proportion is not None:
                    parameters[parameter] *= self.proportion

            setattr(model_object, parameters_attribute, parameters)
            return parameters


class ParametersEqualSubPopEvent(BaseEvent):
    """
    Change parameters to be equal to the sum of a subpopulation.

    Parameters & Attributes
    -----------------------
    name : str
        Name given to event.
    times : floats/ints, ranges or list/tuple/set of floats/ints
        Times at which event occurs.
    changing_parameters : list-like of strings
        Parameters whose values will be changed.
    subpopulation_index : list-like of ints
        Index of subpopulation the sum of which will change self.changing_parameters to.


    Methods
    -------
    process()
        Process event by changing parameters to be equal to the sum of a subpopulation.
    make_event_a_nullevent()
        Changes method into a do null event (do nothing event).

    """
    def __init__(self, name, times, changing_parameters, subpopulation_index):
        super().__init__(name=name, times=times)
        self.changing_parameters = changing_parameters
        self.subpopulation_index = subpopulation_index

    def process(self, model_object, parameters_attribute, parameters, solution_at_t):
        """
        Process event by changing parameters to be equal to the sum of a subpopulation.

        Parameters
        ----------
        model_object : object
            Object used to define and simulate model.
        parameters_attribute : string
            Attribute of model_object that sets parameters (must accept dictionary where keys are strings and values are
            floats/ints).
        parameters : dict {str : floats/ints}
            Parameters being used by model.
        solution_at_t : numpy.array
            State variable values at time t.

        Returns
        -------
        Nothing
        """
        if self._do_nothing:
            super().process()
        else:
            value = solution_at_t[self.subpopulation_index].sum()
            for parameter in self.changing_parameters:
                parameters[parameter] = value

            setattr(model_object, parameters_attribute, parameters)
            return parameters


if __name__ == "__main__":
    pass