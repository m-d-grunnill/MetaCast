"""
Creation:
    Author: Martin Grunnill
    Date: 2024-03-19
Description: 
    
"""
from unittest import TestCase
from metacast import MetaCaster
import copy

class TestMetaCaster(TestCase):

    def setUpClass(cls):
        pass

    def setUp(self):
        self.copy_of_metacast_model = copy.deepcopy(self.metacast_model)

    def tearDown(self):
        self.copy_of_metacast_model = None

    def test_subpop_transfer(self):
        self.fail()

    def test_ode(self):
        self.fail()

    def test_calculate_fois(self):
        self.fail()

    def test_get_state_index_dict_of_coordinate(self):
        self.fail()

    def test_get_indexes_of_coordinate(self):
        self.fail()

    def test_coordinates_to_subpop_suffix(self):
        self.fail()

    def test_integrate(self):
        self.fail()

    def test_results_array_to_df(self):
        self.fail()

    def test_dimensions(self):
        self.fail()

    def test_foi_population_focus(self):
        self.fail()

    def test_subpop_model(self):
        self.fail()

    def test_parameters(self):
        self.fail()

    def test_shape(self):
        self.fail()
