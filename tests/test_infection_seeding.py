"""
Creation:
    Author: Martin Grunnill
    Date: 2024-03-19
Description: 
    
"""
from unittest import TestCase
from src.metacast import MultinomialSeeder
import copy
import unittest

class TestMultnomialSeeder(TestCase):

    @classmethod
    def setUpClass(cls):
        seeding_info = {'unhospitalised': {'E': 'sigma', 'I': 'gamma'},
                        'hospitalised': {'E': 'sigma', 'I': 'gamma', 'H': 'eta'},
                        }
        cls._seeder = MultinomialSeeder(seeding_info)
        cls._seeder.set_seed(42)
        cls.parameters = {'eta': 0.2,
                          'gamma': 0.14285714285714285,
                          'sigma': 0.3333333333333333,
                          'p_[high]': 0.3,
                          'p_[low]': 0.01,
                          'beta_[low]': 2.857142857142857e-07,
                          'beta_[high]': 2.857142857142857e-07,
                          'rho_[low]_[low]': 1,
                          'rho_[low]_[high]': 1,
                          'rho_[high]_[low]': 1,
                          'rho_[high]_[high]': 1}
        cls.N = 1e6
        cls.proportion_high_risk = 0.1
        cls.low_risk_population = cls.N * (1 - cls.proportion_high_risk)
        cls.high_risk_population = cls.N * cls.proportion_high_risk
        cls.prevelance = 0.01

    def setUp(self):
        self.seeder = self._seeder

    def tearDown(self):
        self.seeder = None

    def test_seed_infections(self):
        low_risk_total_infected = self.low_risk_population * self.prevelance
        low_risk_infected = self.seeder.seed_infections(n=low_risk_total_infected,
                                                        branch_probability={'unhospitalised': 1 - self.parameters['p_[low]'],
                                                                            'hospitalised': self.parameters['p_[low]']},
                                                        parameters=self.parameters)
        self.assertEqual(low_risk_infected, {'E': 2615, 'I': 6348, 'H': 37})


if __name__ == '__main__':
    unittest.main()