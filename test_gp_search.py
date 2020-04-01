import numpy as np
from skopt.space import Real, Integer, Categorical
from gaussian_process import GaussianProcessSearch

from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
import json


a_space = Real(name='lr', low=0., high=1.)
b_space = Integer(name='batch_size', low=0, high=200)
c_space = Real(name='alpha', low=0, high=1)
d_space = Real(name='some_param', low=0, high=100)

search_space = [a_space, b_space, c_space, d_space]
fixed_space = {'noise_level': 0.1}


def func(lr, batch_size, alpha, some_param, noise_level):
    # Max = 101
    return lr**3 + batch_size**2 + some_param * alpha + np.random.randn() * noise_level


gp_search = GaussianProcessSearch(search_space=search_space,
                                  fixed_space=fixed_space,
                                  evaluator=func,
                                  input_file=None,  # Use None to start from zero
                                  output_file='test.csv')
gp_search.init_session()
x, y = gp_search.get_maximum(n_calls=10, n_random_starts=0,
                             noise=fixed_space['noise_level'],
                             verbose=True,
                             )

x = gp_search.get_next_candidate(n_points=5)
print('NEXT CANDIDATES: ' + str(x))
