import numpy as np
from skopt.space import Real, Integer, Categorical
from gaussian_process import GaussianProcessSearch

from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
import json


a_space = Real(name='a', low=-5., high=1.)
b_space = Real(name='b', low=-2, high=10)
c_space = Real(name='c', low=0, high=100)

search_space = [a_space, b_space, c_space]
# search_space = [b_space, c_space]

fixed_space = {'noise_level': 0.1}



def func(a, b, c, noise_level):
    # Max = 101
    return a**3 - b**2 + c + np.random.randn() * noise_level


gp_search = GaussianProcessSearch(search_space=search_space,
                                  fixed_space=fixed_space,
                                  evaluator=func)
gp_search.init_session()
x, y = gp_search.get_maximum(n_calls=10, n_random_starts=5, 
                            noise = fixed_space['noise_level'],
                            verbose=False)
print(x)
print(y)
print(func(x[0], x[1], x[2], fixed_space['noise_level']) - y)
