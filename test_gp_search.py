import numpy as np
from skopt.space import Real, Integer, Categorical
from gaussian_process import GaussianProcessSearch


a_space = Real(name='a', low=0., high=1.)
b_space = Integer(name='b', low=0, high=100)
c_space = Integer(name='c', low=0, high=100)

search_space = [a_space, b_space, c_space]

fixed_space = {'d': 1.0}

def func(a, b, c, d):
    # Max is 301
    return a + 2*b + c


gp_search = GaussianProcessSearch(search_space=search_space,
                                  fixed_space=fixed_space,
                                  evaluator=func)
gp_search.init_session()
x, y = gp_search.get_maximum(n_calls=10)
