import numpy as np
from skopt.space import Real, Integer, Categorical
from gaussian_process import GaussianProcessSearch


a_space = Real(name='a', low=0., high=1.)
b_space = Integer(name='b', low=0, high=100)
c_space = Categorical(name='c', categories=[0, 12, 35, 56])

search_space = [a_space, b_space, c_space]

fixed_space = {'d': 1.5, 'e': 0.5}


def func(a, b, c, d, e):
    """Maximum is 1.26

    """
    return a * np.exp(e * (c - 35) / (d * (b - 50)))


gp_search = GaussianProcessSearch(search_space=search_space,
                                  fixed_space=fixed_space,
                                  evaluator=func,
                                  data_file='test.json'
                                  )
gp_search.init_session()
x, y = gp_search.get_maximum(n_calls=20)
