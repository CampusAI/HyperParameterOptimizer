from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
import numpy as np
import json


class GaussianProcessSearch:
    # Session variables
    fixed_params = {}

    def __init__(self, search_space, fixed_space, evaluator, data_file=None):
        """Instantiate the GaussianProcessSearch and create the GaussianProcessRegressor

        Args:
            search_space (list): List of skopt.space.Dimension objects (Integer, Real,
                or Categorical) whose name must match the correspondent variable name in the
                evaluator function
            fixed_space (dict): Dictionary of parameters that will be passed by default to the
                evaluator function. The keys must match the correspondent names in the function.
            evaluator (function): Function of which we want to estimate the maximum. It must take
                the union of search_space and fixed_space as parameters and return a scalar value.
            data_file (str): Path to the file containing points in the search space and
                corresponding values that are already known.
        """
        self.search_space = search_space
        self.fixed_space = fixed_space
        self.evaluator = evaluator
        self.data_file = data_file
        self.x_values = []
        self.y_values = []
        if data_file is not None:
            try:
                data_dict = self._load_values()
                self.x_values = data_dict['x_values']
                self.y_values = data_dict['y_values']
            except OSError as e:
                # We can continue but we warn the user
                print('Cannot read input data from ' + str(data_file))
                print(e)
        self.gp_regressor = self._get_gp_regressor()

    @staticmethod
    def _get_gp_regressor():
        """Creates the GaussianProcessRegressor model

        Returns:
            A skopt.learning.GaussianProcessRegressor

        """
        # TODO: Should these variables be fixed?
        length_scale = 1.0
        nu = 2.5
        noise = 1.
        kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)
        return GaussianProcessRegressor(kernel=kernel, alpha=noise**2)

    def optimize(self, n_calls=1, n_random_starts=0, acq_optimizer='lbfgs', verbose=True):
        res = gp_minimize(func=GaussianProcessSearch.evaluate,
                          dimensions=self.search_space,
                          base_estimator=self.gp_regressor,
                          n_calls=n_calls,
                          n_random_starts=n_random_starts,
                          acq_func='EI',
                          acq_optimizer=acq_optimizer,
                          x0=self.x_values,
                          y0=self.y_values,
                          verbose=verbose)
        for i in range(n_calls):
            self.x_values.append(res.x_iters[i])
            self.y_values.append(res.func_vals[i])
        self._save_values()

    def init_session(self):
        """Save in session variables. the parameters that will be passed to the evaluation function
        by default.

        """
        GaussianProcessSearch.fixed_params['fixed_space'] = self.fixed_space
        GaussianProcessSearch.fixed_params['evaluator'] = self.evaluator

    def reset_session(self):
        """Reset session variables.

        """
        GaussianProcessSearch.fixed_params = {}

    def _load_values(self):
        """Load the data file

        Returns:
            A dictionary {'x_values': x_values, 'y_values': y_values} where
                x_values (numpy.ndarray) the already evaluated points
                y_values (numpy.ndarray) the correspondent values

        """
        with open(self.data_file, 'r') as json_file:
            return json.load(json_file)

    def _save_values(self):
        """Save in the data file the

        Returns:

        """
        data_dict = {'x_values': self.x_values, 'y_values': self.y_values}
        with open(self.data_file, 'w') as json_file:
            json.dump(data_dict, json_file)

    @staticmethod
    def evaluate(point):
        evaluator_func = GaussianProcessSearch.fixed_params['evaluator']
        fixed_space = GaussianProcessSearch.fixed_params['fixed_space']
        args = {**fixed_space, **point}
        return evaluator_func(**args)
