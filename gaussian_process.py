from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
import json

# Session variables
session_params = {}


class GaussianProcessSearch:

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
                corresponding values that are already known. New
        """
        self.search_space = search_space
        self.fixed_space = fixed_space
        self.evaluator = evaluator
        self.data_file = data_file
        self.x_values = []
        self.y_values = []
        if data_file is not None:
            try:
                print("Loading data...")
                data_dict = self._load_values()
                self.x_values = data_dict['x_values']
                self.y_values = data_dict['y_values']
            except OSError as e:
                # We can continue but we warn the user
                print('Cannot read input data from ' + str(data_file))
                print(e)
        self.gp_regressor = self._get_gp_regressor()

    @staticmethod
    def _get_gp_regressor(length_scale=1., nu=2.5, noise=0.1):
        """Creates the GaussianProcessRegressor model

        Args:
            length_scale (Union[float, list]): Length scale of the GP kernel. If float, it is the
                same for all dimensions, if array each element defines the length scale of the
                dimension
            nu (float): Controls the smoothness of the approximation.
                see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

        Returns:
            A skopt.learning.GaussianProcessRegressor with the given parameters

        """
        kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)
        return GaussianProcessRegressor(kernel=kernel, alpha=noise**2)

    def get_maximum(self, n_calls=10, n_random_starts=10, acq_optimizer='lbfgs', verbose=True):
        """Performs Bayesian optimization by iteratively evaluating the given function on points
        that are likely to be a global maximum.

        After the optimization, the evaluated values are stored in self.x_values and
        self.y_values and appended to the data file if provided.

        Args:
            n_calls (int): Number of iterations
            n_random_starts (int): Initial random evaluations if no previpus values are provided
            acq_optimizer (str): Acquisition function. See https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html#skopt.gp_minimize
            verbose (bool): Whether to print optimization details at each evaluation

        Returns:
            A tuple (x, y) with the argmax and max found of the evaluated function.

        """
        x_values = [x for x in self.x_values] if len(self.x_values) > 0 else None
        # Negate y_values because skopt performs minimization instead of maximization
        y_values = [-y for y in self.y_values] if len(self.y_values) > 0 else None
        print(y_values)
        res = gp_minimize(func=GaussianProcessSearch.evaluate,
                          dimensions=self.search_space,
                        #   base_estimator=self.gp_regressor,
                          n_calls=n_calls,
                          n_random_starts=n_random_starts,
                          acq_func='EI',
                          acq_optimizer=acq_optimizer,
                          x0=x_values,
                          y0=y_values,
                          noise=1e-10,
                          n_jobs=-1,
                          verbose=verbose)
        ax = plot_objective(res)
        plt.show()
        ax = plot_evaluations(res)
        plt.show()

        for i in range(n_calls):
            self.x_values.append([float(x) for x in res.x_iters[i]])
            # Appending negated value to return the correct sign
            self.y_values.append(float(-res.func_vals[i]))
        if self.data_file is not None:
            self._save_values()
        return res.x, res.fun

    def init_session(self):
        """Save in session variables. the parameters that will be passed to the evaluation function
        by default.

        """
        global session_params
        session_params['fixed_space'] = self.fixed_space
        session_params['evaluator'] = self.evaluator
        session_params['dimension_names'] = [dim.name for dim in self.search_space]

    def reset_session(self):
        """Reset session variables.

        """
        global session_params
        session_params = {}

    def _load_values(self):
        """Load the data file

        Returns:
            A dictionary {'x_values': x_values, 'y_values': y_values} where
                x_values (list) List of the already evaluated points
                y_values (list) The correspondent values

        """
        with open(self.data_file, 'r') as json_file:
            return json.load(json_file)

    def _save_values(self):
        """Save in the data file the known x_values and y_values

        """
        data_dict = {'x_values': self.x_values, 'y_values': self.y_values}
        with open(self.data_file, 'w') as json_file:
            json.dump(data_dict, json_file)

    @staticmethod
    def _to_key_value(values):
        """Transform the given list of values in a key-value dictionary from the search_space names

        Args:
            values (list): List of values of the same length as self.search_space

        Returns:
            A dictionary key[i]: value[i] where key[i] is the name of the i-th dimension of
            self.search_space and value[i] is the i-th element of values

        """
        global session_params
        name_value_dict = {}
        for i, name in enumerate(session_params['dimension_names']):
            name_value_dict[name] = values[i]
        return name_value_dict

    @staticmethod
    def evaluate(point):
        """Evaluate the evaluator function at the given point

        Args:
            point (list): List of values each one corresponding to a dimension of self.search_space

        Returns:
            The value of self.evaluator at the given point, negated (to be used in minimization)
        """
        global session_params
        evaluator_func = session_params['evaluator']
        fixed_space = session_params['fixed_space']
        name_value_dict = GaussianProcessSearch._to_key_value(point)
        args = {**fixed_space, **name_value_dict}
        return -evaluator_func(**args)
