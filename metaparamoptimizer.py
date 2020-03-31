import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import itertools as it
import numpy as np
import pickle

# TODO(Oleguer): Think about the structure of all this

class MetaParamOptimizer:
    def __init__(self, save_path=""):
        self.save_path = save_path  # Where to save best result and remaining to explore
        pass

    def list_search(self, evaluator, dicts_list, fixed_args):
        """ Evaluates model (storing best) on provided list of param dictionaries
            running evaluator(**kwargs = fixed_args + sample(search_space))
            evaluator should return a dictionary conteining (at least) the field "value" to maximize
            returns result of maximum result["value"] reached adding result["best_params"] that obtained it
        """
        max_result = None
        for indx, evaluable_args in enumerate(dicts_list):
            print("MetaParamOptimizer evaluating:", indx, "/", len(dicts_list), ":", evaluable_args)
            args = {**evaluable_args, **fixed_args}  # Merge kwargs and evaluable_args dicts
            try:
                result = evaluator(**args)
            except Exception as e:
                print("MetaParamOptimizer: Exception found when evaluating:")
                print(e)
                print("Skipping to next point...")
                continue
            if (max_result is None) or (result["value"] > max_result["value"]):
                max_result = result
                max_result["best_params"] = evaluable_args
                self.save(max_result, name="metaparam_search_best_result")  # save best result found so far
            # Save remaning tests (in case something goes wrong, know where to keep testing)
            self.save(dicts_list[indx+1:], name="remaining_tests")
        return max_result

    def grid_search(self, evaluator, search_space, fixed_args):
        """ Performs grid search on specified search_space
            running evaluator(**kwargs = fixed_args + sample(search_space))
            evaluator should return a dictionary conteining (at least) the field "value" to maximize
            returns result of maximum result["value"] reached adding result["best_params"] that obtained it
        """
        points_to_evaluate = self.__get_all_dicts(search_space)
        return self.list_search(evaluator, points_to_evaluate, fixed_args)

    def GPR_optimizer(self, evaluator, search_space, fixed_args):
        pass # The other repo

    def save(self, elem, name="best_result"):
        """ Saves result to disk"""
        with open(self.save_path + "/" + name + ".pkl", 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def load(self, name="best_model", path=None):
        if path is None:
            path = self.save_path
        with open(path + "/" + name, 'rb') as input:
            remaining_tests = pickle.load(input)
        return remaining_tests

    def __get_all_dicts(self, param_space):
        """ Given:
            dict of item: list(elems)
            returns:
            list (dicts of item : elem)
        """
        allparams = sorted(param_space)
        combinations = it.product(*(param_space[Name] for Name in allparams))
        dictionaries = []
        for combination in combinations:
            dictionary = {}
            for indx, name in enumerate(allparams):
                dictionary[name] = combination[indx]
            dictionaries.append(dictionary)
        return dictionaries