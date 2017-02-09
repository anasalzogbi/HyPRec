#!/usr/bin/env python
"""
A module that provides functionalities for grid search
will be used for hyperparameter optimization.
"""

import csv
import numpy
import itertools as it
from lib.evaluator import Evaluator


class GridSearch(object):

    def __init__(self, recommender, hyperparameters, verbose=True, report_name=None):
        """
        Train number of recommenders using UV decomposition using different parameters.

        :param AbstractRecommender recommender:
        :param dict hyperparameters: A dictionary of the hyperparameters.
        """
        self.recommender = recommender
        self.hyperparameters = hyperparameters
        self._verbose = verbose
        self.evaluator = Evaluator(recommender.get_ratings())
        self.all_errors = dict()
        if report_name is None:
            self.results_file_name = 'grid_search_results.csv'
        else:
            self.results_file_name = report_name + '.csv'

    def get_all_combinations(self):
        """
        The method retuns all possible combinations of the hyperparameters.

        :returns: array of dicts containing all combinations
        :rtype: list[dict]

        >>> get_all_combinations({'_lambda': [0, 0.1], 'n_factors': [20, 40]})
        [{'n_factors': 20, '_lambda': 0}, {'n_factors': 40, '_lambda': 0},
        {'n_factors': 20, '_lambda': 0.1}, {'n_factors': 40, '_lambda': 0.1}]
        """
        names = sorted(self.hyperparameters)
        return [dict(zip(names, prod)) for prod in it.product(
            *(self.hyperparameters[name] for name in names))]

    def train(self):
        """
        The method loops on all  possible combinations of hyperparameters and calls
        the train and split method on the recommender. the train and test errors are
        saved and the hyperparameters that produced the best test error are returned
        """
        best_error = numpy.inf
        best_params = dict()
        train, test = self.recommender.evaluator.naive_split()
        predictions = None
        all_results = [['n_factors', '_lambda', 'rmse', 'train_recall', 'test_recall', 'recall_at_200', 'ratio',
                        'mrr @ 5', 'ndcg @ 5', 'mrr @ 10', 'ndcg @ 10']]
        for hyperparameters in self.get_all_combinations():
            if self._verbose:
                print("running config ")
                print(hyperparameters)
            self.recommender.set_hyperparameters(hyperparameters)
            current_result = [hyperparameters['n_factors'], hyperparameters['_lambda']]
            self.recommender.train()
            metrics = self.recommender.get_evaluation_report()
            for metric in metrics:
                current_result.append(metric)
            all_results.append(current_result)
            if predictions is None:
                predictions = self.recommender.get_predictions()
            rounded_predictions = self.recommender.rounded_predictions()
            test_recall = self.evaluator.calculate_recall(test, rounded_predictions)
            train_recall = self.evaluator.calculate_recall(self.recommender.get_ratings(), rounded_predictions)
            if self._verbose:
                print('Train error: %f, Test error: %f' % (train_recall, test_recall))
            if 1 - test_recall < best_error:
                best_params = hyperparameters
                best_error = 1 - test_recall
            current_key = self.get_key(hyperparameters)
            self.all_errors[current_key] = dict()
            self.all_errors[current_key]['train_recall'] = train_recall
            self.all_errors[current_key]['test_recall'] = test_recall
        self.dump_csv(all_results)
        return best_params, all_results

    def get_key(self, config):
        """
        Given a dict (config) the function generates a key that uniquely represents
        this config to be used to store all errors

        :param dict config: given configuration.
        :returns: string reperesenting the unique key of the configuration
        :rtype: str

        >>> get_key({n_iter: 1, n_factors:200})
        'n_iter:1,n_factors:200'
        """
        generated_key = ''
        keys_array = sorted(config)
        for key in keys_array:
            generated_key += key + ':'
            generated_key += str(config[key]) + ','
        return generated_key.strip(',')

    def dump_csv(self, all_results):
        """
        Given some results as a list of lists, the function dumps to a csv file

        :param str[][] all_results: all results from all runs.
        """
        with open(self.results_file_name, "a") as f:
            writer = csv.writer(f)
            writer.writerows(all_results)
        if self._verbose:
            print("Dumped results to {}".format(self.results_file_name))

    def get_all_errors(self):
        """
        The method returns all errors calculated for every configuration.

        :returns: containing every single computed test error.
        :rtype: dict
        """
        return self.all_errors
