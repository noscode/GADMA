#!/usr/bin/env python

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################

import time
import copy
import numpy as np
import os
import sys
import math
import cPickle as pickle

from gadma.demographic_model import Demographic_model
from gadma import options
from gadma import support


class GA(object):
    max_mutation_rate = 1.0

    def __init__(
            self,
            params,
            prefix=None,
            one_initial_model=None):
        """Genetic algorithm class.

        params :    an object with parameters to work with
            parameters for genetic algorithm:
            size_of_generation :    size of the generation of demographic models.
            frac_of_old_models :    the fraction of models from the previous
                                    population in a new one.
            frac_of_mutated_models :    the fraction of mutated models in a new
                                        population.
            frac_of_crossed_models :    the fraction of crossed models in a
                                        new population.
            mutation_rate : the rate to change one parameter of model in order
                            to get mutatetd model from it
            epsilon :   constant to stop genetic algorithm, its "presition"
            out_dir :   output directory
            final_structure :   structure of final model
        (Other fields in params are for Demographic model class)
        prefix :    prefix for output folder.
        """
        # all parameters
        self.params = params

        # some constants on number of iterations:

        self.prefix = prefix
        self.is_custom_model = self.params.initial_structure is None
        self.one_initial_model = one_initial_model
        self.ll_precision = 1 - int(math.log(self.params.epsilon, 10))

        
        # basic parameters
        self.cur_iteration = 0
        self.first_iteration = 0  # is not 0 if we restore ga
        self.work_time = 0
        self.model = None
        self.best_model = None

        # connected files and directories
        self.out_dir = None
        self.log_file = None
        self.best_model_by_aic = None

    def pickle_final_models(self, load=None):
        if load is None:
            pickle_file = os.path.join(self.out_dir, 'final_models_pickle')
            with (open(pickle_file, "wb")) as f:
                pickle.dump(self.final_models, f)
        else:
            if not os.path.isfile(load):
                return
            with (open(load, "rb")) as f:
                self.final_models = pickle.load(f)


    def get_random_model(self, structure=None):
        return Demographic_model(self.params, structure=structure)

    def mean_time(self):
        """Get mean time for one iteration."""
        return self.work_time / (self.cur_iteration + 1 - self.first_iteration)

    def best_model(self):
        """Get best model in current population."""
        return self.best_model

    def best_fitness_value(self):
        """Get best fitness value of current population of models."""
        return self.best_model().get_fitness_func_value()

    def is_stoped(self):
        """Check if we need to stop."""
        return False
    
    def run_one_iteration(self):
        """Iteration step for genetic algorithm."""
        start = time.time()

        # take  self.number_of_old_models best models from previous population
        new_model =  self.get_random_model()

        self.model = new_model
        stop = time.time()
        self.work_time += stop - start

        s = self.work_time
        t = '\n[%(hours)03d:%(minutes)02d:%(seconds)02d]' % {'hours': s / 3600, 'minutes': s % 3600 / 60, 'seconds': s % 60}
        support.write_to_file(self.log_file, self.cur_iteration, t, - self.model.get_fitness_func_value(), self.model)

    
    def run(self, shared_dict=None):
        """Main function to run genetic algorithm.

        shared_dict :   dictionary to share information among processes.
        """
        # checking dirs
        if self.prefix is not None:
            self.out_dir = os.path.join(self.params.output_dir, self.prefix)
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
            self.log_file = os.path.join(self.out_dir, 'GADMA_GA.log')
            open(self.log_file, 'a').close()
        else:
            self.log_file = None

        
        # begin
        support.write_to_file(self.log_file,
                              '--Start random pipeline--')

        self.run_one_iteration() 
        self.best_model = copy.deepcopy(self.model)
        if shared_dict is not None:
            shared_dict[
                self.prefix] = copy.deepcopy(
                    self.best_model)


        self.cur_iteration += 1

        while True:
            self.run_one_iteration()
            if self.best_model.get_fitness_func_value() > self.model.get_fitness_func_value():
                self.best_model = copy.deepcopy(self.model)
                if shared_dict is not None:
                    shared_dict[
                        self.prefix] = copy.deepcopy(
                            self.best_model)

        return self.best_model
