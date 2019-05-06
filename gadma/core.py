#!/usr/bin/env python

############################################################################
# Copyright (c) 2018 Noskova Ekaterina
# All Rights Reserved
# See the LICENSE file for details
############################################################################
from __future__ import print_function
import numpy as np
import os

from gadma import options
from gadma import support
from gadma.genetic_algorithm import GA
from datetime import datetime

import operator
from collections import defaultdict
import multiprocessing
import signal
from multiprocessing import Manager, Pool
import math


def run_genetic_algorithm(params_tuple):
    """Main function to run genetic algorithm with prefix number.

    params_tuple :    tuple of 3 objects:
        number :    number of GA to run (moreover its prefix).
        log_file :  log file to write output.
        shared_dict :   dictionary to share information between processes.
    """
    np.random.seed()
    number = 'ID'
#    try:
    def write_func(string): return support.write_log(log_file, string,
                                                     write_to_stdout=not params.silence)

    number, params, log_file, shared_dict = params_tuple

    write_func('Run genetic algorithm number ' + str(number))

    ga_instance = GA(params, prefix=str(number))
    ga_instance.run(shared_dict=shared_dict)

    write_func('Finish genetic algorithm number ' + str(number))
    return number, ga_instance.best_model()
#    except Exception, e:
#        raise RuntimeError('GA number ' + str(number) + ': ' + str(e))


def worker_init():
    """Graceful way to interrupt all processes by Ctrl+C."""
    # ignore the SIGINT in sub process
    def sig_int(signal_num, frame):
        pass

    signal.signal(signal.SIGINT, sig_int)


def print_best_solution_now(start_time, shared_dict, params,
                            log_file, precision, draw_model):
    """Prints best demographic model by logLL among all processes.

    start_time :    time when equation was started.
    shared_dict :   dictionary to share information between processes.
    log_file :      file to write logs.
    draw_model :    plot model best by logll and best by AIC (if needed).
    """

    def write_func(string): return support.write_log(log_file, string,
                                                     write_to_stdout=not params.silence)

    def my_str(x): return support.float_representation(x, precision)


    all_models_data = dict(shared_dict)
    if not all_models_data:
        return
    all_models = [(i, all_models_data[i]) for i in all_models_data]
    all_models = sorted(all_models, key=lambda x: x[1].get_fitness_func_value())

    s = (datetime.now() - start_time).total_seconds()
    write_func('\n[%(hours)03d:%(minutes)02d:%(seconds)02d]' % {
        'hours': s / 3600,
        'minutes': s % 3600 / 60,
        'seconds': s % 60
    })

    support.print_set_of_models(log_file, all_models, 
                params, first_col='Run number', heading='All best logLL models:', silence=params.silence)


def main():
    params = options.parse_args()

    log_file = os.path.join(
        params.output_dir, 'GADMA.log')
    open(log_file, 'w').close()

    support.write_log(log_file, "--Successful arguments' parsing--\n")
    params_filepath = os.path.join(
        params.output_dir, 'params')
    params.save(params.output_dir)
    if not params.test:
        support.write_log(
            log_file, 'You can find all parameters of this run in:\t\t' + params_filepath + '\n')
        support.write_log(log_file, 'All output is saved (without warnings and errors) in:\t' +
                          os.path.join(params.output_dir, 'GADMA.log\n'))

    support.write_log(log_file, '--Start pipeline--\n')

    # For debug
#    run_genetic_algorithm((1, params, log_file, None))

    # Create shared dictionary
    m = Manager()
    shared_dict = m.dict()

    # Start pool of processes
    start_time = datetime.now()
    
    pool = Pool(processes=params.processes,
                initializer=worker_init)
    try:
        pool_map = pool.map_async(
            run_genetic_algorithm,
            [(i + 1, params, log_file, shared_dict)
             for i in range(params.repeats)])
        pool.close()

        precision = 1 - int(math.log(params.epsilon, 10))

        # graceful way to interrupt all processes by Ctrl+C
        min_counter = 0
        while True:
            try:
                multiple_results = pool_map.get(
                    60 * params.time_for_print)
                break
            # catch TimeoutError and get again
            except multiprocessing.TimeoutError as ex:
                print_best_solution_now(
                    start_time, shared_dict, params, log_file, 
                    precision, draw_model=params.matplotlib_available)
            except Exception, e:
                pool.terminate()
                support.error(str(e))
        print_best_solution_now(start_time, shared_dict, params,log_file, 
                precision, draw_model=params.matplotlib_available)
        support.write_log(log_file, '\n--Finish pipeline--\n')
        if params.test:
            support.write_log(log_file, '--Test passed correctly--')
        if params.theta is None:
            support.write_log(
                log_file, "\nYou didn't specify theta at the beginning. If you want change it and rescale parameters, please see tutorial.\n")
        if params.resume_dir is not None and (params.initial_structure != params.final_structure).any():
            support.write_log(
                log_file, '\nYou have resumed from another launch. Please, check best AIC model, as information about it was lost.\n')

        support.write_log(log_file, 'Thank you for using GADMA!')

    except KeyboardInterrupt:
        pool.terminate()
        support.error('KeyboardInterrupt')


if __name__ == '__main__':
    main()
