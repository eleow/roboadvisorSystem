'''
    Utility functions to Genetic algorithm using DEAP

    Created by Edmund Leow

'''
import glob
# import os
import re
import random
from collections import OrderedDict
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
# import numpy
import pickle
import pandas as pd
import numpy as np

from algorithms import TradingSignalAlgorithm, OptAlgorithm, run
from utils import isnotebook
debug = True

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def saw_ga_trading_fn(stock, date, lookback, **kwargs):

    w = kwargs.get("weights", {})
    social_media = kwargs.get("social_media", None)

    stockname = stock.symbol
    df = social_media
    yesterday_date = date - pd.Timedelta(days=1)
    yesterday_social_media = df.iloc[df.index.get_loc(yesterday_date, method='nearest')]
    sentiment = yesterday_social_media['sent12'] - yesterday_social_media['sent26']

    if sentiment > 0:
        signal = w[stockname]["p"]
    else:
        signal = w[stockname]["n"]

    return signal


def saw_eval_base(individual, stocks, bundle_name, train_start, train_end, capital_base, trade_freq, social_media):
    # convert individual weights into weights for the stocks
    # if debug: print("SAW EVAL BASE")
    w = OrderedDict()
    i = 0
    for s in stocks:
        w[s] = {"p": individual[i], "n": individual[i+1]}
        i = i + 2

    algo = TradingSignalAlgorithm(verbose=False, grp="DALIO", subgrp="ALL_WEATHER",
                                  collect_before_trading=False, history=500,
                                  rebalance_freq=trade_freq, trading_signal=saw_ga_trading_fn,
                                  initial_weights=[0.3, 0.4, 0.15, 0.075, 0.075], normalise_weights=True,
                                  **{"weights": w, "social_media": social_media})
    _, algo_results = run("SAW-GA", algo, bundle_name, train_start, train_end, capital_base, analyze=False)

    return algo_results


def smpt_ga_trading_fn(date, volatility, **kwargs):

    w = kwargs.get("weights", [])
    social_media = kwargs.get("social_media", None)

    # stockname = stock.symbol
    df = social_media
    yesterday_date = date - pd.Timedelta(days=1)
    yesterday_social_media = df.iloc[df.index.get_loc(yesterday_date, method='nearest')]
    sentiment = yesterday_social_media['sent12'] - yesterday_social_media['sent26']

    if sentiment > 0:
        signal = volatility * (1 + sentiment * w[0])  # bullish - take more risk
    else:
        signal = volatility * (1 + sentiment * w[1])  # bearish - take less risk

    # if sentiment > 0:
    #     signal = volatility * (1 + w[0])  # bullish - take more risk
    # else:
    #     signal = volatility * (1 - w[1])  # bearish - take less risk

    return signal


def smpt_eval_base(individual, stocks, bundle_name, train_start, train_end, capital_base, trade_freq, social_media):
    # if debug: print("SMPT EVAL BASE")
    w = list(individual)

    algo = OptAlgorithm(verbose=False, grp="DALIO", subgrp="ALL_WEATHER",
                        collect_before_trading=False, history=500,
                        rebalance_freq=trade_freq, mpt_adjustment=smpt_ga_trading_fn,
                        **{"weights": w, "social_media": social_media})
    _, algo_results = run("SMPT-GA", algo, bundle_name, train_start, train_end, capital_base, analyze=False)

    return algo_results


def eval_cumu_returns(individual, opt_type="saw", **kwargs):

    if opt_type == "saw":
        algo_results = saw_eval_base(individual, **kwargs)
    elif opt_type == "smpt":
        algo_results = smpt_eval_base(individual, **kwargs)

    r = algo_results.get("algorithm_period_return", [])
    return [r[-1].item()]  # maximise cumulative returns at end of in-sample


def eval_min_vol(individual, opt_type="saw", **kwargs):
    if opt_type == "saw":
        algo_results = saw_eval_base(individual, **kwargs)
    elif opt_type == "smpt":
        algo_results - smpt_eval_base(individual, **kwargs)

    return [algo_results["algo_volatility"].max().item()]  # minimise the maximum volatility throughout in-sample


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, results_file="results", post_fix=""):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.

       Modification of eaSimple from https://github.com/DEAP/deap/blob/master/deap/algorithms.py
       - Add saving at every generation
       - Add progress bar

    """
    def toolbox_eval(ind):
        t_ind.update()
        return toolbox.evaluate(ind)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    t_ngen = tqdm(range(0, ngen + 1), desc="Creating initial gen")
    t_ngen.update()

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    t_ind = tqdm(desc="Creating pop", leave=False, total=len(population))
    fitnesses = toolbox.map(toolbox_eval, invalid_ind)
    t_ind.reset()

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        t_ngen.set_description(f"Creating gen {gen}/{ngen}")

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        t_ind.total = len(invalid_ind)
        fitnesses = toolbox.map(toolbox_eval, invalid_ind)
        t_ind.reset()

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # save at the end of each gen
        top10 = tools.selBest(population, k=10)
        top10 = [list(t) for t in top10]  # convert deap Individual to list

        # save latest (overwriting file if exists)
        with open(f"{results_file}.pickle", 'wb') as f:
            pickle.dump(top10, f)

        #  note down seed, pop size and number of generations for reproducibility, and save as a separate file
        with open(f"{results_file}{post_fix}.pickle", 'wb') as f:
            pickle.dump(top10, f)

        t_ngen.update()

    t_ind.n = t_ind.total
    t_ind.close()
    t_ngen.close()
    return population, logbook


def run_ga(fitness_type, npop, ngen, results_file, eval_fn, stocks, opt_type="saw", seed=64, **kwargs):
    random.seed(seed)

    if (fitness_type == "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    else:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # toolbox.register("attr_float", random.random)
    toolbox.register("attr_float", random.uniform, -0.5, 0.5)
    toolbox.register("attr_float_small", random.uniform, -0.1, 0.1)

    if opt_type == "saw":
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(stocks)*2)
    elif opt_type == "smpt":
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float_small, n=2)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_fn, stocks=stocks, opt_type=opt_type, **kwargs)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # keep the best in hof, even in the event of extinction
    # also compile population statistics during evolution
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print(f"Genetic Algorithm. Fitness: {fitness_type}, for {ngen} generations of {npop} population each. Seed={seed}")
    post_fix = f"_p{npop}_g{ngen}_s{seed}"

    pop = toolbox.population(n=npop)
    pop, log = eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen,
                        stats=stats, halloffame=hof, verbose=False, results_file=results_file, post_fix=post_fix)

    top10 = tools.selBest(pop, k=10)
    top10 = [list(t) for t in top10]  # convert deap Individual to list

    return top10, log, hof


def run_saw_ga(fitness_type, npop, ngen, results_file, eval_fn, stocks, **kwargs):
    return run_ga(fitness_type, npop, ngen, results_file, eval_fn, stocks, opt_type="saw", **kwargs)


def run_smpt_ga(fitness_type, npop, ngen, results_file, eval_fn, stocks, **kwargs):
    return run_ga(fitness_type, npop, ngen, results_file, eval_fn, stocks, opt_type="smpt", **kwargs)


def compareResults(base_name="SAW_GA_MAX_RET", opt_type="saw",
                   social_media=None, bundle_name="",
                   train_start=None, test_start=None, test_end=None,
                   stocks=None, trade_freq='weekly',
                   capital_base=1000000, history=500, **kwargs):
    all_ga = []  # [bm_all_weather]
    test_ga = []  # [bm_aw_test]

    for file in glob.glob(f"{base_name}_p*.pickle"):
        m = re.search(f'{base_name}_p(\d*)_g(\d*)_s(\d*).*', file)
        pop = m.group(1)
        gen = m.group(2)
        seed = m.group(3)

        with open(f"{file}", "rb+") as f:
            top10_max_ret = pickle.load(f)

            if opt_type == "saw":
                best_max_ret = top10_max_ret[0]
                w_max_ret = OrderedDict()
                i = 0

                for s in stocks:
                    w_max_ret[s] = {"p": best_max_ret[i], "n": best_max_ret[i+1]}
                    i = i + 2

                algo = TradingSignalAlgorithm(verbose=False, grp="DALIO", subgrp="ALL_WEATHER",
                        collect_before_trading=False, history=history,
                        rebalance_freq=trade_freq, trading_signal=saw_ga_trading_fn,
                        initial_weights=[0.3, 0.4, 0.15, 0.075, 0.075], normalise_weights=True,
                        **{"weights": w_max_ret, "social_media": social_media})

            else:
                w_max_ret = top10_max_ret[0]

                algo = OptAlgorithm(verbose=False, grp="DALIO", subgrp="ALL_WEATHER",
                        collect_before_trading=False, history=history,
                        rebalance_freq=trade_freq, mpt_adjustment=smpt_ga_trading_fn,
                        **{"weights": w_max_ret, "social_media": social_media})

            saw_ga_test = run(f"{pop}-{gen}-{seed}", algo, bundle_name, test_start, test_end, capital_base, analyze=False)
            saw_ga_all = run(f"{pop}-{gen}-{seed}", algo, bundle_name, train_start, test_end, capital_base, analyze=False)
            test_ga.append(saw_ga_test)
            all_ga.append(saw_ga_all)

    return test_ga, all_ga


def example(npop=10, ngen=2, seed=10, capital_base=1000000, opt_type="saw", filepath='data/twitter/sentiments_overall_daily.csv'):
    import pytz
    from datetime import datetime
    # NPOP = 10
    # NGEN = 2
    # seed = 20080808
    stocks = ['VTI', 'TLT', 'IEF', 'GLD', 'DBC']  # list of stocks used by All-Weather
    bundle_name = 'robo-advisor_US'

    tz = pytz.timezone('US/Mountain')
    train_start = tz.localize(datetime.strptime('2018-07-21', '%Y-%m-%d'))
    train_end = tz.localize(datetime.strptime('2020-01-01', '%Y-%m-%d'))

    # test_start = tz.localize(datetime.strptime('2020-01-02', '%Y-%m-%d'))
    # test_end = tz.localize(datetime.strptime(args['end_date'], '%Y-%m-%d'))
    # capital_base = 1000000
    trade_freq = 'weekly'

    # filepath = 'data/twitter/sentiments_overall_daily.csv'
    social_media = pd.read_csv(filepath, usecols=['date', 'buzz', 'finBERT', 'sent12', 'sent26'])
    social_media['date'] = pd.to_datetime(social_media['date'], format="%Y-%m-%d", utc=True)
    social_media.set_index('date', inplace=True, drop=True)

    kwargs = {"social_media": social_media, "bundle_name": bundle_name,
            "train_start": train_start, "train_end": train_end, "capital_base": capital_base, "trade_freq": trade_freq}

    if opt_type.lower() == "smpt":
        # SMPT_GA
        pickle_max_ret = "SMPT_GA_MAX_RET"
        top10_max_ret, log, hof = run_smpt_ga("FitnessMax", npop, ngen, pickle_max_ret,
                                  eval_fn=eval_cumu_returns, stocks=stocks, seed=seed, **kwargs)
    elif opt_type.lower() == "saw":
        # SAW_GA_MAX_RET
        pickle_max_ret = "SAW_GA_MAX_RET"
        top10_max_ret, log, hof = run_saw_ga("FitnessMax", npop, ngen, pickle_max_ret,
                                eval_fn=eval_cumu_returns, stocks=stocks, seed=seed, **kwargs)


if __name__ == "__main__":
    example()
