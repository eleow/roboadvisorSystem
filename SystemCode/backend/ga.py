'''
    Utility functions to Genetic algorithm using DEAP

    Created by Edmund Leow

'''
import random
from collections import OrderedDict
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
# import numpy
from tqdm import tqdm
import pickle
import pandas as pd

from algorithms import TradingSignalAlgorithm, OptAlgorithm, run
debug = True


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


def run_ga(fitnessType, npop, ngen, results_file, eval_fn, stocks, opt_type="saw", seed=64, **kwargs):
    random.seed(seed)

    if (fitnessType == "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    else:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
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
    population = toolbox.population(n=npop)

    print(f"Genetic Algorithm. Fitness: {fitnessType}, for {ngen} generations of {npop} population each. Seed={seed}")
    t_ngen = tqdm(range(ngen))
    for gen in t_ngen:
        t_ngen.set_description(f"Performing GA for generation {gen+1}/{ngen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

        # save at the end of each gen
        top10 = tools.selBest(population, k=10)
        top10 = [list(t) for t in top10]  # convert deap Individual to list

        # save latest (overwriting file if exists)
        with open(f"{results_file}.pickle", 'wb') as f:
            pickle.dump(top10, f)

        #  note down seed, pop size and number of generations for reproducibility, and save as a separate file
        post_fix = f"_p{npop}_g{ngen}_s{seed}"
        with open(f"{results_file}{post_fix}.pickle", 'wb') as f:
            pickle.dump(top10, f)

    return top10


def run_saw_ga(fitnessType, npop, ngen, results_file, eval_fn, stocks, **kwargs):
    return run_ga(fitnessType, npop, ngen, results_file, eval_fn, stocks, opt_type="saw", **kwargs)


def run_smpt_ga(fitnessType, npop, ngen, results_file, eval_fn, stocks, **kwargs):
    return run_ga(fitnessType, npop, ngen, results_file, eval_fn, stocks, opt_type="smpt", **kwargs)


def example():
    import pytz
    from datetime import datetime
    NPOP = 50
    NGEN = 10
    seed = 16
    stocks = ['VTI', 'TLT', 'IEF', 'GLD', 'DBC']  # list of stocks used by All-Weather
    bundle_name = 'robo-advisor_US'

    tz = pytz.timezone('US/Mountain')
    train_start = tz.localize(datetime.strptime('2018-07-21', '%Y-%m-%d'))
    train_end = tz.localize(datetime.strptime('2020-01-01', '%Y-%m-%d'))

    # test_start = tz.localize(datetime.strptime('2020-01-02', '%Y-%m-%d'))
    # test_end = tz.localize(datetime.strptime(args['end_date'], '%Y-%m-%d'))
    capital_base = 1000000
    trade_freq = 'weekly'

    filepath = 'data/twitter/sentiments_overall_daily.csv'
    social_media = pd.read_csv(filepath, usecols=['date', 'buzz', 'finBERT', 'sent12', 'sent26'])
    social_media['date'] = pd.to_datetime(social_media['date'], format="%Y-%m-%d", utc=True)
    social_media.set_index('date', inplace=True, drop=True)

    # top10_max_ret = run_ga("FitnessMax", NPOP, NGEN, "TEST_SAW_GA_MAX_RET",
    #                        eval_fn=eval_cumu_returns, stocks=stocks,
    #                        social_media=social_media, bundle_name=bundle_name,
    #                        train_start=train_start, train_end=train_end,
    #                        capital_base=capital_base, trade_freq=trade_freq
    #
    #

    kwargs = {"social_media": social_media, "bundle_name": bundle_name,
            "train_start": train_start, "train_end": train_end, "capital_base": capital_base, "trade_freq": trade_freq}

    # SMPT_GA
    # pickle_max_ret = "SMPT_GA_MAX_RET"
    # top10_max_ret = run_smpt_ga("FitnessMax", NPOP, NGEN, pickle_max_ret,
    #                               eval_fn=eval_cumu_returns, stocks=stocks, seed=seed, **kwargs)

    # SAW_GA_MAX_RET
    pickle_max_ret = "SAW_GA_MAX_RET"
    top10_max_ret = run_saw_ga("FitnessMax", NPOP, NGEN, pickle_max_ret,
                                eval_fn=eval_cumu_returns, stocks=stocks, seed=seed, **kwargs)

    # a_mpt2 = OptAlgorithm(verbose=False, grp="DALIO", subgrp="ALL_WEATHER",
    #                       collect_before_trading=False,
    #                       history=500, objective='min_volatility')

    # run("MPT (max sharpe)", a_mpt2, bundle_name, train_start, train_end, capital_base)

example()
