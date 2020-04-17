'''
    Utility functions to support trading algorithms

    Created by Edmund Leow

'''
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

from zipline.api import symbols, record, order, order_target_percent
import empyrical as ep

# PyPortfolioOpt imports
from pypfopt import risk_models, expected_returns
from pypfopt.cla import CLA
from pypfopt.base_optimizer import portfolio_performance
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_risk_parity import HRPOpt

# Pyfolio imports
import pyfolio as pf
from pyfolio.utils import print_table
from pyfolio.timeseries import perf_stats


def add_portfolio(all_portfolios, group, subgroup, sym, risk_level):

    if group in all_portfolios:
        if subgroup not in all_portfolios[group]:
            all_portfolios[group][subgroup] = {}
    else:
        all_portfolios[group] = {}
        all_portfolios[group][subgroup] = {}

    all_portfolios[group][subgroup] = {
        'stocks': sym,
        'levels': risk_level
    }


def initialize_portfolio(verbose=False):
    """Initialises pre-assembled fixed basket of ETFs, each with their own predefined risk buckets

    Currently contains
        Vanguard ETF strategic model portfolios (https://advisors.vanguard.com/iwe/pdf/FASINVMP.pdf)
        - CORES_SERIES
        - CRSP_SERIES
        - SP_SERIES
        - RUSSELL_SERIES
        - INCOME_SERIES
        - TAX_SERIES

        Ray Dalio's All-Weather Portfolio

    Returns:
        dict containing all added portfolios

    """

    if verbose: print('Initialising portfolio database')
    all_portfolios = {}

    # 11 SPDR sector ETFs
    add_portfolio(all_portfolios, 'SPDR', 'ALL_SECTORS', symbols('XLE', 'XLRE', 'XLF', 'XLV', 'XLC', 'XLI', 'XLY', 'XLP', 'XLB', 'XLK', 'XLU'), {
        0: tuple(1 for _ in range(11))
    })

    # Ray Dalio's All Weather Portfolio. Rebalancing once a year or more, with the following suggested distributions:
    # * 30% stocks (eg VTI)
    # * 40% long-term bonds (eg TLT)
    # * 15% intermediate-term bonds (eg IEF)
    # * 7.5% gold (eg GLD)
    # * 7.5% commodities (eg DBC)
    add_portfolio(all_portfolios, 'DALIO', 'ALL_WEATHER', symbols('VTI', 'TLT', 'IEF', 'GLD', 'DBC'), {
        0: (0.3, 0.4, 0.15, 0.075, 0.075)
    })

    # Vanguard Core Serie
    add_portfolio(all_portfolios, 'VANGUARD', 'CORE_SERIES', symbols('VTI', 'VXUS', 'BND', 'BNDX'), {
        0: (0, 0, 0.686, 0.294),
        1: (0.059, 0.039, 0.617, 0.265),
        2: (0.118, 0.078, 0.549, 0.235),
        3: (0.176, 0.118, 0.480, 0.206),
        4: (0.235, 0.157, 0.412, 0.176),
        5: (0.294, 0.196, 0.343, 0.147),
        6: (0.353, 0.235, 0.274, 0.118),
        7: (0.412, 0.274, 0.206, 0.088),
        8: (0.470, 0.314, 0.137, 0.059),
        9: (0.529, 0.353, 0.069, 0.029),
        10: (0.588, 0.392, 0, 0)
    })
    # add_portfolio(all_portfolios, 'VANGUARD', 'CRSP_SERIES', symbols('VUG', 'VTV', 'VB', 'VEA', 'VWO', 'BSV', 'BIV', 'BLV', 'VMBS', 'BNDX'), {
    #     0: (0, 0, 0, 0, 0, 0.273, 0.14, 0.123, 0.15, 0.294),
    #     1: (0.024, 0.027, 0.008, 0.03, 0.009, 0.245, 0.126, 0.111, 0.135, 0.265),
    #     2: (0.048, 0.054, 0.016, 0.061, 0.017, 0.218, 0.112, 0.099, 0.12, 0.235),
    #     3: (0.072, 0.082, 0.022, 0.091, 0.027, 0.191, 0.098, 0.086, 0.105, 0.206),
    #     4: (0.096, 0.109, 0.03, 0.122, 0.035, 0.164, 0.084, 0.074, 0.09, 0.176),
    #     5: (0.120, 0.136, 0.038, 0.152, 0.044, 0.126, 0.07, 0.062, 0.075, 0.147),
    #     6: (0.143, 0.163, 0.047, 0.182, 0.053, 0.109, 0.056, 0.049, 0.06, 0.118),
    #     7: (0.167, 0.190, 0.055, 0.213, 0.061, 0.082, 0.042, 0.037, 0.045, 0.088),
    #     8: (0.191, 0.217, 0.062, 0.243, 0.071, 0.055, 0.028, 0.024, 0.030, 0.059),
    #     9: (0.215, 0.245, 0.069, 0.274, 0.079, 0.027, 0.014, 0.013, 0.015, 0.029),
    #     10: (0.239, 0.272, 0.077, 0.304, 0.088, 0, 0, 0, 0, 0)
    # })
    # add_portfolio(all_portfolios, 'VANGUARD', 'SP_SERIES', symbols('VOO', 'VXF', 'VEA', 'VWO', 'BSV', 'BIV', 'BLV', 'VMBS', 'BNDX'), {
    #     0: (0, 0, 0, 0, 0.273, 0.140, 0.123, 0.150, 0.294),
    #     1: (0.048, 0.011, 0.03, 0.009, 0.245, 0.126, 0.111, 0.135, 0.265),
    #     2: (0.097, 0.021, 0.061, 0.017, 0.218, 0.112, 0.099, 0.12, 0.235),
    #     3: (0.145, 0.031, 0.091, 0.027, 0.191, 0.098, 0.086, 0.105, 0.206),
    #     4: (0.194, 0.041, 0.0122, 0.035, 0.164, 0.084, 0.074, 0.09, 0.176),
    #     5: (0.242, 0.052, 0.152, 0.044, 0.136, 0.07, 0.062, 0.075, 0.147),
    #     6: (0.29, 0.063, 0.182, 0.053, 0.109, 0.056, 0.049, 0.06, 0.118),
    #     7: (0.339, 0.073, 0.213, 0.061, 0.082, 0.042, 0.037, 0.045, 0.088),
    #     8: (0.387, 0.083, 0.243, 0.071, 0.055, 0.028, 0.024, 0.03, 0.059),
    #     9: (0.436, 0.093, 0.274, 0.079, 0.027, 0.014, 0.013, 0.015, 0.029),
    #     10: (0.484, 0.104, 0.304, 0.088, 0, 0, 0, 0, 0)
    # })
    # add_portfolio(all_portfolios, 'VANGUARD', 'RUSSELL_SERIES', symbols('VONG', 'VONV', 'VTWO', 'VEA', 'VTWO', 'VEA', 'VWO', 'BSV', 'BIV', 'BLV', 'VMBS', 'BNDX'), {
    #     0: (0, 0, 0, 0, 0, 0.273, 0.14, 0.123, 0.15, 0.294),
    #     1: (0.028, 0.026, 0.005, 0.03, 0.009, 0.245, 0.126, 0.111, 0.135, 0.265),
    #     2: (0.056, 0.052, 0.01, 0.061, 0.017, 0.218, 0.112, 0.099, 0.086, 0.105, 0.206),
    #     3: (0.084, 0.079, 0.013, 0.091, 0.027, 0.191, 0.098, 0.086, 0.105, 0.206),
    #     4: (0.112, 0.105, 0.018, 0.122, 0.035, 0.164, 0.084, 0.074, 0.09, 0.176, 0.02),
    #     5: (0.14, 0.131, 0.023, 0.152, 0.044, 0.136, 0.07, 0.062, 0.075, 0.147),
    #     6: (0.168, 0.157, 0.028, 0.182, 0.053, 0.109, 0.056, 0.049, 0.06, 0.118),
    #     7: (0.196, 0.184, 0.032, 0.213, 0.061, 0.082, 0.042, 0.037, 0.045, 0.088),
    #     8: (0.224, 0.210, 0.036, 0.243, 0.071, 0.055, 0.028, 0.024, 0.03, 0.059),
    #     9: (0.252, 0.236, 0.041, 0.274, 0.079, 0.027, 0.014, 0.013, 0.015, 0.029),
    #     10: (0.281, 0.262, 0.045, 0.304, 0.088, 0, 0, 0, 0, 0)
    # })
    # add_portfolio(all_portfolios, 'VANGUARD', 'INCOME_SERIES', symbols('VTI', 'VYM', 'VXUS', 'VYMI', 'BND', 'VTC', 'BNDX'), {
    #     0: (0, 0, 0, 0, 0.171, 0.515, 0.294),
    #     1: (0.015, 0.044, 0.01, 0.029, 0.154, 0.463, 0.265),
    #     2: (0.03, 0.088, 0.019, 0.059, 0.137, 0.412, 0.235),
    #     3: (0.044, 0.132, 0.03, 0.088, 0.12, 0.36, 0.206),
    #     4: (0.059, 0.176, 0.039, 0.118, 0.103, 0.309, 0.176),
    #     5: (0.073, 0.221, 0.049, 0.147, 0.086, 0.257, 0.147),
    #     6: (0.088, 0.265, 0.059, 0.176, 0.068, 0.206, 0.118),
    #     7: (0.103, 0.309, 0.068, 0.206, 0.052, 0.154, 0.088),
    #     8: (0.117, 0.353, 0.079, 0.235, 0.034, 0.103, 0.059),
    #     9: (0.132, 0.397, 0.088, 0.265, 0.018, 0.051, 0.029),
    #     10: (0.147, 0.441, 0.098, 0.294, 0, 0, 0)
    # })
    # add_portfolio(all_portfolios, 'VANGUARD', 'TAX_SERIES', symbols('VUG', 'VTV', 'VB', 'VEA', 'VWO', 'VTEB'), {
    #     1: (0.024, 0.027, 0.008, 0.03, 0.009, 0.882),
    #     2: (0.048, 0.054, 0.016, 0.061, 0.017, 0.784),
    #     3: (0.072, 0.082, 0.022, 0.091, 0.027, 0.686),
    #     4: (0.096, 0.109, 0.03, 0.122, 0.035, 0.588),
    #     5: (0.12, 0.136, 0.038, 0.152, 0.044, 0.49),
    #     6: (0.143, 0.163, 0.047, 0.182, 0.053, 0.392),
    #     7: (0.167, 0.190, 0.055, 0.213, 0.061, 0.294),
    #     8: (0.191, 0.217, 0.062, 0.243, 0.071, 0.196),
    #     9: (0.215, 0.245, 0.069, 0.274, 0.079, 0.098)
    # })

    return all_portfolios


def get_mu_sigma(prices, returns_model='mean_historical_return', risk_model='ledoit_wolf',
                 frequency=252, span=500):
    """Get mu (returns) and sigma (asset risk) given a expected returns model and risk model

        prices (pd.DataFrame) – adjusted closing prices of the asset,
            each row is a date and each column is a ticker/id.
        returns_model (string, optional) - Model for estimating expected returns of assets,
            either 'mean_historical_return' or 'ema_historical_return' (default: mean_historical_return)
        risk_model (string, optional) - Risk model to quantify risk: sample_cov, ledoit_wolf,
            defaults to ledoit_wolf, as recommended by Quantopian in their lecture series on quantitative finance.
        frequency (int, optional) – number of time periods in a year, defaults to 252 (the number of trading days in a year)
        span (int, optional) – Applicable only for 'ema_historical_return' expected returns.
            The time-span for the EMA, defaults to 500-day EMA)
    """
    CHOICES_EXPECTED_RETURNS = {
        'mean_historical_return': expected_returns.mean_historical_return(prices, frequency),
        'ema_historical_return': expected_returns.ema_historical_return(prices, frequency, span)
    }

    CHOICES_RISK_MODEL = {
        'sample_cov': risk_models.sample_cov(prices),
        'ledoit_wolf': risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    }

    mu = CHOICES_EXPECTED_RETURNS.get(returns_model.lower(), None)
    S = CHOICES_RISK_MODEL.get(risk_model.lower(), None)

    if mu is None:
        raise Exception('Expected returns model %s is not supported. Only mean_historical_return and ema_historical_return are supported currently.' % risk_model)

    if S is None:
        raise Exception('Risk model %s is not supported. Only sample_cov and ledoit_wolf are supported currently.' % risk_model)

    return mu, S


def hrp_portfolio(prices):
    """Solve for Hierarchical risk parity portfolio

    Arguments:
        prices (pd.DataFrame) – adjusted (daily) closing prices of the asset, each row is a date and each column is a ticker/id.
    """

    returns = expected_returns.returns_from_prices(prices)
    hrp = HRPOpt(returns)
    weights = hrp.hrp_portfolio()
    return weights


def optimal_portfolio(mu, S, objective='max_sharpe', get_entire_frontier=True, **kwargs):
    """Solve for optimal portfolio. Wrapper for pypfopt functions

    Arguments:
        mu (pd.Series) - Expected annual returns
        S (pd.DataFrame/np.ndarray) - Expected annual volatility
        objective (string, optional) - Optimise for either 'max_sharpe', or 'min_volatility', defaults to 'max_sharpe'
        get_entire_frontier (boolean, optional) - Also get the entire efficient frontier, defaults to True

    """

    # if need to efficiently compute the entire efficient frontier for plotting, use CLA
    # else use standard EfficientFrontier optimiser.
    # (Note that optimum weights might be slightly different depending on whether CLA or EfficientFrontier was used)
    Optimiser = CLA if get_entire_frontier else EfficientFrontier
    op = Optimiser(mu, S)
    # risk_aversion = kwargs.get("risk_aversion", 1)  # only for max quadratic utility

    if (objective is None):
        # Get weights for both max_sharpe and min_volatility
        opt_weights = []
        op.max_sharpe()
        opt_weights.append(op.clean_weights())
        op.min_volatility()
        opt_weights.append(op.clean_weights())

        # ef = EfficientFrontier(mu, S)
        # ef.max_quadratic_utility(risk_aversion)
        # opt_weights.append(ef.clean_weights())
    else:
        if (objective == 'max_sharpe'):
            op.max_sharpe()
        elif ('min_vol' in objective):
            op.min_volatility()
        elif (objective == 'efficient_risk'):
            target_volatility = kwargs.get("target_volatility", None)
            if target_volatility is None:
                print("Error: You have to specify the target_volatility!")
                return None, None, None, None
            else:
                try:
                    op.efficient_risk(target_volatility)
                except ValueError:
                    # could not solve based on target_volatility, we try lookup table instead
                    cla = CLA(mu, S)
                    cla.max_sharpe()
                    ef_returns, ef_risks, ef_weights = cla.efficient_frontier(points=300)

                    lookup_v_w = dict(zip(ef_risks, ef_weights))
                    lookup_v_w = OrderedDict(sorted(lookup_v_w.items()))
                    w = lookup_v_w[min(lookup_v_w.keys(), key=lambda key: abs(key-target_volatility))]
                    w = [i[0] for i in w]  # flatten
                    return w, None, None

        elif (objective == 'efficient_return'):
            target_return = kwargs.get("target_return", None)
            if target_return is None:
                print("Error: You have to specify the target_return!")
                return None, None, None, None
            else:
                op.efficient_return(target_return)

        # elif (objective == 'max_quadratic_utility'):
        #     op.max_quadratic_utility(risk_aversion)
        #     # print("Using MAX_QUADRATIC UTILITY")

        opt_weights = op.clean_weights()

    if get_entire_frontier:
        opt_returns, opt_risks, _ = op.efficient_frontier(points=200)
        return opt_weights, opt_returns, opt_risks
    else:
        return opt_weights, None, None


def generate_markowitz_bullet(prices, returns_model='mean_historical_return', risk_model='ledoit_wolf',
                              frequency=252, span=500, objective='max_sharpe', num_random=20000,
                              ax=None, plot_individual=True, verbose=True, visualise=True):
    """Plot the markowitz bullet taking reference for plotting style from
        https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f

    Arguments:
        prices (pd.DataFrame) – adjusted closing prices of the asset, each row is a date and each column is a ticker/id.
        returns_model (string, optional) - Model for estimating expected returns of assets,
            either 'mean_historical_return' or 'ema_historical_return' (default: mean_historical_return)
        risk_model (string, optional) - Risk model to quantify risk: sample_cov, ledoit_wolf,
            defaults to ledoit_wolf, as recommended by Quantopian in their lecture series on quantitative finance.
        frequency (int, optional) – number of time periods in a year, defaults to 252 (the number of trading days in a year)
        span (int, optional) – Applicable only for 'ema_historical_return' expected returns.
            The time-span for the EMA, defaults to 500-day EMA)
        objective (string, optional) - Optimise for either 'max_sharpe', or 'min_volatility', defaults to 'max_sharpe'
        num_random (int, optional) - Number of random portfolios to generate for Markowitz Bullet. Set to 0 if not required
        plot_individual (boolean, optional) - If True, plots individual stocks on chart as well
        verbose (boolean, optional) - If True, prints out optimum portfolio allocations
        visualise (boolean, optional) - If True, plots Markowitz bullet

    Returns:
        r_volatility, r_returns, opt_volatility, opt_returns
        where
            r_volatility - array containing expected annual volatility values for generated random portfolios
            r_returns - array containg expected annual returns values for generated random portfolios
            opt_volatility - array containing expected annual volatility values along the efficient frontier
            opt_returns - array containing expected annual returns values along the efficient frontier

    """

    mu, S = get_mu_sigma(prices, returns_model, risk_model, frequency, span)
    opt_weights, opt_returns, opt_volatility = optimal_portfolio(mu, S, None, True)

    if (verbose): print("-"*80 + "\nMaximum Sharpe Ratio Portfolio Allocation\n")
    max_sharpe_returns, max_sharpe_volatility, max_sharpe_ratio = portfolio_performance(mu, S, opt_weights[0], verbose)
    if (verbose): print("-"*80 + "\nMinimum Volatility Portfolio Allocation\n")
    min_vol_returns, min_vol_volatility, min_vol_ratio = portfolio_performance(mu, S, opt_weights[1], verbose)

    if (visualise):
        plt.style.use('fivethirtyeight')

        if (ax is None): fig, ax = plt.subplots(figsize=(12, 8))

        # Plot Efficient Frontier (Annualised returns vs annualised volatility)
        ax.plot(opt_volatility, opt_returns, linestyle='-.', linewidth=1, color='black', label='Efficient frontier')

        # Plot optimum portfolios
        ax.plot(max_sharpe_volatility, max_sharpe_returns, 'r*', label='Max Sharpe', markersize=20)
        ax.plot(min_vol_volatility, min_vol_returns, 'g*', label='Min Volatility', markersize=20)

        # Plot individual stocks in 'prices' pandas dataframe
        if plot_individual:
            stock_names = list(prices.columns)
            s_returns = []
            s_volatility = []
            for i in range(len(stock_names)):
                w = [0] * len(stock_names)
                w[i] = 1
                s_returns, s_volatility, _ = portfolio_performance(mu, S, w)
                ax.plot(s_volatility, s_returns, 'o', markersize=10)
                ax.annotate(stock_names[i], (s_volatility, s_returns), xytext=(10, 0), textcoords='offset points')

        # Generate random portfolios
        if (num_random > 0):
            r_returns, r_volatility, r_sharpe = tuple(zip(*[portfolio_performance(mu, S, rand_weights(len(mu))) for _ in range(num_random)]))
            ax.scatter(r_volatility, r_returns, c=r_sharpe, cmap='YlGnBu', marker='o', s=10, alpha=0.3)  # random portfolios, colormap based on sharpe ratio

        # Set graph's axes
        ax.set_title('Markowitz Bullet')
        ax.set_xlabel('annualised volatility')
        ax.set_ylabel('annualised returns')
        ax.legend()

        plt.style.use('default')

    return r_volatility, r_returns, opt_volatility, opt_returns


def rand_weights(n):
    """Produces n random weights that sum to 1"""
    k = np.random.rand(n)
    return k / sum(k)


def print_table_from_perf_array(perf, factor_returns=None, show_baseline=False):
    APPROX_BDAYS_PER_MONTH = 21
    # APPROX_BDAYS_PER_YEAR = 252

    STAT_FUNCS_PCT = [
        'Annual return',
        'Cumulative returns',
        'Annual volatility',
        'Max drawdown',
        'Daily value at risk',
        'Daily turnover'
    ]

    arr = list(zip(*[(pData[0], pf.utils.extract_rets_pos_txn_from_zipline(pData[1])[0]) for pData in perf]))
    names_arr = arr[0]
    returns_arr = arr[1]

    # get headers
    returns = returns_arr[0]  # take first row as representative of all other backtests
    date_rows = OrderedDict()
    if len(returns.index) > 0:
        date_rows['Start date'] = returns.index[0].strftime('%Y-%m-%d')
        date_rows['End date'] = returns.index[-1].strftime('%Y-%m-%d')
        date_rows['Total months'] = int(len(returns) / APPROX_BDAYS_PER_MONTH)

    # get peformance stats
    perf_stats_arr = []

    # show baseline as one of the columns
    if show_baseline:
        perf_stats_arr.append(
            perf_stats(factor_returns, factor_returns=factor_returns)
        )
        names_arr = ['Baseline'] + list(names_arr)

    for i in range(len(returns_arr)):
        perf_stats_arr.append(
            perf_stats(returns_arr[i], factor_returns=factor_returns)
        )

    perf_stats_all = pd.concat(perf_stats_arr, axis=1)

    for column in perf_stats_all.columns:
        for stat, value in perf_stats_all[column].iteritems():
            if stat in STAT_FUNCS_PCT:
                perf_stats_all.loc[stat, column] = str(np.round(value * 100, 3)) + '%'
    df = pd.DataFrame(perf_stats_all)
    df.columns = names_arr

    # print table
    print_table(df, float_format='{0:.2f}'.format, header_rows=date_rows)


def plot_rolling_returns_from_perf_array(perf, factor_returns=None, extra_bm=0):
    """
    Plot cumulative rolling returns, given an array of performance data and benchmark

    Arguments:
    ----------
    perf (array of tuple of (string, pd.DataFrame))
    - Array of tuple of (run_name, performance). Performance is the output of zipline.api.run_algorithm

    factor_returns (pd.Series, optional)
    - Daily noncumulative returns of the benchmark factor to which betas are computed.
    - Usually a benchmark such as market returns. This is in the same style as returns.

    """
    arr = list(zip(*[(pData[0], pf.utils.extract_rets_pos_txn_from_zipline(pData[1])[0]) for pData in perf]))
    names_arr = arr[0]
    returns_arr = arr[1]

    ax = plot_rolling_returns_multiple(returns_arr, factor_returns, names_arr=names_arr, extra_bm=extra_bm)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax


def plot_rolling_returns_multiple(returns_arr, factor_returns=None, logy=False, ax=None, names_arr=None, extra_bm=0):
    """
    Plots cumulative rolling returns versus some benchmarks'.

    This is based on https://github.com/quantopian/pyfolio/blob/master/pyfolio/plotting.py,
    but modified to plot multiple rolling returns on the same graph

    Arguments
    ----------
    returns_arr : array of pd.Series. Each element contains daily returns of the strategy, noncumulative.t.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
        - This is in the same style as returns.
    logy : bool, optional
        Whether to log-scale the y-axis.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    names_arr: array of names for the plots, optional
    extra_bm: number of extra benchmarks. These will be assumed to be at the front of returns_array and will be plotted differently

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel('')
    ax.set_ylabel('Cumulative returns')
    ax.set_yscale('log' if logy else 'linear')

    for i in range(len(returns_arr)):
        # pData = perfData[i]
        # returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(pData)

        returns = returns_arr[i]
        returns.name = 'Portfolio %i' % i if names_arr is None else names_arr[i]

        cum_rets = ep.cum_returns(returns, 1.0)
        is_cum_returns = cum_rets
        if (i == 0 and factor_returns is not None):
            cum_factor_returns = ep.cum_returns(factor_returns[cum_rets.index], 1.0)
            cum_factor_returns.plot(lw=1, color='gray', label=factor_returns.name, alpha=0.60, ax=ax, style=['-.'])

        is_cum_returns.plot(lw=1, alpha=0.6, label=returns.name, ax=ax, style=['-.'] if (i < extra_bm) else None)
        # is_cum_returns.plot(lw=1, alpha=0.6, label=returns.name, ax=ax)

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    major_fmt = mdates.DateFormatter('%b %Y')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(months)

    return ax


def record_social_media(context):
    # print('recording social media')
    record(buzz=context.buzz, sentiment=context.sentiment)


# def record_allocation(context):
#     """Record allocation data for use in analysis
#     """
#     # targets = list([(k.symbol, np.asscalar(v)) for k, v in context.target_allocation.items()])
#     targets = list([(k.symbol, v) for k, v in context.target_allocation.items()])
#     record(allocation=targets, cash=context.portfolio.cash)
#     # print(type(targets), targets)


def record_current_weights(context, data):
    """Record current weights of portfolio for use in analysis
    """

    weights = []
    for stock in context.stocks:
        current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
        weights.append((stock.symbol, current_weight))

    targets = list([(k.symbol, v) for k, v in context.target_allocation.items()])
    record(allocation=targets, cash=context.portfolio.cash)
    record(curr_weights=weights)


def seriesToDataFrame(recorded_data):
    m = []
    index = []
    columns = [l[0] for l in recorded_data[-1]]
    for k, v in recorded_data.items():
        if (type(v) == list):
            m.append(list(zip(*v))[1])
            # m.append((v[0][1], v[1][1], v[2][1], v[3][1]))
            index.append(k)

    df = pd.DataFrame(m, columns=columns)
    df.index = index  # by right, can just use allocation.index, but there are some NaN values
    return df


# def rebalance_o(context, data, verbose):
#     # allocate(context, data)
#     if verbose: print("-"*30)

#     # Sell first so that got more cash
#     for stock in context.stocks:
#         current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
#         target_weight = context.target_allocation[stock]
#         distance = current_weight - target_weight
#         if (distance > 0):
#             amount = -1 * (distance * context.portfolio.portfolio_value) / data.current(stock, 'close')
#             if (int(amount) == 0):
#                 continue
#             if verbose: print("Selling " + str(int(amount * -1)) + " shares of " + str(stock))
#             print("-"*20)
#             print("BO ", context.portfolio.cash)
#             order(stock, int(amount))
#             print("AO ", context.portfolio.cash)

#     # Buy after selling
#     for stock in context.stocks:
#         current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
#         target_weight = context.target_allocation[stock]
#         distance = current_weight - target_weight
#         if (distance < 0):
#             amount = -1 * (distance * context.portfolio.portfolio_value) / data.current(stock, 'close')
#             if (int(amount) == 0):
#                 continue
#             if verbose: print("Buying " + str(int(amount)) + " shares of " + str(stock))
#             order(stock, int(amount))
#     if verbose: print('-'*30)

#     # record for use in analysis
#     # record_allocation(context)


def rebalance(context, data, verbose):
    """Rebalance portfolio

    If function enters, rebalance is deemed to be necessary, and rebalancing will be done
    """

    # allocate(context, data)
    if verbose: print("-"*30)

    # Just use order_target to rebalance?
    for stock in context.stocks:
        current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
        if stock in context.target_allocation:
            order_target_percent(stock, context.target_allocation[stock])
            if verbose: print("%s: %.5f -> %.5f" % (stock, current_weight, context.target_allocation[stock]))

    # record for use in analysis
    # record_allocation(context)


# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        elif (get_ipython().__class__.__module__ == "google.colab._shell"):
            return True  # Google Colab notebook
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
