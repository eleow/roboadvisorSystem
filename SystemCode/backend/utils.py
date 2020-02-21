'''
    Utility functions to support trading algorithms

'''

from zipline.api import commission, set_commission, symbols, record, order, order_target_percent
# from configparser import ConfigParser
# import datetime
# import ast
import numpy as np
import pandas as pd
# import cvxopt as opt
# from cvxopt import blas, solvers

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import empyrical as ep
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

# PyPortfolioOpt imports
from pypfopt import risk_models, expected_returns
from pypfopt.cla import CLA
from pypfopt.base_optimizer import portfolio_performance
from pypfopt.efficient_frontier import EfficientFrontier

# solvers.options['show_progress'] = False  # no need to show progress of solving


# Typical SG commission is % of transaction with a minimum trade cost
class SGCommission(commission.EquityCommissionModel):
    def __init__(self, cost=-1, min_trade_cost=0, platform='vickers'):
        self.cost_per_dollar = float(cost)
        self.min_trade_cost = min_trade_cost or 0
        self.platform = platform

    def __repr__(self):
        return "{class_name}(cost_per_dollar={cost})".format(
            class_name=self.__class__.__name__,
            cost=self.cost_per_dollar)

    def calculate(self, order, transaction):
        """
        Pay commission based on dollar value of shares, with min trade
        """

        # For SG stocks
        gst = 0.07
        clearing_fee = 0.0325
        trading_fee = 0.0075

        trade_value = abs(transaction.amount) * transaction.price

        # if cost is unspecified, we will calculate based on brokeage rates
        if (self.cost_per_dollar == -1):
            # correct as of 18/12/2019.
            # Rates obtained from https://www.dbs.com.sg/vickers/en/pricing/fee-schedules/singapore-accounts
            if self.platform == 'vickers':
                if (trade_value <= 50000):
                    comm = 0.28
                elif (trade_value <= 100000):
                    comm = 0.22
                else:
                    comm = 0.18

                self.min_trade_cost = 25
                total_comm = (comm + clearing_fee + trading_fee)*(1+gst)/100

            cost_per_share = transaction.price * total_comm
        else:
            cost_per_share = transaction.price * self.cost_per_dollar

        return max(self.min_trade_cost, abs(transaction.amount) * cost_per_share)


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


def initialize_commission(country='SG', platform='vickers'):
    """Sets commissions

    See https://www.quantopian.com/help#ide-commission and
        https://www.quantopian.com/docs/api-reference/algorithm-api-reference#zipline.finance.commission.PerDollar
    """
    if (country == 'SG'):
        set_commission(SGCommission(platform=platform))


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


def optimal_portfolio(mu, S, objective='max_sharpe', get_entire_frontier=True):
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

    if (objective is None):
        # Get weights for both max_sharpe and min_volatility
        opt_weights = []
        op.max_sharpe()
        opt_weights.append(op.clean_weights())
        op.min_volatility()
        opt_weights.append(op.clean_weights())
    else:
        if (objective == 'max_sharpe'):
            op.max_sharpe()
        elif (objective == 'min_volatility'):
            op.min_volatility()

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
    if (verbose): print("-"*80 + "\ninimum Volatility Portfolio Allocation\n")
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

    return r_volatility, r_returns, opt_volatility, opt_returns


# The following are adapted from
# https://www.quantopian.com/posts/the-efficient-frontier-markowitz-portfolio-optimization-in-python
# https://github.com/quantopian/research_public/blob/master/research/Markowitz-Quantopian-Research.ipynb
# def optimal_portfolio(returns, N=100, verbose=1):
#     """Solve for optimal portfolio

#     Arguments:
#         returns: Returns for each asset. Numpy array (dimension of num_obv X num_assets),
#             where num_obv will be the number of past samples to take mean of
#         N: number of mus (expected returns). Note mus will be in a non-linear range

#     """
#     n = len(returns)
#     returns = np.asmatrix(returns)

#     mus = [10**(5.0 * t/N - 1.0) for t in range(N)]  # series of expected return values

#     # Convert to cvxopt matrices
#     S = opt.matrix(np.cov(returns))
#     pbar = opt.matrix(np.mean(returns, axis=1))

#     # Create constraint matrices
#     G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
#     h = opt.matrix(0.0, (n, 1))  # h = opt.matrix(-0.15, (n, 1))
#     A = opt.matrix(1.0, (1, n))
#     b = opt.matrix(1.0)

#     # Calculate efficient frontier weights using quadratic programming
#     portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
#     # CALCULATE RISKS AND RETURNS FOR FRONTIER
#     opt_returns = [blas.dot(pbar, x) for x in portfolios]
#     opt_risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
#     # CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
#     m1 = np.polyfit(opt_returns, opt_risks, 2)

#     # Guard against negative sqrt
#     if (m1[2] < 0):
#         return None, opt_returns, opt_risks

#     x1 = np.sqrt(m1[2] / m1[0])
#     # CALCULATE THE OPTIMAL PORTFOLIO
#     if verbose == 0: solvers.options['show_progress'] = False
#     wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
#     return np.asarray(wt), opt_returns, opt_risks


def rand_weights(n):
    """Produces n random weights that sum to 1"""
    k = np.random.rand(n)
    return k / sum(k)


# def get_mu_sigma(returns, weights=None):
#     """
#     Returns the mean and standard deviation of returns for a portfolio
#     Uses random weights if weights=None
#     """

#     p = np.asmatrix(np.mean(returns, axis=1))
#     w = np.asmatrix(rand_weights(returns.shape[0]) if weights is None else weights)
#     C = np.asmatrix(np.cov(returns))
#     mu = w * p.T
#     sigma = np.sqrt(w * C * w.T)

#     # This recursion reduces outliers to keep plots pretty
#     if sigma > 2:
#         return get_mu_sigma(returns, weights)
#     return mu, sigma


# def generate_markowitz_bullet(returns, n_portfolios=500, visualise=True, optimum=None, title='Markowitz Bullet'):
#     """Based on the assets returns,
#         1) randomly assign weights to get the markowitz bullet
#         2) get optimal portfolio

#     Arguments:
#         returns: Returns for each asset. Numpy array (dimension of num_obv X num_assets),
#             where num_obv will be the number of past samples to take mean of
#         n_portfolios: Number of portfolios to randomly generate for

#     """
#     means = []
#     stds = []

#     # random weighted portfolio
#     if (n_portfolios > 0):
#         means, stds = np.column_stack([get_mu_sigma(returns) for _ in range(n_portfolios)])

#     # optimum portfolio at the efficient frontier
#     if (optimum is None):
#         opt_weights, opt_returns, opt_risks = optimal_portfolio(returns)
#     else:
#         opt_weights, opt_returns, opt_risks = optimum

#     if visualise:
#         import matplotlib.pyplot as plt
#         plt.plot(stds, means, 'o', markersize=5)
#         plt.xlabel('std')
#         plt.ylabel('mean')
#         # plt.xlim(0, 2)
#         plt.plot(opt_risks, opt_returns, 'y-o')
#         plt.title(title)

#     return stds, means, opt_risks, opt_returns


def trigger_rebalance_on_threshold(context, data, rebalance_fn, threshold, verbose):
    """Trigger a rebalance if actual and target allocation differs by 'threshold'

    Arguments:
        rebalance_fn - rebalance function to execute
        threshold - value to trigger rebalance (0-1)

    """
    # total value of portfolio
    value = context.portfolio.portfolio_value  # value = context.portfolio.portfolio_value + context.portfolio.cash
    # calculating current weights for each position
    for stock in context.stocks:
        if ((context.target_allocation.get(stock, None) is None) or context.target_allocation[stock] == 0):
            continue
        current_holdings = data.current(stock, 'close') * context.portfolio.positions[stock].amount
        weight = current_holdings/value
        growth = float(weight) / float(context.target_allocation[stock])
        # if weights of any position exceed THRESHOLD, trigger rebalance
        if (growth >= 1 + threshold or growth <= 1 - threshold):
            rebalance_fn(context, data, verbose)
            break
    # print("No need to rebalance!")


def plot_rolling_returns_multiple(returns_arr, factor_returns=None, logy=False, ax=None, names_arr=None):
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
            cum_factor_returns.plot(lw=2, color='gray', label=factor_returns.name, alpha=0.60, ax=ax)

        is_cum_returns.plot(lw=1, alpha=0.6, label=returns.name, ax=ax)

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    major_fmt = mdates.DateFormatter('%b %Y')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(months)

    return ax


def record_allocation(context):
    """Record allocation data for use in analysis
    """
    # targets = list([(k.symbol, np.asscalar(v)) for k, v in context.target_allocation.items()])
    targets = list([(k.symbol, v) for k, v in context.target_allocation.items()])
    record(allocation=targets, cash=context.portfolio.cash)
    # print(type(targets), targets)


def record_current_weights(context, data):
    """Record current weights of portfolio for use in analysis
    """

    weights = []
    for stock in context.stocks:
        current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
        weights.append((stock.symbol, current_weight))

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


def analyze(context, perf):

    # https://matplotlib.org/tutorials/intermediate/gridspec.html
    gs1 = gridspec.GridSpec(2, 1)
    gs1.update(hspace=2.5)  # set the spacing between axes.

    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    ax1 = plt.subplot(gs1[0])
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_title('portfolio value in $')

    # Retrieve extra parameters that were stored
    # df_allocation = seriesToDataFrame(perf['allocation'])
    df_weights = seriesToDataFrame(perf['curr_weights'])

    # ax2 = fig.add_subplot(212)
    # ax2 = plt.subplot(gs1[1])
    # df_allocation.plot.area(ax=ax2)
    # ax2.set_title('Portfolio target allocation')
    # # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    #
    ax3 = plt.subplot(gs1[1])
    df_weights.plot.area(ax=ax3)
    ax3.set_title('Portfolio weights')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    # export to pickle for debugging
    import pickle
    with open('data.pickle', 'wb') as f:
        pickle.dump(perf['curr_weights'], f)


def rebalance_o(context, data, verbose):
    # allocate(context, data)
    if verbose: print("-"*30)

    # Sell first so that got more cash
    for stock in context.stocks:
        current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
        target_weight = context.target_allocation[stock]
        distance = current_weight - target_weight
        if (distance > 0):
            amount = -1 * (distance * context.portfolio.portfolio_value) / data.current(stock, 'close')
            if (int(amount) == 0):
                continue
            if verbose: print("Selling " + str(int(amount * -1)) + " shares of " + str(stock))
            order(stock, int(amount))

    # Buy after selling
    for stock in context.stocks:
        current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
        target_weight = context.target_allocation[stock]
        distance = current_weight - target_weight
        if (distance < 0):
            amount = -1 * (distance * context.portfolio.portfolio_value) / data.current(stock, 'close')
            if (int(amount) == 0):
                continue
            if verbose: print("Buying " + str(int(amount)) + " shares of " + str(stock))
            order(stock, int(amount))
    if verbose: print('-'*30)

    # record for use in analysis
    record_allocation(context)


def rebalance(context, data, verbose):
    """Rebalance portfolio

    If function enters, rebalance is deemed to be necessary, and rebalancing will be done
    """

    # allocate(context, data)
    if verbose: print("-"*30)

    # Just use order_target to rebalance?
    for stock in context.stocks:
        current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
        order_target_percent(stock, context.target_allocation[stock])
        if verbose: print("%s: %.5f -> %.5f" % (stock, current_weight, context.target_allocation[stock]))

    # record for use in analysis
    record_allocation(context)
