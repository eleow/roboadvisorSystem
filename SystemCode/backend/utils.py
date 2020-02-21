'''
    Utility functions to support trading algorithms

'''

from zipline.api import commission, set_commission, symbols
# from configparser import ConfigParser
# import datetime
# import ast
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers

import matplotlib.pyplot as plt
import empyrical as ep
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

solvers.options['show_progress'] = False  # no need to show progress of solving


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


# The following are adapted from
# https://www.quantopian.com/posts/the-efficient-frontier-markowitz-portfolio-optimization-in-python
# https://github.com/quantopian/research_public/blob/master/research/Markowitz-Quantopian-Research.ipynb
def optimal_portfolio(returns, N=100, verbose=1):
    """Solve for optimal portfolio

    Arguments:
        returns: Returns for each asset. Numpy array (dimension of num_obv X num_assets),
            where num_obv will be the number of past samples to take mean of
        N: number of mus (expected returns). Note mus will be in a non-linear range

    """
    n = len(returns)
    returns = np.asmatrix(returns)

    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]  # series of expected return values

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))  # h = opt.matrix(-0.15, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
    # CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    # CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)

    # Guard against negative sqrt
    if (m1[2] < 0):
        return None, returns, risks

    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    if verbose == 0: solvers.options['show_progress'] = False
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


def rand_weights(n):
    """Produces n random weights that sum to 1"""
    k = np.random.rand(n)
    return k / sum(k)


def get_mu_sigma(returns, weights=None):
    """
    Returns the mean and standard deviation of returns for a portfolio
    Uses random weights if weights=None
    """

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]) if weights is None else weights)
    C = np.asmatrix(np.cov(returns))
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return get_mu_sigma(returns, weights)
    return mu, sigma


def generate_markowitz_bullet(returns, n_portfolios=500, visualise=True, optimum=None, title='Markowitz Bullet'):
    """Based on the assets returns,
        1) randomly assign weights to get the markowitz bullet
        2) get optimal portfolio

    Arguments:
        returns: Returns for each asset. Numpy array (dimension of num_obv X num_assets),
            where num_obv will be the number of past samples to take mean of
        n_portfolios: Number of portfolios to randomly generate for

    """
    means = []
    stds = []

    # random weighted portfolio
    if (n_portfolios > 0):
        means, stds = np.column_stack([get_mu_sigma(returns) for _ in range(n_portfolios)])

    # optimum portfolio at the efficient frontier
    if (optimum is None):
        opt_weights, opt_returns, opt_risks = optimal_portfolio(returns)
    else:
        opt_weights, opt_returns, opt_risks = optimum

    if visualise:
        import matplotlib.pyplot as plt
        plt.plot(stds, means, 'o', markersize=5)
        plt.xlabel('std')
        plt.ylabel('mean')
        # plt.xlim(0, 2)
        plt.plot(opt_risks, opt_returns, 'y-o')
        plt.title(title)

    return stds, means, opt_risks, opt_returns


def trigger_rebalance_on_threshold(context, data, rebalance, threshold):
    """Trigger a rebalance if actual and target allocation differs by 'threshold'

    Arguments:
        rebalance - rebalance function to execute
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
            rebalance(context, data)
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
