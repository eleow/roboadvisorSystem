'''
    Algo 2
    - Pre-assembled fixed basket of ETFs based on VANGUARD series
    - Allocate and rebalance based on Modern Portfolio Theory with efficient frontier, at end of every month
    - Rebalance triggered when target allocation differs from current allocation by more than a threshold

'''

from zipline.api import order, record, symbols
from zipline.api import schedule_function, date_rules, time_rules
from utils import initialize_portfolio, optimal_portfolio, trigger_rebalance_on_threshold
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# inputs
VERBOSE = True
GRP = 'VANGUARD'
SUBGRP = 'CORE_SERIES'
STOCKS = None
# RISK_LEVEL = 5      # 1-9 for Tax Efficient Series, 0-10 otherwise
THRESHOLD = 0.05    # fixed threshold to trigger rebalance (0 to 1)
HISTORY = 100       # number of days to observe to get 'average returns'
COLLECT_BEFORE_TRADING = True  # collect data for HISTORY days before trading

TRADING_PLATFORM = 'vickers'
COUNTRY = 'SG'

all_portfolios = {}


def initialize(context):
    global all_portfolios

    if VERBOSE: print("Starting the robo advisor")

    # Populate Portfolios
    all_portfolios = initialize_portfolio(VERBOSE)

    # Set Commission model
    # initialize_commission(country=country, platform=trading_platform)

    # set universe and risk level
    context.stocks = all_portfolios[GRP][SUBGRP]['stocks'] if STOCKS is None else symbols(*STOCKS)  # context.stocks = symbols('VXUS', 'VTI', 'BND', 'BNDX')
    context.weights = False
    context.target_allocation = dict(zip(context.stocks, [0]*len(context.stocks)))  # initialise target allocations to zero
    print(context.target_allocation)

    context.tick = 0
    schedule_function(
        func=before_trading_starts,
        date_rule=date_rules.month_end(),
        time_rule=time_rules.market_open(hours=1))


def before_trading_starts(context, data):
    trigger_rebalance_on_threshold(context, data, rebalance, THRESHOLD)


def rebalance(context, data):
    allocate(context, data)
    for stock in context.stocks:
        current_weight = (data.current(
            stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
        target_weight = context.target_allocation[stock]
        distance = current_weight - target_weight
        if (distance > 0):
            amount = -1 * \
                (distance * context.portfolio.portfolio_value) / \
                data.current(stock, 'close')
            if (int(amount) == 0):
                continue
            if VERBOSE: print("Selling " + str(int(amount * -1)) + " shares of " + str(stock))
            order(stock, int(amount))
    for stock in context.stocks:
        current_weight = (data.current(
            stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
        target_weight = context.target_allocation[stock]
        distance = current_weight - target_weight
        if (distance < 0):
            amount = -1 * \
                (distance * context.portfolio.portfolio_value) / \
                data.current(stock, 'close')
            if (int(amount) == 0):
                continue
            if VERBOSE: print("Buying " + str(int(amount)) + " shares of " + str(stock))
            order(stock, int(amount))
    if VERBOSE: print('----')

    # record for use in analysis
    targets = list([(k.symbol, np.asscalar(v)) for k, v in context.target_allocation.items()])
    # print(type(targets), targets)
    record(allocation=targets)


def allocate(context, data):
    # Get rolling window of past prices and compute returns
    prices = data.history(context.stocks, 'price', HISTORY, '1d').dropna()
    returns = prices.pct_change().dropna()
    context.weights, _, _ = optimal_portfolio(returns.T)

    if (isinstance(context.weights, type(None))):
        if VERBOSE: print("Error in weights. Skipping")
    else:
        if VERBOSE: print("WEIGHTS:", str(context.weights))
        context.target_allocation = dict(
            zip(context.stocks, tuple(context.weights)))


def handle_data(context, data):
    # Allow history to accumulate 'HISTORY' days of prices before trading
    # and rebalance every x days (depending on schedule function) thereafter.
    context.tick += 1
    if COLLECT_BEFORE_TRADING and context.tick < HISTORY + 1:
        return

    if type(context.weights) == bool:
        allocate(context, data)
        for stock in context.stocks:
            amount = (
                context.target_allocation[stock] * context.portfolio.cash) / data.current(stock, 'price')
            # print(str(context.portfolio.starting_cash))
            # only buy if cash is allocated
            if (amount != 0):
                order(stock, int(amount))
                # log purchase
            if VERBOSE: print("buying " + str(int(amount)) + " shares of " + str(stock))


def analyze(context, perf):

    # https://matplotlib.org/tutorials/intermediate/gridspec.html
    gs1 = gridspec.GridSpec(2, 1)
    gs1.update(hspace=1) # set the spacing between axes.

    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    ax1 = plt.subplot(gs1[0])
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')

    m = []
    index = []
    columns = [l[0] for l in perf['allocation'][-1]]
    for k, v in perf['allocation'].items():
        if (type(v) == list):
            m.append(list(zip(*v))[1])
            # m.append((v[0][1], v[1][1], v[2][1], v[3][1]))
            index.append(k)

    df = pd.DataFrame(m, columns=columns)
    df.index = index  # by right, can just use allocation.index, but there are some NaN values

    # ax2 = fig.add_subplot(212)
    ax2 = plt.subplot(gs1[1])
    df.plot.area(ax=ax2)
    ax2.set_ylabel('Portfolio allocation')

    plt.legend(loc=0)
    plt.show()
