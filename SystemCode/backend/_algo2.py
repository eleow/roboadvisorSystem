'''
    Algo 2
    - Pre-assembled fixed basket of ETFs based on VANGUARD series
    - Allocate and rebalance based on Modern Portfolio Theory with efficient frontier, at end of every month
    - Rebalance triggered when target allocation differs from current allocation by more than a threshold

'''

from zipline.api import order, symbols
from zipline.api import schedule_function, date_rules, time_rules
from utils import initialize_portfolio, optimal_portfolio, get_mu_sigma, trigger_rebalance_on_threshold, rebalance, record_allocation, record_current_weights


# inputs
VERBOSE = True
GRP = 'VANGUARD'
SUBGRP = 'CORE_SERIES'
STOCKS = None
# RISK_LEVEL = 5      # 1-9 for Tax Efficient Series, 0-10 otherwise
THRESHOLD = 0.05    # fixed threshold to trigger rebalance (0 to 1)

COLLECT_BEFORE_TRADING = True  # collect data for HISTORY days before trading
HISTORY = 252       # number of days to observe to get 'average returns'. Typically 252 trading days in a year
FREQUENCY = 252     # number of time periods in a year, defaults to 252 (the number of trading days in a year). Used to get 'annualised' numbers

RETURNS_MODEL = 'mean_historical_return'
RISK_MODEL = 'ledoit_wolf'
OBJECTIVE = 'max_sharpe'

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
    if VERBOSE: print(context.stocks)

    context.tick = 0
    schedule_function(
        func=before_trading_starts,
        date_rule=date_rules.month_end(),
        time_rule=time_rules.market_open(hours=1))

    # record daily weights
    schedule_function(
        func=record_current_weights,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_open(hours=1),
    )


def before_trading_starts(context, data):
    allocate(context, data)  # get new optimum weights
    trigger_rebalance_on_threshold(context, data, rebalance, THRESHOLD, VERBOSE)  # trigger rebalance if exceed threshold

    # record_current_weights(context, data)  # record current weights


def allocate(context, data):
    # Get rolling window of past prices and compute returns
    prices = data.history(context.stocks, 'price', HISTORY, '1d').dropna()

    mu, S = get_mu_sigma(prices, RETURNS_MODEL, RISK_MODEL, FREQUENCY)
    context.weights, _, _ = optimal_portfolio(mu, S, OBJECTIVE, get_entire_frontier=False)
    context.weights = list(context.weights.values())

    # returns = prices.pct_change().dropna()
    # context.weights, _, _ = optimal_portfolio(returns.T)

    if (isinstance(context.weights, type(None))):
        if VERBOSE: print("Error in weights. Skipping")
    else:
        if VERBOSE: print("-"*30 + "\nOPTIMUM WEIGHTS:", str(context.weights))
        # context.target_allocation = dict(zip(context.stocks, tuple(context.weights)))
        context.target_allocation = dict(zip(context.stocks, context.weights))


def handle_data(context, data):
    # Allow history to accumulate 'HISTORY' days of prices before trading
    # and rebalance every x days (depending on schedule function) thereafter.
    context.tick += 1
    if COLLECT_BEFORE_TRADING and context.tick < HISTORY + 1:
        return

    # intial allocation
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

        # record for use in analysis
        record_allocation(context)
