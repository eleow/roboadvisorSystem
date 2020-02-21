'''
    Algo 1
    - Pre-assembled fixed basket of ETFs based on VANGUARD series
    - Allocate and rebalance based on constant-mix of pre-defined risk buckets (risk-based allocation)
    - Rebalance triggered when target allocation differs from current allocation by more than a threshold

'''

from zipline.api import schedule_function, date_rules, time_rules
from utils import initialize_portfolio, trigger_rebalance_on_threshold, rebalance, record_current_weights

# inputs
VERBOSE = True
GRP = 'VANGUARD'
SUBGRP = 'CORE_SERIES'
RISK_LEVEL = 5      # 1-9 for Tax Efficient Series, 0-10 otherwise
THRESHOLD = 0.05    # fixed threshold to trigger rebalance (0 to 1)

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
    # config = ConfigParser()
    context.stocks = all_portfolios[GRP][SUBGRP]['stocks']
    risk_based_allocation = all_portfolios[GRP][SUBGRP]['levels']  # section_to_dict(subgrp, config)
    if VERBOSE: print(context.stocks)

    if (RISK_LEVEL not in risk_based_allocation):
        raise Exception("Portfolio Doesn't Have Risk Level " + str(RISK_LEVEL))
    context.target_allocation = dict(zip(context.stocks, risk_based_allocation[RISK_LEVEL]))
    context.bought = False

    schedule_function(
        func=before_trading_starts,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_open(hours=1),
    )


def handle_data(context, data):
    # if not context.bought:
    #     for stock in context.stocks:
    #         #Allocate cash based on weight, and then divide by price to buy shares
    #         amount = (context.target_allocation[stock] * context.portfolio.cash) / data.current(stock,'price')
    #         #only buy if cash is allocated
    #         if (amount != 0):
    #             order(stock, int(amount))
    #             #log purchase
    #         print("Buying " + str(int(amount)) + " shares of " + str(stock))
    #     #now won't purchase again and again
    context.bought = True


def before_trading_starts(context, data):
    trigger_rebalance_on_threshold(context, data, rebalance, THRESHOLD, VERBOSE)

    record_current_weights(context, data)  # record current weights


# def rebalance(context, data):
#     # Sell first so that got more cash
#     for stock in context.stocks:
#         # print(data.current(stock,'close'))
#         current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
#         target_weight = context.target_allocation[stock]
#         distance = current_weight - target_weight
#         if (distance > 0):
#             amount = -1 * (distance * context.portfolio.portfolio_value) / data.current(stock, 'close')
#             if (int(amount) == 0):
#                 continue
#             if VERBOSE: print("Selling " + str(abs(int(amount))) + " shares of " + str(stock))
#             order(stock, int(amount))

#     # Buy after selling
#     for stock in context.stocks:
#         current_weight = (data.current(stock, 'close') * context.portfolio.positions[stock].amount) / context.portfolio.portfolio_value
#         target_weight = context.target_allocation[stock]
#         distance = current_weight - target_weight
#         if (distance < 0):
#             amount = -1 * (distance * context.portfolio.portfolio_value) / data.current(stock, 'close')
#             if (int(amount) == 0):
#                 continue
#             if VERBOSE: print("Buying " + str(int(amount)) + " shares of " + str(stock))
#             order(stock, int(amount))
