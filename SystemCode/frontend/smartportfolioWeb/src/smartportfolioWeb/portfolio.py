'''
  Functions to modify portfolio based on algorithm
'''
import sys
import pandas as pd
from datetime import datetime
import pytz
from yahoofinancials import YahooFinancials
from decimal import Decimal, ROUND_DOWN

sys.path.append('../../../backend')  # add parent folder to sys path
from algorithms import CRBAlgorithm, OptAlgorithm, TradingSignalAlgorithm


def get_details_from_stock_type(stock_type):
    if (stock_type == 'SPDR'):
        grp = stock_type
        subgrp = "ALL_SECTORS"
        tickers = ['XLE', 'XLRE', 'XLF', 'XLV', 'XLC', 'XLI', 'XLY', 'XLP', 'XLB', 'XLK', 'XLU']
    elif (stock_type == 'ALL_WEATHER'):
        grp = 'DALIO'
        subgrp = 'ALL_WEATHER'
        tickers = ['VTI', 'TLT', 'IEF', 'GLD', 'DBC']

    return grp, subgrp, tickers


def get_allocation_CRB(stock_type, grp, subgrp):
    if stock_type == "ALL_WEATHER":
        target_allocation = [0.3, 0.4, 0.15, 0.075, 0.075]
    elif stock_type == "SPDR":
        target_allocation = [1] * 11

    target_allocation = [Decimal(t) for t in target_allocation]
    return target_allocation


def get_allocation_MPT(stock_type, grp, subgrp, criteria):

    # TODO
    # mpt = OptAlgorithm(verbose=False, grp=grp, subgrp=subgrp, rebalance_freq='monthly',
    #                    objective=criteria, collect_before_trading=False)
    # # mpt.initialize({})
    # target_allocation = mpt.context.target_allocation
    # print(target_allocation)
    # return target_allocation
    return []


def get_allocation_SAW(stock_type, grp, subgrp, criteria):
    # TODO

    return []


def get_allocation_SMPT(stock_type, grp, subgrp, criteria):
    # TODO

    return []


def get_ticker_price(ticker, timezone='US/Mountain'):
    tz = pytz.timezone(timezone)
    t_end = datetime.now(tz)
    t_start = t_end - 1 * pd.tseries.offsets.BDay()  # "today" might be  a weekend

    yf = YahooFinancials([ticker])
    prices = yf.get_historical_price_data(
        t_start.strftime('%Y-%m-%d'),
        t_end.strftime('%Y-%m-%d'),
        'daily')

    p = Decimal(prices[ticker]['prices'][-1]['adjclose']).quantize(Decimal('.01'), rounding=ROUND_DOWN)
    return p


def get_commission(num_shares, cost_per_share=0.005, minimum_per_order=1):
    # IB broker for US is the top stock broker in US
    # https://brokerchooser.com/best-brokers/best-stock-brokers-in-the-us
    # Typical commission is USD 0.005 per share, minimum per order USD 1.00
    # https://www1.interactivebrokers.com/en/index.php?f=1590&p=stock
    commission = max(cost_per_share * num_shares, minimum_per_order)
    return Decimal(commission)


def calculate_current_val(transactions):
    list_stocks = {}
    for t in transactions:
        for s in t["stocks"]:
            if s["ticker"] not in list_stocks:
                list_stocks[s["ticker"]] = {"shares": 0}
            list_stocks[s["ticker"]]["shares"] += s["shares"]

    total_value = 0
    for key in list_stocks.keys():
        s = list_stocks[key]["shares"]
        p = get_ticker_price(key)
        list_stocks[key]["price"] = p
        list_stocks[key]["value"] = p * s
        total_value += list_stocks[key]["value"]

    return total_value


def calculate_portfolio(transactions, ptype, stocks_type, criteria, amt):
    # Calculate portfolio and returns an array of dict (key=ticker, val= dict of price, shares, commission),
    #  and left-over cash
    # Example of dict
    #   {
    #        {
    #          "ticker": "ABC",
    #          "price/share": 1.2,
    #          "shares": 100,
    #          "commission": 10
    #        }
    #   }
    grp, subgrp, tickers = get_details_from_stock_type(stocks_type)

    if ptype == "CRB":
        allocation = get_allocation_CRB(stocks_type, grp, subgrp)
    elif ptype in ["MPT", "HRP"]:
        allocation = get_allocation_MPT(stocks_type, grp, subgrp, criteria)
    elif ptype == "SAW":
        allocation = get_allocation_SAW(stocks_type, grp, subgrp, criteria)
    elif ptype == "SMPT":
        allocation = get_allocation_SMPT(stocks_type, grp, subgrp, criteria)

    # Check if we have existing transactions for this portfolio, if so we should take the chance to rebalance
    existing_shares = {}
    for t in transactions:
        s_list = t['stocks']
        additional_shares = {s['ticker']: s['shares'] + existing_shares.get(s['ticker'], 0) for s in s_list}
        existing_shares.update(additional_shares)

    existing_value = 0
    for k, v in existing_shares.items():
        p = get_ticker_price(k)
        existing_value += (p * v)

    # Rebalance based on target allocation
    portfolio_value = amt + existing_value

    stocks = []
    invested = 0
    for i in range(0, len(tickers)):
        p = get_ticker_price(tickers[i])
        curr_w = p * existing_shares.get(tickers[i], 0) / Decimal(portfolio_value)
        target_w = allocation[i]
        distance = target_w - curr_w

        amount = int(Decimal(distance * portfolio_value) / p)
        commission = get_commission(abs(amount))

        bs = "Buy" if amount > 0 else "Sell"
        print(f"{bs} {abs(amount)} shares of {tickers[i]} @ ${p:.2f} with commission ${commission:.2f}")

        stocks.append({
            "ticker": tickers[i],
            "price/share": p,  # random.uniform(1, 2),
            "shares": amount,
            "commission": commission
        })

        invested += amount*p + commission

    return stocks, invested
