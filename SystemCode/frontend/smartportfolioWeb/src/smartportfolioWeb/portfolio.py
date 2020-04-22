'''
  Functions to modify portfolio based on algorithm
'''
import sys
import pandas as pd
import numpy as np
from collections import OrderedDict
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
import hashlib
import os.path
import pickle

import pytz
from yahoofinancials import YahooFinancials
from pypfopt.base_optimizer import portfolio_performance

sys.path.append('../../../backend')  # add parent folder to sys path
from algorithms import CRBAlgorithm, OptAlgorithm, TradingSignalAlgorithm
from utils import hrp_portfolio, get_mu_sigma, optimal_portfolio, retrieve_social_media
from ga import saw_ga_trading_fn, smpt_ga_trading_fn

social_media_path = '../../../backend/data/twitter/sentiments_overall_daily.csv'
social_media = retrieve_social_media(social_media_path)


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


def get_allocation_CRB(tickers, stock_type):
    if stock_type == "ALL_WEATHER":
        target_allocation = [0.3, 0.4, 0.15, 0.075, 0.075]
    elif stock_type == "SPDR":
        target_allocation = [1] * 11

    target_allocation = [Decimal(t) for t in target_allocation]
    return dict(zip(tickers, target_allocation))


def get_allocation_MPT(tickers, objective, history=500, get_mpt_adjustment=None, **kwargs):

    _, df_prices = get_ticker_prices(tickers, history=history)
    if objective == "hrp":
        weights = hrp_portfolio(df_prices)
    else:
        returns_model = kwargs.get('returns_model', 'mean_historical_return')
        risk_model = kwargs.get('risk_model', 'ledoit_wolf')
        frequency = kwargs.get('frequency', 252)
        timezone = kwargs.get('timezone', 'US/Mountain')

        tz = pytz.timezone(timezone)
        t_end = datetime.now(tz)

        mu, S = get_mu_sigma(df_prices, returns_model, risk_model, frequency)

        if (get_mpt_adjustment is None):
            weights, _, _ = optimal_portfolio(mu, S, objective, get_entire_frontier=False, **kwargs)
        else:
            w_max_sharpe, opt_ret, opt_risk = optimal_portfolio(mu, S, objective, get_entire_frontier=True)
            r, v, _ = portfolio_performance(mu, S, w_max_sharpe)

            ga_model = kwargs.get('ga_model', 'SMPT_GA_MAX_RET_p200_g5_s2020')
            with open(f"../../../backend/data/ga/{ga_model}.pickle", "rb+") as f:
                top10_max_ret = pickle.load(f)
            kwargs['weights'] = top10_max_ret[0]

            v_adjusted = get_mpt_adjustment(t_end, v, **kwargs)
            weights, _, _ = optimal_portfolio(mu, S, "efficient_risk", get_entire_frontier=False,
                                              **{"target_volatility": v_adjusted})

    weights = {k: Decimal(v) for k, v in weights.items()}
    return weights


def get_allocation_SAW(tickers, trading_signal=saw_ga_trading_fn,
                       timezone='US/Mountain', social_media=None,
                       ga_model='SMPT_GA_MAX_RET_p200_g5_s2020',
                       initial_weights=[0.3, 0.4, 0.15, 0.075, 0.075]):

    tz = pytz.timezone(timezone)
    t_end = datetime.now(tz)

    with open(f"../../../backend/data/ga/{ga_model}.pickle", "rb+") as f:
        top10_max_ret = pickle.load(f)

    best_max_ret = top10_max_ret[0]
    w_max_ret = OrderedDict()

    i = 0
    for t in tickers:
        w_max_ret[t] = {"p": best_max_ret[i], "n": best_max_ret[i+1]}
        i = i + 2

    # get buy/sell/hold signal (-1 to 1) and adjust target_allocation accordingly
    weights = dict(zip(tickers, initial_weights))
    for t in tickers:
        signal = trading_signal(t, t_end, **{'weights': w_max_ret, 'social_media': social_media})
        weights[t] += signal
        weights[t] = min(max(weights[t], 0), 1)

    sum_allocation = sum(weights.values())
    if sum_allocation > 0:
        weights = {k: v/sum_allocation for k, v in weights.items()}

    weights = {k: Decimal(v) for k, v in weights.items()}
    return weights


def get_ticker_prices(tickers, timezone='US/Mountain', history=1):

    tz = pytz.timezone(timezone)
    t_end = datetime.now(tz)
    t_start = t_end - max(1, history) * pd.tseries.offsets.BDay()  # "today" might be  a weekend, so min history=1
    tickers = sorted(tickers)

    # generate hash based on input params, so that we can cache results
    md5input = f"{timezone}{t_end.strftime('%Y%m%d')}_{history}" + "-".join(tickers)
    md5hash = hashlib.md5(md5input.encode())
    filename = f"/cache/{md5hash.hexdigest()}.pickle"

    if (os.path.isfile(filename)):
        print("Retrieving ticker prices using cached file: " + filename)
        with open(filename, "rb+") as f:
            data = pickle.load(f)
        p = data["arr"]
        df_p = data["dataframe"]

    else:
        print("Retrieving ticker prices from Yahoo")
        yf = YahooFinancials(tickers)
        prices = yf.get_historical_price_data(
            t_start.strftime('%Y-%m-%d'),
            t_end.strftime('%Y-%m-%d'),
            'daily')

        p = []

        df_p = pd.DataFrame(columns=tickers)  # initialise df with all columns of tickers
        for t in tickers:
            adjclose = [Decimal(dp['adjclose']).quantize(Decimal('.001'), rounding=ROUND_DOWN) for dp in prices[t]['prices']]
            p.append(adjclose)

            # extract adjclose data for ticker t from price data, and assign to the relevant column of df
            df_p[t] = pd.DataFrame(prices[t]['prices'], columns=['formatted_date', 'adjclose']).set_index('formatted_date')['adjclose']

        data = {"arr": p, "dataframe": df_p}

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb+") as f:
            pickle.dump(data, f)

    return dict(zip(tickers, p)), df_p


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

    all_prices, _ = get_ticker_prices(list(list_stocks.keys()))

    total_value = 0
    for key in list_stocks.keys():
        s = list_stocks[key]["shares"]
        p = all_prices.get(key)[-1]
        list_stocks[key]["price"] = p
        list_stocks[key]["value"] = p * s
        total_value += list_stocks[key]["value"]

    return total_value


def calculate_portfolio(amt, transactions, ptype, stocks_type, criteria, ga_model):
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
        allocation = get_allocation_CRB(tickers, stocks_type)
    elif ptype in ["MPT", "HRP"]:
        allocation = get_allocation_MPT(tickers, criteria)
    elif ptype == "SAW":
        allocation = get_allocation_SAW(tickers, trading_signal=saw_ga_trading_fn, social_media=social_media, ga_model=ga_model)
    elif ptype == "SMPT":
        allocation = get_allocation_MPT(tickers, criteria,
            get_mpt_adjustment=smpt_ga_trading_fn, social_media=social_media, ga_model=ga_model)

    # Check if we have existing transactions for this portfolio, if so we should take the chance to rebalance
    existing_shares = {}
    for t in transactions:
        s_list = t['stocks']
        additional_shares = {s['ticker']: s['shares'] + existing_shares.get(s['ticker'], 0) for s in s_list}
        existing_shares.update(additional_shares)

    # get all tickers, and then get prices for all these tickers
    # just in case there were different tickers in prev transactions in this portfolio
    all_tickers = set(tickers + list(existing_shares.keys()))
    all_prices, _ = get_ticker_prices(all_tickers)

    existing_value = 0
    for k, v in existing_shares.items():
        p = all_prices.get(k)[-1]  # get latest price for stock
        existing_value += (p * v)

    portfolio_value = amt + existing_value

    stocks = []
    invested = 0
    transact_type = 'buy' if amt > 0 else 'sell'
    print(f'\nTo {transact_type} ${abs(amt):,.2f} for {ptype} {stocks_type} {criteria} portfolio')

    # Rebalance based on target allocation
    for t in tickers:
        p = all_prices.get(t)[-1]
        curr_w = p * existing_shares.get(t, 0) / Decimal(portfolio_value)
        target_w = allocation.get(t)
        distance = target_w - curr_w

        amount = int(Decimal(distance * portfolio_value) / p)
        commission = get_commission(abs(amount))

        if (amount != 0):
            bs = "Buy" if amount > 0 else "Sell"
            print(f"- {bs} {abs(amount)} shares of {t} @ ${p:,.2f} with commission ${commission:,.2f}")

            stocks.append({
                "ticker": t,
                "price/share": p,  # random.uniform(1, 2),
                "shares": amount,
                "commission": commission
            })

            invested += amount*p + commission

    print(f'* Actual amount bought/sold: ${abs(invested):,.2f}')
    return stocks, invested
