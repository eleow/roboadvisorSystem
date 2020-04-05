'''
  Functions to modify portfolio based on algorithm
'''
import random


def get_ticker_price(ticker):
    # TODO currently stub
    ticker_price = random.uniform(1, 2)
    return ticker_price


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


def calculate_portfolio(id, amt):
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

    # TODO currently stub
    stocks = []
    stock_tickers = 'VTI,TLT,IEF,GLD,DBC'.split(',')
    for i in range(0, 3):
        stocks.append({
            "ticker": stock_tickers[i],
            "price/share": random.uniform(1, 2),
            "shares": random.randint(100, 1000),
            "commission": 10
        })
    invested = amt - (amt % 10)  # just round to nearest $10
    # cash = amt - invested

    return stocks, invested
