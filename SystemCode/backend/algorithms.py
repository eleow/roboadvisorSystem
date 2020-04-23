'''
    Algorithms for trading using zipline

    Created by Edmund Leow

'''

from abc import ABC, abstractmethod
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from zipline import run_algorithm
from zipline.api import schedule_function, date_rules, time_rules
from zipline.api import commission, set_commission, symbols, order
from zipline.api import slippage, set_slippage
from utils import optimal_portfolio, get_mu_sigma, hrp_portfolio
from utils import rebalance, record_current_weights
from utils import seriesToDataFrame, initialize_portfolio
import pandas as pd
from pypfopt.base_optimizer import portfolio_performance


###############################################################################
# Algorithm Classes
###############################################################################
class Algorithm(ABC):
    '''Base Algorithm class
    '''
    def __init__(self, name, verbose=False,
                 grp='VANGUARD', subgrp='CORE_SERIES', threshold=0.05, rebalance_freq='monthly',
                 stocks=None, country='US', trading_platform='', **kwargs
                 ):
        """
        Arguments:
            name (str) - name of algorithm
            verbose (bool, optional) - True to show more messages, defaults to False
            grp (str, optional) - Portfolio group for universe (default VANGUARD)
            subgrp (str, optional) - Portfolio subgroup for universe (default CORE_SERIES)
            threshold (float, optional) - Distance deviation threshold from target allocation.
                If actual allocation is greater than the threshold, rebalance will be triggered
            stocks (list, optional) - List of stocks to consider in the universe.
                If None, universe will be determined by grp and subgrp (default None)
            country (str, optional) - Country where stocks are. This is used for commission models (default 'US')
            trading_platform (str, optional) - Trading platform for stocks. This is used for commission models
            rebalance_freq (str, optional) - How often to call rebalance function.
                Upon calling function, allocations will be compared with target allocations, and if it exceeds threshold,
                then rebalance will occur. Valid values are 'daily', 'weekly', 'monthly' (default 'monthly')

        """
        self.name = name
        self.verbose = verbose
        self.grp = grp
        self.subgrp = subgrp
        self.threshold = threshold
        self.stocks = stocks
        self.country = country
        self.trading_platform = trading_platform
        self.rebalance_freq = rebalance_freq

        self.all_portfolios = {}
        self.kwargs = kwargs
        # self.get_social_media()

    @abstractmethod
    def initialize(self, context):
        if self.verbose: print("Starting the robo advisor")

        # Populate Portfolios
        self.all_portfolios = initialize_portfolio(self.verbose)

        # Set Commission model
        self.initialize_commission(country=self.country, platform=self.trading_platform)

        # Set Slippage model
        set_slippage(slippage.FixedSlippage(spread=0.0))  # assume spread of 0

        # Schedule Rebalancing check
        rebalance_check_freq = date_rules.month_end()
        if (self.rebalance_freq == 'daily'): rebalance_check_freq = date_rules.every_day()
        elif (self.rebalance_freq == 'weekly'): rebalance_check_freq = date_rules.week_end()

        if self.verbose: print('Rebalance checks will be done %s' % self.rebalance_freq)
        schedule_function(
            func=self.before_trading_starts,
            date_rule=rebalance_check_freq,
            time_rule=time_rules.market_open(hours=1))

        # record daily weights at the end of each day
        schedule_function(
            func=record_current_weights,
            date_rule=date_rules.every_day(),
            time_rule=time_rules.market_close()
        )

        # # define target exposure
        # context.exposure = ExposureMngr(target_leverage=1.0,
        #                                 target_long_exposure_perc=1.0,
        #                                 target_short_exposure_perc=0.0)

    @abstractmethod
    def handle_data(self, context, data):
        pass

    @abstractmethod
    def before_trading_starts(self, context, data):
        pass

    def get_social_media(self, filepath='../data/twitter/sentiments_overall_daily.csv'):
        # self.sentiment = pd.read_csv(filepath, index_col='date')
        self.social_media = pd.read_csv(filepath, usecols=['date', 'buzz', 'finBERT', 'sent12'])
        self.social_media['date'] = pd.to_datetime(self.social_media['date'], format="%Y-%m-%d", utc=True)
        self.social_media.set_index('date', inplace=True, drop=True)

        # return pd.read_csv(filepath)

    def analyze(self, context, perf):

        # export to pickle for debugging
        import pickle
        with open('data.pickle', 'wb') as f:
            pickle.dump(perf, f)

        # https://matplotlib.org/tutorials/intermediate/gridspec.html
        gs1 = gridspec.GridSpec(3, 1)
        gs1.update(hspace=2.0)  # set the spacing between axes.

        # returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perf)
        # pf.create_full_tear_sheet(returns,
        #                           positions=positions,
        #                           transactions=transactions,
        #                           )

        # fig = plt.figure()
        # ax1 = fig.add_subplot(211)
        ax1a = plt.subplot(gs1[0])
        ax1b = ax1a.twinx()
        # perf.portfolio_value.columns = ['PV']
        perf.portfolio_value.plot(ax=ax1a, color='r', legend=None)  # portfolio value (cash + sum(shares * price))
        pd.DataFrame(perf['cash']).plot(ax=ax1b, color='b', legend=None)  # cash
        ax1a.set_ylabel('Portfolio value')
        ax1a.yaxis.label.set_color('red')
        ax1b.set_ylabel('Cash')
        ax1b.yaxis.label.set_color('blue')
        ax1a.set_title('Portfolio value and cash in $')

        # Social media scores
        # ax2a = plt.subplot(gs1[1])
        # ax2b = ax2a.twinx()
        # pd.DataFrame(perf['sentiment']).plot(ax=ax2a, color='r', legend=None)
        # pd.DataFrame(perf['buzz']).plot(ax=ax2b, color='b', legend=None)
        # ax2a.set_ylabel('Sentiment')
        # ax2a.yaxis.label.set_color('red')
        # ax2b.set_ylabel('Buzz')
        # ax2b.yaxis.label.set_color('blue')
        # ax2a.set_title('Sentiment and Buzz')
        # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # Plot actual weights over time
        # df_allocation = seriesToDataFrame(perf['allocation'])
        df_weights = seriesToDataFrame(perf['curr_weights'])
        ax3 = plt.subplot(gs1[1])
        df_weights.plot.area(ax=ax3)
        ax3.set_title('Portfolio weights')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.show()

    def initialize_commission(self, country='US', platform='IB'):
        """Sets commissions

        See https://www.quantopian.com/help#ide-commission and
            https://www.quantopian.com/docs/api-reference/algorithm-api-reference#zipline.finance.commission.PerDollar
        """
        if (country == 'SG'):
            set_commission(SGCommission(platform=platform))
        else:
            # IB broker for US is the top stock broker in US
            # https://brokerchooser.com/best-brokers/best-stock-brokers-in-the-us
            # Typical commission is USD 0.005 per share, minimum per order USD 1.00
            # https://www1.interactivebrokers.com/en/index.php?f=1590&p=stocks
            set_commission(commission.PerShare(cost=0.005, min_trade_cost=1))

    def trigger_rebalance_on_threshold(self, context, data, rebalance_fn, threshold, verbose):
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


class TradingSignalAlgorithm(Algorithm):
    '''Trades based on a trading signal
    '''

    def __init__(self, verbose=False, grp='VANGUARD', subgrp='CORE_SERIES', threshold=0.05, rebalance_freq='monthly',
                 stocks=None, country='US', trading_platform='', name='constant_rebalanced',
                 trading_signal=None, lookback=7, initial_weights=[1], normalise_weights=False, **kwargs):
        super(TradingSignalAlgorithm, self).__init__(name, verbose, grp, subgrp, threshold, rebalance_freq, stocks, country, trading_platform, **kwargs)
        self.get_trading_signal = trading_signal  # function to call to retrieve trading signal for a given date
        self.lookback = lookback
        self.initial_weights = initial_weights
        self.normalise_weights = normalise_weights
        # self.kwargs = kwargs

    def initialize(self, context):
        super(TradingSignalAlgorithm, self).initialize(context)

        # set universe
        context.stocks = self.all_portfolios[self.grp][self.subgrp]['stocks'] if self.stocks is None else symbols(*self.stocks)
        if self.verbose: print(context.stocks)

        # initialise weights and target allocation
        context.weights = False
        context.target_allocation = dict(zip(context.stocks, self.initial_weights))  # initialise target allocations

    def handle_data(self, context, data):
        pass

    def before_trading_starts(self, context, data):
        super(TradingSignalAlgorithm, self).before_trading_starts(context, data)
        self.allocate(context, data)  # get new optimum weights
        self.trigger_rebalance_on_threshold(context, data, rebalance, self.threshold, self.verbose)  # trigger rebalance if exceed threshold

    def allocate(self, context, data):
        # Filter for stocks that exist 'today'
        stocks_that_exist = ([s for s in context.stocks if s.start_date < context.datetime])

        # get buy/sell/hold signal (-1 to 1) and adjust target_allocation accordingly
        for s in stocks_that_exist:
            signal = self.get_trading_signal(s, context.datetime, self.lookback, **self.kwargs)
            context.target_allocation[s] += signal
            context.target_allocation[s] = min(max(context.target_allocation[s], 0), 1)

        # normalise target_allocations
        if (self.normalise_weights):
            sum_allocation = sum(context.target_allocation.values())
            if sum_allocation > 0:
                context.target_allocation = {k: v/sum_allocation for k, v in context.target_allocation.items()}

        context.stocks_that_exist = stocks_that_exist


class CRBAlgorithm(Algorithm):
    '''Constant Rebalanced Portfolio
    - Pre-assembled fixed basket of tickers (default: ETFs based on VANGUARD series)
    - Allocate and rebalance based on constant-mix of pre-defined risk buckets (risk-based allocation)
    - Rebalance triggered when target allocation differs from current allocation by more than a threshold
    '''

    def __init__(self, verbose=False, grp='VANGUARD', subgrp='CORE_SERIES', threshold=0.05, rebalance_freq='monthly',
                 stocks=None, country='US', trading_platform='', name='constant_rebalanced',
                 risk_level=0):
        super(CRBAlgorithm, self).__init__(name, verbose, grp, subgrp, threshold, rebalance_freq, stocks, country, trading_platform)
        self.risk_level = risk_level  # 1-9 for Tax Efficient Series, 0-10 otherwise

    def initialize(self, context):
        super(CRBAlgorithm, self).initialize(context)

        # set universe and risk level
        context.stocks = self.all_portfolios[self.grp][self.subgrp]['stocks']
        if self.verbose: print(context.stocks, [s.start_date for s in context.stocks])
        risk_based_allocation = self.all_portfolios[self.grp][self.subgrp]['levels']  # section_to_dict(subgrp, config)

        if (self.risk_level not in risk_based_allocation):
            raise Exception("Portfolio Doesn't Have Risk Level " + str(self.risk_level))

        # if sum of allocation weights is greater than 1, we will automatically normalise them
        allocation = risk_based_allocation[self.risk_level]

        sum_allocation = sum(allocation)
        if sum_allocation > 1:
            allocation = [r/sum_allocation for r in allocation]

        context.target_allocation_initial = dict(zip(context.stocks, allocation))
        context.target_allocation = context.target_allocation_initial
        context.bought = False

    def handle_data(self, context, data):
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

    def before_trading_starts(self, context, data):
        super(CRBAlgorithm, self).before_trading_starts(context, data)

        # ensure stocks exist. If does not, then we set that weight to zero, and normalise based on non-zero weights
        context.target_allocation = dict([t for t in context.target_allocation_initial.items() if t[0].start_date < context.datetime])
        sum_weights = sum(context.target_allocation.values())
        context.target_allocation = dict([(t[0], t[1]/sum_weights) for t in context.target_allocation.items()])

        self.trigger_rebalance_on_threshold(context, data, rebalance, self.threshold, self.verbose)
        # record_current_weights(context, data)  # record current weights


class OptAlgorithm(Algorithm):
    '''Optimisation portfolio using either Modern Portfolio Theory (MPT) / Mean-Variance Optimisation, or Hierarchical Risk Parity (HRP)

    - Pre-assembled fixed basket of tickers (default: ETFs based on VANGUARD series)
    - Allocate and rebalance based on MPT/ HRP, at end of every month
    - Rebalance triggered when target allocation differs from current allocation by more than a threshold

    objective (str)
    - 'max_sharpe' - optimise with MPT for maximum sharpe ratio.
    - 'min_volatility' - optimise with MPT for minimum volatility
    - 'hrp' - optimise using HRP

    '''
    def __init__(self, verbose=False, grp='VANGUARD', subgrp='CORE_SERIES', threshold=0.05, rebalance_freq='monthly',
                 stocks=None, country='US', trading_platform='', name='optimisation',
                 collect_before_trading=True, history=252, frequency=252,
                 returns_model='mean_historical_return', risk_model='ledoit_wolf', objective='max_sharpe',
                 mpt_adjustment=None, **kwargs
                 ):
        super(OptAlgorithm, self).__init__(name, verbose, grp, subgrp, threshold, rebalance_freq, stocks, country, trading_platform, **kwargs)
        # a = self.algo
        self.collect_before_trading = collect_before_trading
        self.history = history
        self.frequency = frequency
        self.returns_model = returns_model
        self.risk_model = risk_model
        self.objective = objective
        self.get_mpt_adjustment = mpt_adjustment  # adjustment function for optimal portfolio

    def initialize(self, context):
        super(OptAlgorithm, self).initialize(context)

        # set universe
        context.stocks = self.all_portfolios[self.grp][self.subgrp]['stocks'] if self.stocks is None else symbols(*self.stocks)
        if self.verbose: print(context.stocks)
        context.weights = False
        context.target_allocation = dict(zip(context.stocks, [0]*len(context.stocks)))  # initialise target allocations to zero

        context.tick = 0

    def before_trading_starts(self, context, data):
        super(OptAlgorithm, self).before_trading_starts(context, data)
        self.allocate(context, data)  # get new optimum weights
        self.trigger_rebalance_on_threshold(context, data, rebalance, self.threshold, self.verbose)  # trigger rebalance if exceed threshold

        # record_current_weights(context, data)  # record current weights

    def allocate(self, context, data):

        # Filter for stocks that exist 'today'
        stocks_that_exist = ([s for s in context.stocks if s.start_date < context.datetime])

        # Get rolling window of past prices and compute returns
        prices = data.history(stocks_that_exist, 'price', self.history, '1d').dropna()
        context.weights = self.get_weights(context, prices)

        if (isinstance(context.weights, type(None))):
            if self.verbose: print("Error in weights. Skipping")
        else:
            if self.verbose: print("-"*30 + "\nOPTIMUM WEIGHTS:", str(context.weights))
            # context.target_allocation = dict(zip(context.stocks, tuple(context.weights)))
            context.target_allocation = dict(zip(stocks_that_exist, context.weights))

        context.stocks_that_exist = stocks_that_exist

    def handle_data(self, context, data):
        # Allow history to accumulate 'HISTORY' days of prices before trading
        # and rebalance every x days (depending on schedule function) thereafter.
        context.tick += 1
        if self.collect_before_trading and context.tick < self.history + 1:
            return

        # intial allocation
        if type(context.weights) == bool:
            self.allocate(context, data)
            for stock in context.stocks_that_exist:
                amount = (context.target_allocation[stock] * context.portfolio.cash) / data.current(stock, 'price')
                # only buy if cash is allocated
                if (amount > 0):
                    order(stock, int(amount))
                if self.verbose: print("buying " + str(int(amount)) + " shares of " + str(stock))

            # record for use in analysis
            # record_allocation(context)

    def get_weights(self, context, prices):

        if self.objective == "hrp":
            # Hierarchical Risk Parity
            weights = hrp_portfolio(prices)
        else:
            # Modern Portfolio Theory
            mu, S = get_mu_sigma(prices, self.returns_model, self.risk_model, self.frequency)

            if (self.get_mpt_adjustment is None):
                weights, _, _ = optimal_portfolio(mu, S, self.objective, get_entire_frontier=False, **self.kwargs)
            else:
                w_max_sharpe, opt_ret, opt_risk = optimal_portfolio(mu, S, self.objective, get_entire_frontier=True)
                r, v, _ = portfolio_performance(mu, S, w_max_sharpe)

                v_adjusted = self.get_mpt_adjustment(context.datetime, v, **self.kwargs)
                weights, _, _ = optimal_portfolio(mu, S, "efficient_risk", get_entire_frontier=False, **{"target_volatility": v_adjusted})

        if type(weights) == dict: weights = list(weights.values())
        return weights


class BuyAndHoldAlgorithm(CRBAlgorithm):
    '''Buy-and-Hold some tickers based on initial weights
    '''

    def __init__(self, verbose=False,
                 grp='VANGUARD', subgrp='CORE_SERIES', name="BuyAndHold",
                 stocks=None, country='US', trading_platform='', **kwargs
                 ):

        super(BuyAndHoldAlgorithm, self).__init__(verbose, grp, subgrp, threshold=0.05, rebalance_freq='monthly',
                 stocks=stocks, country=country, trading_platform=trading_platform,
                 name=name, risk_level=0)

    def initialize(self, context):
        super(BuyAndHoldAlgorithm, self).initialize(context)
        context.bought = False

    def handle_data(self, context, data):
        if not context.bought:
            for stock in context.stocks:
                # Allocate cash based on weight, and then divide by price to buy shares
                amount = (context.target_allocation[stock] * context.portfolio.cash) / data.current(stock,'price')
                # only buy if cash is allocated
                if (amount != 0):
                    order(stock, int(amount))
                # print("Buying " + str(int(amount)) + " shares of " + str(stock))

        context.bought = True

    def before_trading_starts(self, context, data):
        pass


###############################################################################
# Commission Models
###############################################################################
class SGCommission(commission.EquityCommissionModel):
    '''Typical SG commission is % of transaction with a minimum trade cost
    '''
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
###############################################################################


def run(name, algo, bundle_name, start, end, capital_base, analyze=True):
    '''Helper to run algorithm
    '''
    return (name, run_algorithm(start=start, end=end,
                                initialize=algo.initialize, handle_data=algo.handle_data, analyze=algo.analyze if analyze else None,
                                capital_base=capital_base, environ=os.environ, bundle=bundle_name
                                ))
