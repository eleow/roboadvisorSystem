from abc import ABC, abstractmethod
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from zipline.api import schedule_function, date_rules, time_rules
from zipline.api import commission, set_commission, symbols, order
from utils import optimal_portfolio, get_mu_sigma
from utils import rebalance, record_allocation, record_current_weights
from utils import seriesToDataFrame, initialize_portfolio


###############################################################################
# Algorithm Classes
###############################################################################
class Algorithm(ABC):
    '''Base Algorithm class
    '''
    def __init__(self, name, verbose=False,
                 grp='VANGUARD', subgrp='CORE_SERIES',
                 threshold=0.05, stocks=None, country='US', trading_platform=''
                 ):
        self.name = name
        self.verbose = verbose
        self.grp = grp
        self.subgrp = subgrp
        self.threshold = threshold
        self.stocks = stocks
        self.country = country
        self.trading_platform = trading_platform

        self.all_portfolios = {}

    @abstractmethod
    def initialize(self, context):
        if self.verbose: print("Starting the robo advisor")

        # Populate Portfolios
        self.all_portfolios = initialize_portfolio(self.verbose)

        # Set Commission model
        self.initialize_commission(country=self.country, platform=self.trading_platform)

    @abstractmethod
    def handle_data(self, context, data):
        pass

    def analyze(self, context, perf):
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

    def initialize_commission(self, country='SG', platform='vickers'):
        """Sets commissions

        See https://www.quantopian.com/help#ide-commission and
            https://www.quantopian.com/docs/api-reference/algorithm-api-reference#zipline.finance.commission.PerDollar
        """
        if (country == 'SG'):
            set_commission(SGCommission(platform=platform))
        else:
            # TODO
            pass

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


class CRBAlgorithm(Algorithm):
    '''Constant Rebalanced Portfolio
    - Pre-assembled fixed basket of tickers (default: ETFs based on VANGUARD series)
    - Allocate and rebalance based on constant-mix of pre-defined risk buckets (risk-based allocation)
    - Rebalance triggered when target allocation differs from current allocation by more than a threshold
    '''

    def __init__(self, verbose=False, grp='VANGUARD', subgrp='CORE_SERIES', threshold=0.05,
                 stocks=None, country='US', trading_platform='', name='algo_constant_rebalanced',
                 risk_level=0):
        super(CRBAlgorithm, self).__init__(name, verbose, grp, subgrp, threshold, stocks, country, trading_platform)
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
        sum_allocation = sum(risk_based_allocation[self.risk_level])
        if sum_allocation > 1:
            risk_based_allocation[self.risk_level] = [r/sum_allocation for r in risk_based_allocation[self.risk_level]]

        context.target_allocation_initial = dict(zip(context.stocks, risk_based_allocation[self.risk_level]))
        context.target_allocation = context.target_allocation_initial
        context.bought = False

        schedule_function(
            func=self.before_trading_starts,
            date_rule=date_rules.every_day(),
            time_rule=time_rules.market_open(hours=1),
        )

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

        # ensure stocks exist. If does not, then we set that weight to zero, and normalise based on non-zero weights
        context.target_allocation = dict([t for t in context.target_allocation_initial.items() if t[0].start_date < context.datetime])
        sum_weights = sum(context.target_allocation.values())
        context.target_allocation = dict([(t[0], t[1]/sum_weights) for t in context.target_allocation.items()])

        self.trigger_rebalance_on_threshold(context, data, rebalance, self.threshold, self.verbose)
        record_current_weights(context, data)  # record current weights


class MPTAlgorithm(Algorithm):
    '''Modern Portfolio Theory / Mean-Variance Optimisation
    - Pre-assembled fixed basket of tickers (default: ETFs based on VANGUARD series)
    - Allocate and rebalance based on Modern Portfolio Theory with efficient frontier, at end of every month
    - Rebalance triggered when target allocation differs from current allocation by more than a threshold
    '''
    def __init__(self, verbose=False, grp='VANGUARD', subgrp='CORE_SERIES', threshold=0.05,
                 stocks=None, country='US', trading_platform='', name='algo_mpt_optimisation',
                 collect_before_trading=True, history=252, frequency=252,
                 returns_model='mean_historical_return', risk_model='ledoit_wolf', objective='max_sharpe'
                 ):
        super(MPTAlgorithm, self).__init__(name, verbose, grp, subgrp, threshold, stocks, country, trading_platform)
        # a = self.algo
        self.collect_before_trading = collect_before_trading
        self.history = history
        self.frequency = frequency
        self.returns_model = returns_model
        self.risk_model = risk_model
        self.objective = objective

    def initialize(self, context):
        super(MPTAlgorithm, self).initialize(context)

        # set universe and risk level
        context.stocks = self.all_portfolios[self.grp][self.subgrp]['stocks'] if self.stocks is None else symbols(*self.stocks)
        if self.verbose: print(context.stocks)
        context.weights = False
        context.target_allocation = dict(zip(context.stocks, [0]*len(context.stocks)))  # initialise target allocations to zero

        context.tick = 0
        schedule_function(
            func=self.before_trading_starts,
            date_rule=date_rules.month_end(),
            time_rule=time_rules.market_open(hours=1))

        # record daily weights
        schedule_function(
            func=record_current_weights,
            date_rule=date_rules.every_day(),
            time_rule=time_rules.market_open(hours=1),
        )

    def before_trading_starts(self, context, data):
        self.allocate(context, data)  # get new optimum weights
        self.trigger_rebalance_on_threshold(context, data, rebalance, self.threshold, self.verbose)  # trigger rebalance if exceed threshold

        # record_current_weights(context, data)  # record current weights

    def allocate(self, context, data):

        # Filter for stocks that exist 'today'
        stocks_that_exist = ([s for s in context.stocks if s.start_date < context.datetime])

        # Get rolling window of past prices and compute returns
        prices = data.history(stocks_that_exist, 'price', self.history, '1d').dropna()

        mu, S = get_mu_sigma(prices, self.returns_model, self.risk_model, self.frequency)
        context.weights, _, _ = optimal_portfolio(mu, S, self.objective, get_entire_frontier=False)
        context.weights = list(context.weights.values())

        # returns = prices.pct_change().dropna()
        # context.weights, _, _ = optimal_portfolio(returns.T)

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
                if (amount != 0):
                    order(stock, int(amount))
                if self.verbose: print("buying " + str(int(amount)) + " shares of " + str(stock))

            # record for use in analysis
            record_allocation(context)


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
