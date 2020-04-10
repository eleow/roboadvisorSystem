from zipline.api import *
from zipline.api import symbols
from zipline import run_algorithm
from datetime import datetime
import pytz
import os


###############################################################################
# Reading in of input arguments
###############################################################################

import argparse
import sys
parser = argparse.ArgumentParser()
str2bool = lambda x: (str(x).lower() == 'true')
parser.add_argument("-m", "--mode", default='1', help="0-download data, 1-run backtest")
parser.add_argument("-v", "--visualise", type=str2bool, default=True, help="Bool indicating whether to visualise results")
parser.add_argument("-s", "--start_date", default='2015-01-05', help="Start-date of data in %y-%m-%d format")
parser.add_argument("-e", "--end_date", default='2020-04-05', help="End-date of data in %y-%m-%d format")


SPDR_sector_ETFs = 'XLE,XLRE,XLF,XLV,XLC,XLI,XLY,XLP,XLB,XLK,XLU'
ALL_WEATHER_ETFs = 'VTI,TLT,IEF,GLD,DBC'
VANGUARD = 'VTI,VXUS,BND,BNDX'
INDEX = 'SPY'

with open("data/sti.txt", "r") as f:
    STI = f.readlines()
STI = ",".join(STI).replace("\n", "")

# specific arguments for mode==0 (Download)
parser.add_argument("-f", "--freq", default='daily', help="Only for mode 0. Data frequency. 'daily', 'weekly' or 'monthly'")
# parser.add_argument("-t", "--ticker", default='NVDA,AAPL,FB,NFLX', help="Only for mode 0. Ticker(s) to get data for")
parser.add_argument("-t", "--ticker", default=VANGUARD, help="Only for mode 0. Ticker(s) to get data for")
# parser.add_argument("-t", "--ticker", default=ALL_WEATHER_ETFs, help="Only for mode 0. Ticker(s) to get data for")
# parser.add_argument("-t", "--ticker", default=SPDR_sector_ETFs, help="Only for mode 0. Ticker(s) to get data for")


# specific arguments for mode==1 (Backtest)
parser.add_argument("-b", "--bundle", default='robo-advisor_US', help="Only for mode 1. Data bundle to use")
parser.add_argument("-tz", "--timezone", default='US/Mountain', help="Only for mode 1. Timezone")
parser.add_argument("-c", "--capital", type=int, default=100000, help="Only for mode 1. Starting capital for backtest")
parser.add_argument("-bm", "--benchmark", default='SPY', help="Only for mode 1. Benchmark ticker")
#
#

# args = parser.parse_args()
args = vars(parser.parse_args())

if len(sys.argv) == 0:
    parser.print_help()
    sys.exit(1)
# else:
#     print(args['mode'])

if (args['mode'] == '0'):
    # Mode 0 - Download data
    from _download import downloadData
    downloadData(args)

elif (args['mode'] == '1'):
    # Mode 1 - Back-testing

    tz = pytz.timezone(args['timezone'])
    bundle = args['bundle']
    # bundle = 'alpaca'

    start = tz.localize(datetime.strptime(args['start_date'], '%Y-%m-%d'))
    end = tz.localize(datetime.strptime(args['end_date'], '%Y-%m-%d'))
    capital_base = args['capital'] # 100000.00

    # get the initialize and handle_data functions from another python file
    # from 'buy-and-hold' import initialize, handle_data
    b = __import__("robo-advisor")
    initialize = b.initialize
    handle_data = b.handle_data

    import matplotlib.pyplot as plt

    def analyze(context, perf):
        # use this to automatically visualise after running algorithm, if desired
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        perf.portfolio_value.plot(ax=ax1)
        ax1.set_ylabel('portfolio $ value')
        plt.legend(loc=0)
        plt.show()

    # Run the algorithm from within the py file
    perfData = run_algorithm(start=start,
                             end=end,
                             initialize=initialize,
                             # analyze=analyze,
                             capital_base=capital_base,
                             handle_data=handle_data,
                             environ=os.environ,
                             bundle=bundle
                             )


    # # visualise data
    # # import matplotlib.pyplot as plt
    # # from matplotlib import style

    # # style.use('ggplot')
    # # perfData.portfolio_value.plot()
    # # plt.show()

    # # print('Returns: ', perfData.returns)
    # # print('Alpha: ', perfData.alpha)
    # # print('Beta: ', perfData.beta)
    # # print('Sharpe: ', perfData.sharpe)
    # # print('Drawdown: ', perfData.max_drawdown)

    # Use pyfolio (by Quantopian) to generate tearsheet
    import pyfolio as pf
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perfData)
    # print(positions.tail())

    # Get benchmark returns and compare it with our algorithm's returns
    from zipline.data.benchmarks import get_benchmark_returns
    bm_returns = get_benchmark_returns(args['benchmark'], start, end)  # requires network connection

    bm_returns.name = 'Benchmark (%s)' % args['benchmark']
    returns.name = 'Algorithm'
    ax = plt.gca()

    pf.plot_rolling_returns(returns, factor_returns=bm_returns, logy=False, ax=ax)
    # pf.plot_rolling_returns(returns, logy=False, ax=ax)

    # pf.create_returns_tear_sheet(returns)
    # pf.create_full_tear_sheet(perfData, positions=positions, transactions=transactions,
    #                           live_start_date='2018-08-1', round_trips=False)

    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    major_fmt = mdates.DateFormatter('%b %Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(months)

    plt.show()
    print()
    # Total Returns - The total percentage return of the portfolio from the start to the end of the backtest
    # Sharpe Ratio - 6-month rolling Sharpe ratio (risk-free rate of 0)
    # Max Drawdown - The largest peak-to-trough drop in the portfolio's history.
    # Volatility - The standard deviation of the portfolio’s returns.
    pf.show_perf_stats(returns, factor_returns=bm_returns)
    plt.show()
