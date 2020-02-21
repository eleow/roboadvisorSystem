import os
import matplotlib.pyplot as plt
import pandas as pd
from yahoofinancials import YahooFinancials  # https://github.com/JECSand/yahoofinancials


def downloadData(args):
    dataDirectory = './data/'
    freq = args['freq']
    tickers = args['ticker'].split(',')

    # if 'ticker' is a filename, then we will read the list of tickers to get data for
    # TODO
    #

    def download_csv_data(ticker, start_date, end_date, freq, path):

        yahoo_financials = YahooFinancials(ticker)

        print('Downloading data for ', ticker, 'from ', start_date, ' to ', end_date)

        df = yahoo_financials.get_historical_price_data(start_date, end_date, freq)
        df = pd.DataFrame(df[ticker]['prices']).drop(['date'], axis=1) \
            .rename(columns={'formatted_date': 'date'}) \
            .loc[:, ['date', 'open', 'high', 'low', 'close', 'volume']] \
            .set_index('date')
        df.index = pd.to_datetime(df.index)
        df['dividend'] = 0
        df['split'] = 1

        # save data to csv for later ingestion
        dirPath = os.path.dirname(path)  # directory of file
        if not os.path.exists(dirPath):
            os.makedirs(dirPath, exist_ok=True)

        df.to_csv(path, header=True, index=True)

        # plot the time series
        if args['visualise']:
            df.close.plot(title='{} prices --- {}:{}'.format(ticker, start_date, end_date))
            plt.show()

    # print(os.path.join(dataDirectory, freq, tickers + '.csv'))

    for t in tickers:
        download_csv_data(ticker=t,
                          start_date=args['start_date'],
                          end_date=args['end_date'],
                          freq=freq,
                          path=os.path.join(dataDirectory, freq, t + '.csv'))
    print('\nNote that price data will differ slightly from Quantopian\'s')
    print(' See: https://www.quantopian.com/posts/incorrect-pricing-on-quantopian')

# Ingest our custom bundle 'roboadvisor'
# note: Remember to update the start and end dates in ~/.zipline/extension.py
# (See https://towardsdatascience.com/backtesting-trading-strategies-using-custom-data-in-zipline-e6fd65eeaca0)


# Due to bug in benchmarks.py, automatic downloading of benchmark data will fail
# Therefore, replace benchmarks.py and loader.py with files from setup folder
