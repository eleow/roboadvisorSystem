#!/usr/bin/env python
# coding: utf-8

# ## Setup - Do once (Only in Colab)

# In[1]:


try:
    import google.colab
    IN_COLAB = True

    from google.colab import drive
    drive.mount('/content/drive')

    import os
    os.chdir('/content/drive/My Drive/Colab Notebooks/RoboAdvisor')
    print('Working directory changed to ' + os.getcwd())

except:
    IN_COLAB = False



# In[2]:


if (IN_COLAB):
    get_ipython().run_line_magic('capture', '')
    get_ipython().system('pip install zipline')
    get_ipython().system('pip install pyfolio')
    get_ipython().system('pip install cvxopt')
    get_ipython().system('pip install yahoofinancials')


# In[3]:


if (IN_COLAB):
    # Some files need to be modified
    get_ipython().system('ls /root')
    #!cp setup/extension.py ~/.zipline/extension.py
    get_ipython().system('cp setup/benchmarks.py /usr/local/lib/python3.6/dist-packages/zipline/data/benchmarks.py')
    get_ipython().system('cp setup/loader.py /usr/local/lib/python3.6/dist-packages/zipline/data/loader.py')


# ## Input and Function definitions

# In[1]:


#@title Input Variables
_visualise = True #@param ["True", "False"] {type:"raw"}
_start_date = '2015-01-05' #'2015-01-05' #@param {type:"date"}
_end_date = '2019-12-27' #@param {type:"date"}
_capital = 1000000 #@param {type:"slider", min:1000, max:1000000, step:1000}
_benchmark = 'SPY' #@param ["SPY"] {allow-input: true}
_history = 500 #@param {type:"slider", min:100, max:500, step:1}
_calendar = 'NYSE' #@param ["NYSE"] {allow-input: true}

args = {
    'mode': '1',
    'visualise': _visualise,
    'start_date': _start_date,
    'end_date': _end_date,
    'bundle': 'robo-advisor_US',
    'timezone': 'US/Mountain',
    'calendar': _calendar,
    'capital': _capital,
    'benchmark': _benchmark,
    'history': _history
}


# ### Register and ingest bundle

# In[2]:


import pandas as pd
from zipline.data.bundles import register, ingest, unregister, bundles
from zipline.data.bundles.csvdir import csvdir_equities

import os
from zipline.utils.run_algo import load_extensions

start_session = pd.Timestamp(args['start_date'], tz='utc')
end_session = pd.Timestamp(args['end_date'], tz='utc')
bundle_name = args['bundle']

load_extensions(default=True, extensions=[], strict=True, environ=os.environ)

# unregister bundle if already exists
if bundle_name in list(bundles):
    unregister(bundle_name)

# register and ingest the bundle
register(
    bundle_name,  # name we select for the bundle
    csvdir_equities(
        ['daily'], # name of the directory as specified above (named after data frequency)
        './data', # path to directory containing the data
    ),
    calendar_name=args['calendar'],  # US Equities
    start_session=start_session,
    end_session=end_session
)


ingest(bundle_name)


# In[3]:


# Verify that bundle has been registered
from zipline.data import bundles
bundle = bundles.load(bundle_name)
assets = bundle.asset_finder.retrieve_all(bundle.asset_finder.sids)
symbols = [a.symbol for a in assets]

print('Bundle details:', bundles.bundles[bundle_name])
print('\nAsset list:', bundle.asset_finder.retrieve_all(bundle.asset_finder.sids))
print()

# See sample of data
from zipline.data.data_portal import DataPortal
from zipline.utils.calendars import get_calendar
my_data = DataPortal(bundle.asset_finder, get_calendar(args['calendar']),
                       bundle.equity_daily_bar_reader.first_trading_day,
                       equity_minute_reader=bundle.equity_minute_bar_reader,
                       equity_daily_reader=bundle.equity_daily_bar_reader,
                       adjustment_reader=bundle.adjustment_reader)

my_data_pricing = my_data.get_history_window(assets, end_session, 100, '1d', 'close', 'daily')
my_data_pricing.plot()


# ## Run Algorithms

# In[4]:


# initialise variables
import matplotlib.pyplot as plt
from zipline.api import *
from zipline.api import symbols
from zipline.utils.calendars import get_calendar
from zipline import run_algorithm
from datetime import datetime, timedelta
import pytz

tz = pytz.timezone(args['timezone'])
bundle_name = args['bundle'] # bundle = 'alpaca'

raw_start = tz.localize(datetime.strptime(args['start_date'], '%Y-%m-%d'))

# calculate actual start date,
# taking into account required HISTORY of trading days for given trading calendar
tc = get_calendar(args['calendar'])
start = tc.sessions_window(start_session, args['history'])[-1] # start = raw_start + trading_days(HISTORY)
end = tz.localize(datetime.strptime(args['end_date'], '%Y-%m-%d'))
capital_base = args['capital'] # 100000.00

print(f'Algorithm will start with capital of ${capital_base:,d}.')
print('Backtest from %s to %s' % (start, end))
print('History will be collected for %d trading-days from %s' % (args['history'], raw_start))


# ### Portfolios with universe of 11 SPDR sector ETFs

# In[5]:


u = __import__("utils")

# Constant-Rebalancing with equal-weights
# spy1 = __import__("algo_constant_rebalanced")
spy1 = __import__("algo_mpt_optimisation")
spy1.VERBOSE = True
spy1.GRP = 'SPDR'
spy1.SUBGRP = 'ALL_SECTORS'
spy1.RISK_LEVEL = 0

if ('perfSPDR' not in locals()): perfSPDR = []
perfSPDR.append(("Constant-Rebalancing (Equal weights)", run_algorithm(start=start, end=end,
        initialize=spy1.initialize, handle_data=spy1.handle_data,
        capital_base=capital_base, environ=os.environ, bundle=bundle_name,
        analyze=u.analyze
)))


# # Testing area

# ### Toy-example for Modern Portfolio Theory (MPT)
# Modern portfolio theory applied on randomly-generated returns for a number of assets
#
# Adapted from https://www.quantopian.com/posts/the-efficient-frontier-markowitz-portfolio-optimization-in-python

# In[4]:


import numpy as np
import pandas as pd
from utils import optimal_portfolio, generate_markowitz_bullet
from pypfopt import expected_returns

n_assets = 4 # number of different stocks
n_obs = 1000 # number of history points to take mean of
return_vec = np.random.randn(n_obs, n_assets) # randomly create returns

prices_ = expected_returns.prices_from_returns(pd.DataFrame(return_vec))
_ = generate_markowitz_bullet(prices_, num_random=20000, plot_individual=False)


# ### Modern Portfolio Theory (MPT) applied on actual stocks
# Using assets selected in Vanguard ETF strategic model portfolios (Core Series), according to
# https://advisors.vanguard.com/iwe/pdf/FASINVMP.pdf

# In[15]:


import pandas as pd
from yahoofinancials import YahooFinancials  # https://github.com/JECSand/yahoofinancials

end = pd.Timestamp.utcnow()
start = end - 2500 * pd.tseries.offsets.BDay()

# Vanguard Core Series
tickers = ['VTI', 'VXUS', 'BND', 'BNDX']

yahoo_financials = YahooFinancials(tickers)
data = yahoo_financials.get_historical_price_data(
    start.strftime('%Y-%m-%d'),
    end.strftime('%Y-%m-%d'),
    'daily'
)


# In[16]:


import plotly.graph_objects as go
import numpy as np

SIZE = 100
data_title = 'VANGUARD Core Series'

df_data_all = {}
df_data = {} # only last n

for i in range(len(tickers)):
    df_data_all[i] = pd.DataFrame(data[tickers[i]]['prices']).filter(['formatted_date', 'adjclose'])
    df_data[i] = pd.DataFrame(data[tickers[i]]['prices'])[-(SIZE+1):].filter(['formatted_date', 'adjclose'])

# returns_V = np.column_stack((df1, df2, df3, df4))
returns_V = np.column_stack(tuple(df_data[i]['adjclose'] for i in range(len(tickers))))
returns_V = np.diff(returns_V, axis=0) / returns_V[1:,:] * 100
print('Shape of returns:', returns_V.shape)
print(returns_V[0:5])

# Plot historical data
fig = go.Figure()
for i in range(len(tickers)):
    fig.add_trace(go.Scatter(x=df_data_all[i]['formatted_date'],
                             y=df_data_all[i]['adjclose'],
                             name=tickers[i],
#                              line_color='deepskyblue',
                             opacity=0.8))

# Use date string to set xaxis range
fig.update_layout(xaxis_range=[start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')],
                   title_text="Components in %s" % data_title)
fig.show()


# In[21]:


# Markowitz bullet representation inspired from
# https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f

import pandas as pd
import numpy as np
from utils import generate_markowitz_bullet, get_mu_sigma

prices = pd.DataFrame(np.column_stack(tuple(df_data[i]['adjclose'] for i in range(len(tickers)))), columns=tickers)
_ = generate_markowitz_bullet(prices)


# In[23]:


from pypfopt.base_optimizer import portfolio_performance
import matplotlib.pyplot as plt
from utils import get_mu_sigma


# VTI, VXUS, BND, BNDX
core_series_allocation = {
    0: (0, 0, 0.686, 0.294),
    1: (0.059, 0.039, 0.617, 0.265),
    2: (0.118, 0.078, 0.549, 0.235),
    3: (0.176, 0.118, 0.480, 0.206),
    4: (0.235, 0.157, 0.412, 0.176),
    5: (0.294, 0.196, 0.343, 0.147),
    6: (0.353, 0.235, 0.274, 0.118),
    7: (0.412, 0.274, 0.206, 0.088),
    8: (0.470, 0.314, 0.137, 0.059),
    9: (0.529, 0.353, 0.069, 0.029),
    10: (0.588, 0.392, 0, 0)
}

# Get mu and S from prices
mu, S = get_mu_sigma(prices)

# Plot different risk-level portfolios from Vanguard Core Series on same graph
for k,allocation in core_series_allocation.items():
    returns, volatility, _ = portfolio_performance(mu, S, allocation)
    plt.plot(volatility, returns, 'o', markersize=5, label='%i%% Equity' % (k*10))

plt.xlabel('volatility')
plt.ylabel('returns')
plt.title('Vanguard Core Series')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
print()


# In[ ]:




