import pandas as pd
import numpy as np

from bs4 import BeautifulSoup as bs
import requests
import yaml
from datetime import datetime, date, time
from dateutil.relativedelta import relativedelta

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

import matplotlib.pyplot as plt

def get_dji():
    """ Dataframe of info of all tickers in Dow Jones Industrial Average. """
    url = 'https://www.dogsofthedow.com/dow-jones-industrial-average-companies.htm'
    request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
    soup = bs(request.text, "lxml")
    stats = soup.find('table',class_='tablepress tablepress-id-42 tablepress-responsive')
    df = pd.read_html(str(stats))[0]
    print(f"DJI: Contains {len(df)} tickers.")
    return df

def get_spy():
    """ Dataframe of info of all tickers in SP&500. """
    url = 'https://www.slickcharts.com/sp500'
    request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
    soup = bs(request.text, "lxml")
    stats = soup.find('table',class_='table table-hover table-borderless table-sm')
    df = pd.read_html(str(stats))[0]
    df['% Chg'] = df['% Chg'].str.strip('()-%')
    df['% Chg'] = pd.to_numeric(df['% Chg'], errors='coerce').fillna(0)
    df['Chg'] = pd.to_numeric(df['Chg'])
    df = df.drop('#', axis=1)
    print(f"SPY: Contains {len(df)} tickers.")
    return df

def get_qqq():
    """ Dataframe of info of all tickers in Nasdaq 100. """
    df = pd.DataFrame()
    urls = ['https://www.dividendmax.com/market-index-constituents/nasdaq-100',
    'https://www.dividendmax.com/market-index-constituents/nasdaq-100?page=2',
    'https://www.dividendmax.com/market-index-constituents/nasdaq-100?page=3']

    for url in urls:
        request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs(request.text, "lxml")
        stats = soup.find('table',class_='mdc-data-table__table')
        temp = pd.read_html(str(stats))[0]
        temp.rename(columns={'Market Cap':'Market Cap $bn'},inplace=True)
        temp['Market Cap $bn'] = temp['Market Cap $bn'].str.strip("£$bn")
        temp['Market Cap $bn'] = temp['Market Cap $bn'].str.replace('m', '*1e-3').astype(str)
        temp['Market Cap $bn'] = temp['Market Cap $bn'].apply(lambda x: eval(x))
        temp['Market Cap $bn'] = pd.to_numeric(temp['Market Cap $bn'])
        df = df.append(temp)
    df = df.sort_values('Market Cap $bn',ascending=False)
    df = df.drop('Unnamed: 2', axis=1)
    df.rename(columns={'Ticker':'Symbol'},inplace=True)
    df = df.reset_index(drop=True)
    print(f"QQQ: Contains {len(df)} tickers.")
    return df

class NestedObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, NestedObject(value))
            else:
                setattr(self, key, value)

def load_args(filepath='args.yaml'):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return NestedObject(data)

def str2datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')

def parse_period_date(period):
    d = period.days
    m = period.months
    y = period.years
    return relativedelta(days=d, months=m, years=y)

def get_dates(args):
    if args.method == 'interval':
        startdate = str2datetime(args.interval.start_date)
        enddate = str2datetime(args.interval.end_date)
    elif args.method == 'back_from_today':
        enddate = date.today()
        startdate = enddate - parse_period_date(args.back_from_today)
    else:
        ValueError("Incorrect value in args.yaml for method.")
    return startdate, enddate

def get_pdr_data(tickers, startdate, enddate, progress=False, clean=True):
    """ Returns Yahoo Finance price data for the given tickers. """
    df = pdr.get_data_yahoo(tickers, start=startdate, end=enddate, progress=progress)['Close']
    # df = df.xs(key='Adj Close', level='Price', axis=1)
    if not clean:
        return df.sort_index()
    else:
        return clean_df(df.sort_index())

def get_benchmark_data(market_ticker, riskfree_ticker, startdate, enddate):
    """ Get data for the benchmark tickers provided in the args.yaml file. """
    market_data = get_pdr_data(market_ticker, startdate, enddate, progress=False, clean=False)
    nonholiday_dates = market_data.index[:-1]
    market_return = compute_return(market_data, was_annual=False, retain_symbols=True)

    riskfree_data = get_pdr_data(riskfree_ticker, startdate, enddate, progress=False, clean=False)
    riskfree_return = compute_return(riskfree_data, was_annual=True, retain_symbols=True)
    return market_return, riskfree_return, sorted(list(nonholiday_dates.to_pydatetime()))

def is_df_okay(df):
    """ Returns True if there are no missing entries (NaN) in the df. """
    return not df.isnull().any().any()

def clean_df(df):
    """ Remove columns that contain NaN in any row. """
    bad_cols = df.columns[df.isnull().any()]
    return df.drop(bad_cols, axis=1)

def nearest_datetime(datetime_list, item):
    """ Get nearest date in a list. """
    diffs = [abs(dt - item) for dt in datetime_list]
    i = np.argmin(diffs)
    return datetime_list[i]

def linspace_datetime(datetime_list, start, end, delta, include_end=False):
    """ Return a list of linearly spaced dates. """
    start = datetime.combine(start, time())
    end = datetime.combine(end, time())

    if delta == relativedelta():
        return [nearest_datetime(datetime_list, start)]

    result = set()
    current = start
    while current < end:
        result.add(nearest_datetime(datetime_list, current))
        current += delta
    if include_end:
        result.add(end)
    return sorted(list(result))

def compute_return(df, was_annual=False, retain_symbols=False):
    """ Compute daily return for every ticker in the provided df. """
    if was_annual:
        # Riskfree data should be converted to daily from annual.
        riskfree_annual_return = (df/100)[:-1]
        riskfree_daily_return = (1 + riskfree_annual_return)**(1/252) - 1
        riskfree_daily_return = riskfree_daily_return.to_numpy()
        # riskfree_annual_log_return = np.log(1 + riskfree_annual_return)
        # riskfree_daily_log_return = (riskfree_annual_log_return/252).to_numpy()
        # riskfree_daily_return = riskfree_daily_log_return
        if not retain_symbols:
            return riskfree_daily_return
        elif isinstance(df, pd.Series):
            return pd.DataFrame(riskfree_daily_return, index=df.index[:-1])
        else:  # df is a pd.DataFrame instance.
            return pd.DataFrame(riskfree_daily_return, columns=df.columns)
    else:
        # Stock market data is already retrieved as daily from Yahoo.
        # market_return = np.diff(np.log(df), axis=0)
        market_return = df.pct_change().dropna().values
        if not retain_symbols:
            return market_return
        elif isinstance(df, pd.Series):
            return pd.DataFrame(market_return, index=df.index[:-1])
        else:  # df is already a pd.DataFrame instance.
            return pd.DataFrame(market_return, columns=df.columns, index=df.index[:-1])
    
def compute_sharpe_ratio(ticker_return_df, riskfree_return_df, retain_symbols=False):
    """ Compute daily Sharpe ratio (what we use). """
    if np.array(ticker_return_df).ndim == 1:
        excess_return = ticker_return_df - riskfree_return_df
    else:
        excess_return = ticker_return_df - riskfree_return_df.reshape(-1, 1)
    sharpe = excess_return.mean(axis=0) / excess_return.std(axis=0, ddof=1)
    if not retain_symbols:
        return np.array(sharpe)
    elif retain_symbols and isinstance(ticker_return_df, np.ndarray):
        return pd.DataFrame(sharpe, columns=ticker_return_df.columns)
    else:  # is a pd.Series instance, and already has symbols retained.
        return sharpe
    
def compute_sortino_ratio(ticker_return_df, riskfree_return_df, retain_symbols=False):
    """ Compute daily Sortino ratio. """
    if np.array(ticker_return_df).ndim == 1:
        excess_return = ticker_return_df - riskfree_return_df
    else:
        excess_return = ticker_return_df - riskfree_return_df.reshape(-1, 1)
    downside_return = excess_return.copy()
    downside_return[downside_return > 0] = 0
    sortino = excess_return.mean(axis=0) / np.sqrt(np.mean(downside_return**2, axis=0))
    if not retain_symbols:
        return np.array(sortino)
    elif retain_symbols and isinstance(ticker_return_df, np.ndarray):
        return pd.DataFrame(sortino, columns=ticker_return_df.columns)
    else:  # is a pd.Series instance, and already has symbols retained.
        return sortino
    
def get_stocks_utility(stocks_tickers, riskfree_ticker, startdate, enddate, utility, progress=True, clean=True):
    """ Download, and compute sharpe/sortino ratio for the provided tickers. """
    stocks_data = get_pdr_data(stocks_tickers, startdate, enddate, progress=progress, clean=clean)
    assert is_df_okay(stocks_data)
    
    riskfree_data = get_pdr_data(riskfree_ticker, startdate, enddate, progress=False, clean=False)
    riskfree_return = compute_return(riskfree_data, was_annual=True)
    stocks_return = compute_return(stocks_data, retain_symbols=True)
    
    if utility == 'sharpe':
        stocks_utility = compute_sharpe_ratio(stocks_return, riskfree_return, retain_symbols=True)
    elif utility == 'sortino':
        stocks_utility = compute_sortino_ratio(stocks_return, riskfree_return, retain_symbols=True)
    else:
        ValueError("Incorrect arg for utility.")
    return stocks_utility.sort_values(ascending=False)
    

def get_stocks_utility_from_data(stocks_data, riskfree_ticker, startdate, enddate, utility):
    """ Faster than above. Compute sharpe/sortino ratios from data downloaded earlier. """
    assert is_df_okay(stocks_data)
    riskfree_data = get_pdr_data(riskfree_ticker, startdate, enddate, progress=False, clean=False)

    startdate = datetime.combine(startdate, time())
    enddate = datetime.combine(enddate, time())

    # mask = (startdate <= stocks_data.index.to_pydatetime()) & (stocks_data.index.to_pydatetime() <= enddate)
    mask = [(item in riskfree_data.index) for item in stocks_data.index]  # returns list of boolean mask
    stocks_data = stocks_data[mask]
    
    stocks_return = compute_return(stocks_data, retain_symbols=True)
    riskfree_return = compute_return(riskfree_data, was_annual=True, retain_symbols=False)
    
    if utility == 'sharpe':
        stocks_utility = compute_sharpe_ratio(stocks_return, riskfree_return, retain_symbols=True)
    if utility == 'sortino':
        stocks_utility = compute_sortino_ratio(stocks_return, riskfree_return, retain_symbols=True)
    else:
        ValueError("Incorrect arg for utility.")
    return stocks_utility.sort_values(ascending=False)

class Portfolio():
    def __init__(self, init_balance):
        self.previous_tickers = None
        self.current_tickers = None
        self.value = init_balance  # float, US Dollars
        self.size = 0  # int, num. of stocks
        # self.fig, self.ax = plt.subplots()

    def verbose(self):
        print(self.current_tickers)

    def plot(self):
        pass

    def rebalance(self, df, min_threshold=0.0):
        self.previous_tickers = self.current_tickers
        tickers = df.index
        vals = df.values

        mask = (vals >= min_threshold)
        tickers = tickers[mask]
        vals = vals[mask]

        self.current_tickers = pd.DataFrame(np.nan, index=tickers, columns=['!Utility', '%Weight', '$Value'])
        self.current_tickers.loc[tickers, '!Utility'] = vals
        self.current_tickers.loc[tickers, '%Weight'] = vals/vals.sum()
        self.current_tickers.loc[tickers, '$Value'] = self.value * self.current_tickers.loc[tickers, '%Weight']
        self.size = len(tickers)

    def update(self, stocks_return, dt):
        dt_str = dt.strftime('%Y-%m-%d')
        todays_returns = stocks_return.loc[dt_str, self.current_tickers.index]
        self.current_tickers['$Value'] *= (1+todays_returns)

        self.value = self.current_tickers['$Value'].sum()
        self.current_tickers['%Weight'] = self.current_tickers['$Value'] / self.value

        # self.ax.plot(x_values, y_values)