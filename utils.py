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
    df['% Chg'] = pd.to_numeric(df['% Chg'])
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
    return datetime.strptime(date_str, '%m-%d-%Y')

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

    # dt1 = startdate - parse_period_date(args.sharpe.period)
    # dt2 = startdate - parse_period_date(args.rebalance.period)
    # startdate = min(dt1, dt2)
    return startdate, enddate


def get_pdr_data(tickers, startdate, enddate, progress=False, clean=True):
    df = pdr.get_data_yahoo(tickers, start=startdate, end=enddate, progress=progress)['Adj Close']
    # df = df.xs(key='Adj Close', level='Price', axis=1)
    if not clean:
        return df.sort_index()
    else:
        return clean_df(df.sort_index())

def get_benchmark_data(market_ticker, riskfree_ticker, startdate, enddate):
    market_data = get_pdr_data(market_ticker, startdate, enddate, progress=False, clean=False)
    nonholiday_dates = market_data.index[:-1]
    market_return = compute_log_return(market_data, was_annual=False, retain_symbols=True)

    riskfree_data = get_pdr_data(riskfree_ticker, startdate, enddate, progress=False, clean=False)
    riskfree_return = compute_log_return(riskfree_data, was_annual=True, retain_symbols=True)
    return market_return, riskfree_return, sorted(list(nonholiday_dates.to_pydatetime()))

def is_df_okay(df):
    """ Returns True if there are no missing entries (NaN) in the df. """
    return not df.isnull().any().any()

def clean_df(df):
    """ Remove columns that contain NaN in any row. """
    bad_cols = df.columns[df.isnull().any()]
    return df.drop(bad_cols, axis=1)

def nearest_datetime(datetime_list, item):
    diffs = [abs(dt - item) for dt in datetime_list]
    i = np.argmin(diffs)
    return datetime_list[i]

def linspace_datetime(datetime_list, start, end, delta, include_end=False):
    # start = datetime.datetime(2019, 12, 1, 0, 0)
    # end = datetime.datetime(2020, 1, 1, 0, 0)
    # delta = relativedelta(months=6)
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

def compute_log_return(df, was_annual=False, retain_symbols=False):
    if was_annual:
        # Riskfree data should be converted to daily from annual.
        riskfree_annual_return = (df/100)[:-1]
        riskfree_annual_log_return = np.log(1+riskfree_annual_return)
        riskfree_daily_log_return = (riskfree_annual_log_return/252).to_numpy()
        if not retain_symbols:
            return riskfree_daily_log_return
        elif isinstance(df, pd.Series):
            return pd.DataFrame(riskfree_daily_log_return, index=df.index[:-1])
        else:  # df is a pd.DataFrame instance.
            return pd.DataFrame(riskfree_daily_log_return, columns=df.columns)
    else:
        # Stock market data is already retrieved as daily.
        market_return = np.diff(np.log(df), axis=0)
        if not retain_symbols:
            return market_return
        elif isinstance(df, pd.Series):
            return pd.DataFrame(market_return, index=df.index[:-1])
        else:  # df is a pd.DataFrame instance.
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
    
def get_stocks_sharpe(stocks_tickers, riskfree_ticker, startdate, enddate, progress=True, clean=True):
    stocks_data = get_pdr_data(stocks_tickers, startdate, enddate, progress=progress, clean=clean)
    assert is_df_okay(stocks_data)
    
    riskfree_data = get_pdr_data(riskfree_ticker, startdate, enddate, progress=False, clean=False)
    riskfree_return = compute_log_return(riskfree_data, was_annual=True)
    
    stocks_return = compute_log_return(stocks_data, retain_symbols=True)
    stocks_sharpe = compute_sharpe_ratio(stocks_return, riskfree_return, retain_symbols=True)
    return stocks_sharpe.sort_values(ascending=False)

def get_stocks_sharpe_from_data(stocks_data, riskfree_ticker, startdate, enddate):
    assert is_df_okay(stocks_data)
    riskfree_data = get_pdr_data(riskfree_ticker, startdate, enddate, progress=False, clean=False)

    startdate = datetime.combine(startdate, time())
    enddate = datetime.combine(enddate, time())

    # mask = (startdate <= stocks_data.index.to_pydatetime()) & (stocks_data.index.to_pydatetime() <= enddate)
    mask = [(item in riskfree_data.index) for item in stocks_data.index]  # returns list of boolean mask
    stocks_data = stocks_data[mask]
    
    stocks_return = compute_log_return(stocks_data, retain_symbols=True)
    riskfree_return = compute_log_return(riskfree_data, was_annual=True, retain_symbols=False)
    
    stocks_sharpe = compute_sharpe_ratio(stocks_return, riskfree_return, retain_symbols=True)
    return stocks_sharpe.sort_values(ascending=False)

class Portfolio():
    def __init__(self):
        self.previous_tickers = {}
        self.current_tickers = {}
        self.value = 0.00  # US Dollars
        self.size = 0

    def rebalance(self, df, min_threshold=0.0):
        self.previous_tickers = self.current_tickers
        tickers = df.index
        vals = df.values

        mask = (vals >= min_threshold)
        tickers = tickers[mask]
        vals = vals[mask]

        self.current_tickers = {t:v for (t,v) in zip(tickers, vals)}
        self.size = len(tickers)

    def display(self):
        print(f'Ticker\tUtility\tWeight')
        total = sum(self.current_tickers.values())
        for key, value in self.current_tickers.items():
            print(f'{key}:\t{round(value, ndigits=4)}\t{round(value/total, ndigits=4)}')

    def update(self, stocks_return, dt):
        dt_str = dt.strftime('%Y-%m-%d')
        total = sum(self.current_tickers.values())
        todays_returns = [stocks_return.loc[dt_str][k] * v/total for (k,v) in self.current_tickers.items()]
        self.value += sum(todays_returns)
