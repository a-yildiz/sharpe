import pandas as pd
import numpy as np

from bs4 import BeautifulSoup as bs
import requests
import yaml
from datetime import datetime, date
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

def get_dates(args):
    if args.method == 'interval':
        startdate = str2datetime(args.interval.start_date)
        enddate = str2datetime(args.interval.end_date)
    elif args.method == 'back_from_today':
        d = args.back_from_today.days
        m = args.back_from_today.months
        y = args.back_from_today.years
        enddate = date.today()
        startdate = enddate - relativedelta(days=d, months=m, years=y)
    else:
        ValueError("Incorrect value in args.yaml for method.")
    return startdate, enddate

def get_pdr_data(tickers, startdate, enddate):
    market_data = pdr.get_data_yahoo(tickers, start=startdate, end=enddate, progress=False)['Adj Close']
    return market_data.sort_index()

def get_benchmark_data(args):
    startdate, enddate = get_dates(args)
    market_data = pdr.get_data_yahoo(args.benchmark.market, start=startdate, end=enddate, progress=False)['Adj Close']
    riskfree_data = pdr.get_data_yahoo(args.benchmark.riskfree, start=startdate, end=enddate, progress=False)['Adj Close']
    common_entries = set(market_data.keys()).intersection(set(riskfree_data.keys()))
    return market_data[common_entries].sort_index(), riskfree_data[common_entries].sort_index()

def is_df_okay(df):
    """ Returns True if there are no missing entries (NaN) in the df. """
    return not df.isnull().any().any()

def clean_df(df):
    """ Remove columns that contain NaN in any row. """
    # df = df.dropna(how='all', axis=1)
    bad_cols = df.columns[df.isnull().any()]
    return df.drop(bad_cols, axis=1)

def compute_return(df, was_annual=False, retain_symbols=False):
    if was_annual:
        # Riskfree data should be converted to daily from annual.
        riskfree_annual_return = (df/100)[1:]
        riskfree_annual_log_return = np.log(1+riskfree_annual_return)
        riskfree_daily_log_return = riskfree_annual_log_return/252
        return riskfree_daily_log_return.to_numpy()
    else:
        # Stock market data is already retrieved as daily.
        market_return = np.diff(np.log(df), axis=0)
        if not retain_symbols:
            return market_return
        else:
            return pd.DataFrame(market_return, columns=df.columns)
    
def sharpe_ratio(ticker_return_df, riskfree_return_df, retain_symbols=False):
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