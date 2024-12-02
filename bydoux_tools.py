from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from etienne_tools import lin_mini
import yfinance as yf
from astropy.time import Time

from astropy.table import Table
import numpy as np
import os
import wget

import datetime
import pickle

from currency_converter import CurrencyConverter
from datetime import date

from scipy.interpolate import UnivariateSpline as ius

from tqdm import tqdm
import pandas as pd

def today():
    return datetime.date.today().isoformat().replace('-','')

def xchange_range(mjd):
    c = CurrencyConverter()
    mjd = np.array(mjd)
    dates = Time(mjd, format='mjd').iso

    rates = np.zeros(len(mjd))+np.nan
    for i in range(len(mjd)):
        try:
            yr,mo,day = dates[i].split(' ')[0].split('-')
            rate = c.convert(100, 'CAD', 'USD', date=date(int(yr), int(mo), int(day)))
            rates[i] = rate
        except:
            pass
    valid = np.isfinite(rates)
    rates = ius(mjd[valid], rates[valid], k=1,ext=3)(mjd)

    return rates


def write_pickle(tbl, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tbl, f)

def read_pickle(filename):
    with open(filename,'rb') as f:
        tbl = pickle.load(f)
    return tbl

def get_snp500():
    """
    This function returns the list of the S&P500 tickers
    
    It reads the list from the wikipedia page

    :return: the list of tickers as a numpy array
    """
    link = (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks"
    )
    df = pd.read_html(link, header=0)[0]

    ticker = np.array([ticker for ticker in df['Symbol']])

    return ticker

def read_google_sheet_csv(sheet_id: str, gid: str) -> Table:
    """
    This function reads a Google sheet and returns the content as an
    astropy table

    :param sheet_id:
    :param gid:
    :return: the astropy table
    """
    if os.path.exists('.tmp.csv'):
        os.remove('.tmp.csv')

    GOOGLE_URL = 'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'

    csv_url = GOOGLE_URL.format(sheet_id=sheet_id, gid=gid)

    _ = wget.download(csv_url, out='.tmp.csv', bar=None)
    tbl = Table.read('.tmp.csv', format='ascii.csv')



    # just for sanity, we remove the empty temporary file
    os.remove('.tmp.csv')
    for key in tbl.keys():
        try:
            masked = tbl[key].mask

            tbl[key][masked] = '0'
            # make it a string of 20 characters
            tbl[key] = np.array(tbl[key], dtype='S20')

        except:
            pass

    if 'yf' in tbl.keys():
        tbl['yf'] = tbl['yf'].astype(str)
        for i in range(len(tbl)):
            if tbl['yf'][i] == '0':
                tbl['yf'][i] = tbl['TICKER'][i] + '.TO'

    if 'Date de transaction' in tbl.keys():

        tbl['MJD'] = 0.
        tbl['Time_ago'] = 0.
        now = Time.now().mjd
        for i in range(len(tbl)):
            if '-' not in tbl['Date de transaction'][i]:
                tbl['Date de transaction'][i] = tbl['Date de règlement'][i]
            tbl['MJD'][i] = Time(tbl['Date de transaction'][i]).mjd
            tbl['Time_ago'][i] = now - tbl['MJD'][i]

    columns_float = 'Commission payée',"Montant de l'opération"
    for key in columns_float:
        if key in tbl.keys():
            tmp = tbl[key]
            for i in range(len(tmp)):
                try:
                    tmp[i] = float(tmp[i].replace(',','.'))
                except:
                    pass
            tbl[key] = tbl[key].astype(float)

    return tbl

def get_info(key):

    outname = '/Users/eartigau/pydoux_data/quotes/' + key + '_info_' + today() + '_info.pkl'


    if os.path.exists(outname):
        info = read_pickle(outname)
    else:
        delname = outname.replace(today(),'*')
        os.system('rm '+delname)

        info = yf.Ticker(key).info
        write_pickle(info, outname)

    # make all keys upper case

    #info = {k.upper(): info[k] for k in info.keys()}

    # add Currency if not present and financialCurrency is present
    if 'currency' not in info.keys() and 'financialCurrency' in info.keys():
        info['currency'] = info['financialCurrency']


    return info

def printc(info):
    tt =  Time.now().iso.split(' ')[1]+ ' | '
    print(tt, info)

def get_sp500_history():
    today = datetime.date.today().isoformat().replace('-','')

    all_sp500_name = f'/Users/eartigau/pydoux_data/quotes/snp500_all_{today}_index.pkl'
    if os.path.exists(all_sp500_name):
        printc('Accessing the S&P500 history in pickles')
        printc('\t'+all_sp500_name)
        all_sp500 = read_pickle(all_sp500_name)
    else:
        printc('Downloading the S&P500 history')

        data = yf.download('^GSPC', period='max')
        tbl = Table()

        tbl['date'] =  Time(data.index).iso
        tbl['mjd'] = Time(data.index).mjd

        dd = dict(data)
        for col in dd.keys():
            tbl[col] = np.array(data[col])

        tbl['Close_dividends'] = np.array(data['Close'])

        all_sp500 = write_pickle(tbl, all_sp500_name)

    return all_sp500

def read_quotes(ticker):
    printc(f'Reading quotes for {ticker}')

    data = yf.Ticker(ticker).history(period='max')

    tbl = Table()

    tbl['date'] =  Time(data.index).iso
    tbl['mjd'] = Time(data.index).mjd

    dd = dict(data)
    for col in dd.keys():
        tbl[col] = np.array(data[col])

    tbl['Close_dividends'] = np.array(data['Close'])

    bad = tbl['Close_dividends'] == 0
    tbl = tbl[bad == False]


    # We apply an increase of the value with the dividend
    for i in range(len(tbl)):
        if tbl['Dividends'][i] > 0:
            gain_frac = 1+tbl['Dividends'][i]/tbl['Close_dividends'][i]
            tbl['Close_dividends'][i:] *= gain_frac

    tbl['log_close'] = np.log(tbl['Close_dividends'])

    return tbl


def batch_quotes(sample, full = False, force = False):

    if sample == 'S&P500':
        tickers = get_snp500()
        name_prefix = 'snp500_'
        
    else:
        if sample == 'FNB':
            name_prefix = 'fnb_'
            GID = '0'

        if sample == 'TSX':
            name_prefix = 'tsx_'
            GID = '1977022957'

            ID = '1bx3oBEFAmksB6no7_DV7AP_qM9zQ5iHUepc9wcPjXO8'

            tbl = read_google_sheet_csv(ID, GID)

            tickers = tbl['yf']


    all_sp500_name = f'/Users/eartigau/pydoux_data/quotes/{name_prefix}all_{today()}.pkl'

    if force and os.path.exists(all_sp500_name):
        os.remove(all_sp500_name)

    if os.path.exists(all_sp500_name):
        all_sp500 = read_pickle(all_sp500_name)
    else:

        doplot = False



        all_sp500 = {}
        for ticker in tickers:

            printc(f' input of {ticker} in batch_quotes')
            outname = '/Users/eartigau/pydoux_data/quotes/' + ticker + '_' + today() + '.csv'

            try:
                if not os.path.exists(outname) :
                    tbl1 = read_quotes(ticker)
                    tbl1['mjd'] = Time(tbl1['date']).mjd

                    delname = outname.replace(today(), '*')
                    os.system('rm ' + delname)

                    # convert to dictionary
                    tbl1.write(outname, overwrite=True)
                else:
                    tbl1 = Table.read(outname)
                    if len(tbl1) == 0:
                        continue

                if not full:
                    rem_keys = ['Open','High','Low','Volume','Dividends','Stock Splits']
                    for k in rem_keys:
                        del tbl1[k]

                # make a dictionary
                tbl1 = {k: tbl1[k].data for k in tbl1.keys()}

                all_sp500[ticker] = tbl1
            except:
                printc(f'failed for {ticker}')

        delname = all_sp500_name.replace(today(),'*')
        os.system('rm '+delname)
        write_pickle(all_sp500, all_sp500_name)

    return all_sp500
