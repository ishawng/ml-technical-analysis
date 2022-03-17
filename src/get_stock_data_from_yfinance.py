from datetime import date
import yfinance as yf

from Constants import CLOSE_IND, HIGH_IND, LOW_IND, OPEN_IND, VOL_IND
yf.pdr_override()
import os
import csv
import pandas as pd
from ta.utils import dropna
from ta import add_all_ta_features
from datetime import datetime
import numpy as np

# date format: "yyyy-mm-dd" e.g. 2021-12-31
# folder is either "training_data" or "test_data"
def get_stock_data(start_date : str, folder : str, n = 10, end_date = date.today()):  
    ticker_list = get_top_n_stocks(n)
    output_dir = "stock_data/" + folder
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"folder = '{folder}' Does not exist")
    os.chdir(output_dir)
    for i in range(len(ticker_list)):
        ticker = ticker_list[i]
        data_df = yf.download(ticker, start=start_date, end=end_date)
        if not data_df.empty:
            data_df.to_csv(ticker+'.csv')
            print(f"{ticker} downloaded")
        else:
            print(f"{ticker} download ERROR")
    os.chdir("../..")
    
def get_top_n_stocks(n):
    #read csv file for ranked stock names
    ticker_list = []
    data_dir = 'stock_data'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)
    with open('SP500_Weighted_list.csv') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i > 0:
                stock_name = row[2]
                ticker_list.append(stock_name)
            if i == n:
                break
            else:
                i += 1
        csvfile.close()
    os.chdir("..")
    return ticker_list
    
    
# symb : symbol of stock in capital letters
# folder : either "training_data" or "test_data"
# returns numpy arrays for dates and combined symb_array of format [open, high, low, close, vol]
def pull_data_from_csv(symb : str, folder):
    output_dir = "stock_data/" + folder
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"folder = '{folder}' Does not exist")
    os.chdir(output_dir)
    fname = symb + ".csv"
    df = pd.read_csv(fname, sep=',')
    df = dropna(df)
    df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    dates = df['Date'].values
    dates = np.array([datetime.strptime(date, '%Y-%m-%d').date() for date in dates])
    N = dates.shape[0]
    # print(f"N = {N}")
    symb_data = np.zeros((N,5))
    symb_data[:,CLOSE_IND] = np.array(df['Close'].values)
    symb_data[:,OPEN_IND] = np.array(df['Open'].values)
    symb_data[:,HIGH_IND] = np.array(df['High'].values)
    symb_data[:,LOW_IND] = np.array(df['Low'].values)
    symb_data[:,VOL_IND] = np.array(df['Volume'].values)
    # print(f"close_vals.shape = {close_vals.shape}")
    # print(f"dates.shape = {dates.shape}")
    # print(f"symb_data = {symb_data}")
    os.chdir("../..")
    return dates, symb_data
    
    
if __name__ == "__main__": 
    n = 50
    print(f"DOWNLOADING TEST DATA")
    get_stock_data(n=n, start_date="2021-1-1", end_date="2022-3-1", folder='test_data')
    print(f"DOWNLOADING TRAINING DATA")
    get_stock_data(n=n, start_date="2013-1-1", end_date="2020-12-31", folder='training_data')
    
    # Testing
    # pull_data_from_csv("AAPL", "training_data")
    