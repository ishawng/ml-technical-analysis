import numpy as np
import pandas as pd
from ta import trend
from get_stock_data_from_yfinance import pull_data_from_csv
from simulate import simulate, plot_account
from Constants import CLOSE_IND
from matplotlib import pyplot as plt

# our novel method: maximizes profit per day
def daily_peak_valley_labels(symb_data : np.array):
    close_vals = symb_data[:,CLOSE_IND]
    N = len(close_vals)
    labels = np.array(close_vals[1:] > close_vals[:-1], dtype=np.int16)
    labels[labels == 0] = -1 # -1 = sell (hold in cash) condition, 1 = buy (hold in stock) condition
    labels_mask = np.array(labels == 1, dtype=np.bool8)
    daily_gain_vals = close_vals[1:] / close_vals[:-1]
    profit_vals = daily_gain_vals[labels_mask]
    profit_ratio = np.prod(profit_vals)
    # print(f"DailyPV profit_ratio = {profit_ratio}")
    return labels, profit_ratio

# FROM PAPER: https://www.sciencedirect.com/science/article/pii/S2405918815300179#tbl1
# Section 5.2
def MA15_trend_analysis_labels(symb_data : np.array):
    close_vals = symb_data[:,CLOSE_IND]
    N = len(close_vals)
    ma15 = (trend.sma_indicator(pd.Series(close_vals), window=15)).to_numpy()
    # condition1 (c1): past 5 days all gain => c1 up; past 5 days all loss => c1 down;
    ma15_gl = np.zeros((N,))
    ma15_gl[1:] = ma15[1:] - ma15[:-1] # gets daily ma15 gain / loss
    # condition2 (c2): price > ma15 => c2 up; price < ma15 => c2 down;
    ma15_lead_lag = close_vals - ma15
    labels = np.zeros((N-1,))
    for i in range(5, N):
        if ma15_lead_lag[i] > 0:
            # c2 is uptrend, check c1
            if np.all(ma15_gl[i-5:i] > 0):
                labels[i-1] = 1
        elif ma15_lead_lag[i] < 0:
            # c2 is downtrend, check c1
            if np.all(ma15_gl[i-5:i] < 0):
                labels[i-1] = -1
    labels_mask = np.array(labels == 1, dtype=np.bool8)
    daily_gain_vals = close_vals[1:] / close_vals[:-1]
    profit_vals = daily_gain_vals[labels_mask]
    profit_ratio = np.prod(profit_vals)
    # print(f"MA15 profit_ratio = {profit_ratio}")
    return labels, profit_ratio

# TESTING  
if __name__ == "__main__": 
    # testing with AAPL data
    dates, symb_data = pull_data_from_csv("AAPL", "training_data")
    # plt.plot(dates, symb_data[:,CLOSE_IND] / symb_data[0,CLOSE_IND])
    # plt.show()
    decisions1, _ = daily_peak_valley_labels(symb_data)
    decisions2, _ = MA15_trend_analysis_labels(symb_data)
    decisions0 = np.ones((len(decisions1),)) # buy-hold
    plot_account(dates, simulate(symb_data, decisions0), "buy-hold")
    plot_account(dates, simulate(symb_data, decisions1), "our-novel-label")
    plot_account(dates, simulate(symb_data, decisions2), "MA15-label")
    
    