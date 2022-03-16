'''
Install TA Library before using (need Python >= v3.6):
https://technical-analysis-library-in-python.readthedocs.io/en/latest/
'''
from ta import trend
from ta import momentum
from ta import volatility
from ta import volume
import numpy as np
import pandas as pd
from get_stock_data_from_yfinance import pull_data_from_csv
import matplotlib.pyplot as plt

from Constants import CLOSE_IND, HIGH_IND, LOW_IND, VOL_IND

def get_features_data(symb_data : np.array):
    close_vals = symb_data[:,CLOSE_IND]
    high_vals = symb_data[:,HIGH_IND]
    low_vals = symb_data[:,LOW_IND]
    vol_vals = symb_data[:,VOL_IND]
    D = 40
    N = close_vals.shape[0]
    feat = np.zeros((N, D))
    # ======== STEP1: Get SMAs, EMAs, MACDs etc. =====================
    std_ema = get_ema(close_vals, 5) # 5 is the standardizing ema metric
    sma34 = get_sma(close_vals, 34)
    sma50 = get_sma(close_vals, 50)
    sma100 = get_sma(close_vals, 100)
    sma200 = get_sma(close_vals, 200)
    ema21 = get_ema(close_vals, 21)
    ema55 = get_ema(close_vals, 55)
    ema144 = get_ema(close_vals, 144)
    macd1 = get_macd(close_vals, 26, 12, 9)
    macd2 = get_macd(close_vals, 60, 2, 2)
    rsi2 = get_rsi(close_vals, 2)
    rsi5 = get_rsi(close_vals, 5)
    rsi14 = get_rsi(close_vals, 14)
    rsi28 = get_rsi(close_vals, 28)
    atr14 = get_atr(high_vals, low_vals, close_vals, 14)
    atr21 = get_atr(high_vals, low_vals, close_vals, 21)
    atr50 = get_atr(high_vals, low_vals, close_vals, 50)
    vol30 = get_sma(vol_vals, 30)
    vwap5 = get_vwap(high_vals, low_vals, close_vals, vol_vals, 5)
    vwap14 = get_vwap(high_vals, low_vals, close_vals, vol_vals, 14)
    vwap30 = get_vwap(high_vals, low_vals, close_vals, vol_vals, 30)
    # ======== STEP2: Develop features ================================
    # MOVING AVERAGE FEATURES >>>
    # feature1: price-std_ema ratio
    feat[:,0] = (close_vals - std_ema) / std_ema
    # feature2: price-sma34 ratio
    feat[:,1] = (close_vals - sma34) / std_ema
    # feature3: price-sma50 ratio
    feat[:,2] = (close_vals - sma50) / std_ema
    # feature4: price-sma100 ratio
    feat[:,3] = (close_vals - sma100) / std_ema
    # feature5: price-sma200 ratio
    feat[:,4] = (close_vals - sma200) / std_ema
    # feature6: price-ema21 ratio
    feat[:,5] = (close_vals - ema21) / std_ema
    # feature7: price-ema55 ratio
    feat[:,6] = (close_vals - ema55) / std_ema
    # feature8: price-ema144 ratio
    feat[:,7] = (close_vals - ema144) / std_ema
    # feature9: sma34-sma50 ratio
    feat[:,8] = (sma34 - sma50) / std_ema
    # feature10: sma34-sma100 ratio
    feat[:,9] = (sma34 - sma100) / std_ema
    # feature11: sma34-sma200 ratio
    feat[:,10] = (sma34 - sma200) / std_ema
    # feature12: sma50-sma100 ratio
    feat[:,11] = (sma50 - sma100) / std_ema
    # feature13: sma50-sma200 ratio
    feat[:,12] = (sma50 - sma200) / std_ema
    # feature14: sma100-sma100 ratio
    feat[:,13] = (sma100 - sma200) / std_ema
    # feature15: ema21-ema55 ratio
    feat[:,14] = (ema21 - ema55) / std_ema
    # feature16: ema21-ema144 ratio
    feat[:,15] = (ema21 - ema144) / std_ema
    # feature17: ema55-sma144 ratio
    feat[:,16] = (ema55 - ema144) / std_ema
    # feature18: sma34-sma144 ratio
    feat[:,17] = (sma34 - ema144) / std_ema
    # MACD FEATURES >>>
    # feature19: macd1_diff ratio
    feat[:,18] = macd1.macd_diff() / std_ema
    # feature20: macd1 ratio
    feat[:,19] = macd1.macd() / std_ema
    # feature21: macd2_diff ratio
    feat[:,20] = macd2.macd_diff() / std_ema
    # feature22: macd2 ratio
    feat[:,21] = macd2.macd() / std_ema
    # RSI FEATURES >>>
    # feature23: rsi2
    feat[:,22] = rsi2
    # feature24: rsi5
    feat[:,23] = rsi5
    # feature25: rsi14
    feat[:,24] = rsi14
    # feature26: rsi28
    feat[:,25] = rsi28
    # feature27: rsi5 - rsi14
    feat[:,26] = rsi5 - rsi14
    # ATR FEATURES >>>
    # feature28: atr14 ratio
    feat[:,27] = atr14 / std_ema
    # feature29: atr21 ratio
    feat[:,28] = atr21 / std_ema
    # feature30: atr50 ratio
    feat[:,29] = atr50 / std_ema
    # feature31: atr14-atr50 ratio
    feat[:,30] = (atr14 - atr50) / std_ema
    # feature32: atr21-atr50 ratio
    feat[:,31] = (atr21 - atr50) / std_ema
    # VOLUME FEATURES >>>
    # feature33: vol-vol30 ratio
    feat[:,32] = (vol_vals - vol30) / vol30
    # VWAP FEATURES >>>
    # feature34: price-vwap5 ratio
    feat[:,33] = (close_vals-vwap5) / std_ema
    # feature35: price-vwap14 ratio
    feat[:,34] = (close_vals-vwap14) / std_ema
    # feature36: price-vwap30 ratio
    feat[:,35] = (close_vals-vwap30) / std_ema
    # feature37: vwap14-sma34 ratio
    feat[:,36] = (vwap14-sma34) / std_ema
    # feature38: vwap14-sma200 ratio
    feat[:,37] = (vwap14-sma200) / std_ema
    # feature39: vwap14-ema144 ratio
    feat[:,38] = (vwap14-ema144) / std_ema
    # feature40: vwap5-vwap14 ratio
    feat[:,39] = (vwap5-vwap14) / std_ema
    
    
    # plot_feature(feat[:,33], "vwap5-vwap14")

def get_sma(close_vals : np.array, window : int):
    return (trend.sma_indicator(pd.Series(close_vals), window=window)).to_numpy()

def get_ema(close_vals : np.array, window : int):
    return (trend.ema_indicator(pd.Series(close_vals), window=window)).to_numpy()
 
# NOTE: macd returns an object, macd has .macd(), .macd_diff(), .macd_signal()
def get_macd(close_vals : np.array, window_slow : int, window_fast : int, window_sign : int):
    return (trend.MACD(pd.Series(close_vals), window_slow=window_slow, window_fast=window_fast,
                       window_sign=window_sign))
    
def get_rsi(close_vals : np.array, window : int):
    return (momentum.rsi(pd.Series(close_vals), window=window)).to_numpy()

def get_atr(high_vals : np.array, low_vals : np.array, close_vals : np.array, window : int):
    return (volatility.average_true_range(pd.Series(high_vals), pd.Series(low_vals),
                                        pd.Series(close_vals), window=window)).to_numpy()
    
def get_vwap(high_vals : np.array, low_vals : np.array, close_vals : np.array,
             vol_vals : np.array, window : int):
    return (volume.volume_weighted_average_price(pd.Series(high_vals), pd.Series(low_vals),
                                        pd.Series(close_vals), pd.Series(vol_vals), window=window)).to_numpy()
    
    
def plot_feature(feat_col, label):
    plt.plot(feat_col)
    plt.ylabel(label)
    plt.show()
    
  
# TESTING  
if __name__ == "__main__": 
    # testing with AAPL data
    dates, symb_data = pull_data_from_csv("AAPL", "training_data")
    get_features_data(symb_data)
    # dates, symb_data = pull_data_from_csv("FB", "training_data")
    # get_features_data(symb_data)
    
