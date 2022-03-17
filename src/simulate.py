import numpy as np
from Constants import CLOSE_IND
from matplotlib import pyplot as plt

# symb_data (shape = N,5)
# decisions (len=N-1): array of +1=buy,+0=do nothing,-1=sell
def simulate(symb_data : np.array, decisions : np.array):
    close_vals = symb_data[:,CLOSE_IND]
    N = len(close_vals)
    acc = np.ones((N,)) # account (starting at unit 1)
    daily_gl = close_vals[1:] / close_vals[:-1]
    for i in range(1,N):
        if decisions[i-1] == 1:
            # buy/hold stock
            acc[i] = acc[i-1] * daily_gl[i-1]
        else:
            # sell/hold cash
            acc[i] = acc[i-1]
    return acc

def plot_account(dates : np.array, acc : np.array, title : str):
    plt.plot(dates, acc)
    plt.xlabel("Time")
    plt.ylabel("Unit Account")
    plt.title (title)
    plt.show()
    