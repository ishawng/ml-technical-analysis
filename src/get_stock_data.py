from datetime import date
import yfinance as yf
yf.pdr_override()
import os
import csv


# date format: "yyyy-mm-dd" e.g. 2021-12-31
# folder is either "training_data" or "test_data"
def get_stock_data(start_date : str, folder : str, n = 10, end_date = date.today()):
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
        
    output_dir = folder
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
    
    
    
if __name__ == "__main__": 
    n = 50
    print(f"DOWNLOADING TEST DATA")
    get_stock_data(n=n, start_date="2021-1-1", end_date="2022-3-1", folder='test_data')
    print(f"DOWNLOADING TRAINING DATA")
    get_stock_data(n=n, start_date="2013-1-1", end_date="2020-12-31", folder='training_data')