#import jason
import requests
import alpha_vantage
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from data_transfer import *

class DataHist():
    def __init__(self, symbol):        
        self.symbol = symbol

    ' Check if the request and connect is valid '
    def CheckStatus(self):
        # Alpha Vantage API Key CPTY0OT2JK6T70HK
        response = requests.get(
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=CPTY0OT2JK6T70HK")
        # Print the status code of the response.
        print(response.status_code)
        return response.status_code

    ' Set symbol '
    def Setsymbol(self, symbol):
        self.symbol = symbol

    ' Get symbol'
    def Getsymbol(self):
        print(self.symbol)

    ' Get raw data from api, and then reset format of col '
    ' Output: a dataframe n*7 '
    def GetRawData(self):
        ####### raw data from Alpha_vantage #######
        #API_URL = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=CPTY0OT2JK6T70HK"
        ts = TimeSeries(key='CPTY0OT2JK6T70HK', output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol=str(self.symbol), outputsize='full')

        data.sort_index()
        ####### reset all column of dataframe #######
        data = data.drop("7. dividend amount", axis=1)
        data = data.drop("8. split coefficient", axis=1)
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = data.reset_index()
        return df

    ' use data_transfer.py document to compute the necessary data from raw data and output a new dataframe '
    ' Input: a dataframe n*7 '
    ' Output: a dataframe n*20 '
    def DataTransfer(self, df):
        ####### use data_transfer to transfer the raw data to ready to use data as x*20 format #######
        c = Category(df)
        c.createDataset()

        ####### since we compute some value, Nan shows in first 90 rows, so delete first 90 rows #######
        c_data = (c.dataframe.iloc[90:, ])   

        ####### Check if there is any invalid data in dataframe #######
        if (c_data.isnull().any().any() == True):
            print("There are Nan value in dataset!")
        else:
            return c_data

    def RequestFinaldf(self):
        symbol = self.symbol
        connect = self.CheckStatus()
        if connect == 200:
            'connect valid'
            df = self.DataTransfer(self.GetRawData())
            if(df.shape[0] > 2000):
                df = df[-2000:]
            return df


'''
if __name__ == "__main__":
    c = DataHist('^GSPC')
    print(c.RequestFinaldf())
'''
            






