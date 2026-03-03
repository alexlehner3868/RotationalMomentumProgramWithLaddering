#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import yfinance as yf
import numpy as np
import time


start_date=datetime.datetime(2022, 9, 1)
end_date=datetime.datetime(2025, 3, 8)
#end_date= datetime.datetime.now()

#Negative correlation regime:
#When TLT is negatively correlated with SPY on a rolling 90 day window:
#use SPY.TLT.csv to calculate the sharpe surface
#use TLT as the safe asset
#no need to include UUP

#Positive correlation regime:
#When TLT is positively correlated with SPY on a rolling 90 day window:
#use SPY.SHY.csv to calculate the sharpe surface
#use SHY as the safe asset
#include UUP in your investment list, since the FED is keeping interest rates high

#Dow Jones Index + TLT or SHY:
#StockList = ['UUP', AXP', 'AAPL', 'BA','CAT','CVX','CSCO','KO','DIS','XOM','GS', 'HD', 'IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE', 'PG', 'TRV','UTX','UNH', 'VZ','V','WMT','WBA', 'SHY']
#Dow Jones Index:
#StockList = ['UUP', AXP', 'AAPL', 'BA','CAT','CVX','CSCO','KO','DIS','XOM','GS', 'HD', 'IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE', 'PG', 'TRV','UTX','UNH', 'VZ','V','WMT','WBA']

#20+ industrial sectors+ TLT or SHY:
#stock_list = ["UUP","FDN","IBB","IEZ","IGV","IHE","IHF","IHI","ITA","ITB","IYJ","IYT","IYW","IYZ","KBE","KCE","KIE","PBJ","PBS","SMH","VNQ","SHY"]
#21 industrial sectors:
#stock_list = ["UUP","FDN","IBB","IEZ","IGV","IHE","IHF","IHI","ITA","ITB","IYJ","IYT","IYW","IYZ","KBE","KCE","KIE","PBJ","PBS","SMH","VNQ"]

#10 sectors + TLT or SHY:
#stock_list = ["UUP","XLB","XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XRT", "SHY"] 
#10 sectors:
#stock_list = ["UUP","XLB","XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XRT"] 

#Bond etfs:
stock_list = ["BIL","TIP","IEI"] 
#BIL = 1 to 3 Months, IEI = 3 to 7 Years, IEF = 7 to 10 Years, TLH = 10 to 20 Years, TLT = 20+ Years, SHY = CASH, TIP = Inflation protected

#our ETF list + TLT or SHY:
#stock_list = ["UUP","FDN","IBB","IEZ","IGV","IHE","IHF","IHI","ITA","ITB","IYJ","IYT","IYW","IYZ","KBE","KCE","KIE","PBJ","PBS","SMH","VNQ","SHY"]
#our ETF list
#stock_list = ["UUP","FDN","IBB","IEZ","IGV","IHE","IHF","IHI","ITA","ITB","IYJ","IYT","IYW","IYZ","KBE","KCE","KIE","PBJ","PBS","SMH","VNQ"]


#stock_list = ["SPY", "TLT"]  #********************************************************this is good for sharpe surface (negative correlation reginme)
#stock_list = ["SPY", "SHY"]  #********************************************************this is good for sharpe surface


stock_str = ""
for i in range(len(stock_list)):
    stock_str  = stock_str + stock_list[i] + "."

stock_str = ""
for stock in stock_list:
    stock_str  = stock_str + stock + "."

main_df = pd.DataFrame()

for stock in stock_list:
    try:
         print(f"Fetching {stock}...")
         time.sleep(5)
         stock_data = yf.download(stock, start=start_date, end=end_date)
         df = pd.DataFrame(stock_data.values, index=stock_data.index, columns=["Close","High","Low","Open","Volume"])
         df.drop(['High', 'Low' , 'Open', 'Volume'], axis=1, inplace=True)
         df.rename(columns={'Close': stock}, inplace=True)
         if main_df.empty:
             main_df = df
         else:
            main_df = main_df.join(df)
    except Exception as e:
         print(f"Error fetching {stock}: {e}")  

#main_df.to_csv(r"C:\Users\Rosario\Documents\PortfolioOptimizationOnline\Markowitz_2\HarrysProblem.SPY.TLT.YFINANCE\\"+stock_str+"AP.csv")
main_df.to_csv(stock_str+"AP.csv")
#print(f"Saved {stock_str}AP.csv")

main_df = pd.DataFrame()

for stock in stock_list:
     try:
         print(f"Fetching {stock}...")
         time.sleep(5)
         stock_data = yf.download(stock, start=start_date, end=end_date)
         df = pd.DataFrame(stock_data.values, index=stock_data.index, columns=["Close","High","Low","Open","Volume"])
         df.drop(['High', 'Low' , 'Open', 'Volume'], axis=1, inplace=True)
         df.rename(columns={'Close': stock}, inplace=True)
         if main_df.empty:
             main_df = df
         else:
            main_df = main_df.join(df)
     except Exception as e:
         print(f"Error fetching {stock}: {e}")  

#main_df.to_csv(r"C:\Users\Rosario\Documents\PortfolioOptimizationOnline\Markowitz_2\HarrysProblem.SPY.TLT.YFINANCE\\"+stock_str+"csv")
main_df.to_csv(stock_str+"csv")
print(f"Saved {stock_str}csv")


# In[ ]:




