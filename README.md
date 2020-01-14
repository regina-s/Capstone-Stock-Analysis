# Capstone-Stock-Analysis
Coursera Capstone Project: My first stock analysis tool with Python

With download_SP500_timeseries_capstone.py the stock symbols and basic company information are downloaded from Wikipedia:
https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
For further use the data are stored in SP500.csv

Using the stock symbols the timeseries of the last 20 years of the stock data are downloaded using ALPHA_VANTAGE, which is free.
For each symbol a separate csv file containting, e.g. 'Close' data are created. 
Thise files are used for further analysis.

AnalysisSP500_capstone.py is reading all the time series of all SP500 stocks and certain values are calculated.
The values are stored 'Analysis.csv' and can be used for quick selection of stocks.

AnalysisSP500_2.py is selection certain stocks depending on criteria, e.g. moving average.
The time series of the selected stocks together with some basic company information are printed. 
