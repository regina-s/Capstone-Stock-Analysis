
import pandas as pd
import requests
import alpha_vantage
import time

API_KEY = 'YOUR API KEY'
from alpha_vantage.timeseries import TimeSeries 
from alpha_vantage.techindicators import TechIndicators 
from alpha_vantage.sectorperformance import SectorPerformances
from matplotlib.pyplot import figure 
import matplotlib.pyplot as plt 

from bs4 import BeautifulSoup
website_text = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies').text
soup = BeautifulSoup(website_text,'xml')

table = soup.find('table',{'class':'wikitable sortable'})
table_rows = table.find_all('tr')

data = []
for row in table_rows:
    data.append([t.text.strip() for t in row.find_all('td')])

df = pandas.DataFrame(data, columns=['Symbol', 'Security', 'SEC filings','GIC S Sector', 'GIC S Sub Industry','Headquarters Location', 'Date first added', 'CIK', 'Founded'])

# Create csv file with date
df.to_csv("../../Data/SP500.csv")
df = pd.read_csv('../../Data/SP500.csv', index_col=[0])

# Your key here 
key = API_KEY 
ts = TimeSeries(key) 
ts = TimeSeries(key, output_format='pandas') 
ti = TechIndicators(key) 

for sym in df['Symbol']:
#for sym in df['Symbol']:
#for sym in ['FB','MSFT']:
    print(sym)
    # Get the data, returns a tuple 
    # aapl_data is a pandas dataframe, aapl_meta_data is a dict, 20 years 
    aapl_data, aapl_meta_data = ts.get_daily(symbol=sym,outputsize='full')
    #aapl_data.rename(columns={"date": "Date", '1. open':'Open' , '2. high':'High', '3. low':'Low', '4. close':'Close', '5. volume':'Volume'}, inplace=True)
    Filename='../../Data/'+sym+'.csv'
    aapl_data.to_csv(Filename)
    figure(num=None, figsize=(3, 3), dpi=80, facecolor='w', edgecolor='k') 
    aapl_data['4. close'].plot() 
    plt.tight_layout() 
    plt.title(sym)
    plt.grid() 
    plt.show() 
    time.sleep(20) #waiting loop for avoiding exeeding maximum number of calls per minute
    
