import pandas as pd
import numpy as np
import requests
from matplotlib.pyplot import figure 
import matplotlib.pyplot as plt 

df = pd.read_csv('../../Data/SP500.csv', index_col=[0])
df = df[df['Symbol'] != 'BF.B']
df = df.dropna(how='all')

print(df['GIC S Sector'].value_counts())
ii=0
data = []
data = pd.DataFrame(data, columns=['Symbol', 'drop 40%  220 d', 'drop 20% 220 d', 'Single d risk','5% quant.', '95% quant.','25% quant.', '75% quant.','MA20','MA40','MA200','Pricediff40','Close'])

Symbol='WMB'

for Symbol in df['Symbol']:
#for Symbol in df_energy['Symbol']:
    print(ii,Symbol)
    path='../../Data/'+Symbol+'.csv'
    df1=pd.read_csv(path)
    df1.rename(columns={"date": "Date", '1. open':'Open' , '2. high':'High', '3. low':'Low', '4. close':'Close', '5. volume':'Volume'}, inplace=True)
    #Add Columns
#    df1['Price1']=df1['Close'].shift(-1) # Shits up by one column
    df1['Price1']=df1['Close'].shift(-40) # Shits up by one column
    df1['PriceDiff']=df1['Price1']-df1['Close']
    pricediff40=df1['PriceDiff'][0]
    df1.head
    df1['Return']=df1['PriceDiff']/df1['Close']
    #Create Direction: 1 or -1
    df1['Direction']=[1 if df1.loc[ei,'PriceDiff'] > 0 else -1
    for ei in df1.index]
    #Moving Average of the last 3 days
    df1['Average3']=(df1['Close']+df1['Close'].shift(1)+df1['Close'].shift(2))/3
    #Rolling:
    df1['MA20']=df1['Close'].rolling(20).mean().shift(-20)
    df1['MA40']=df1['Close'].rolling(40).mean().shift(-40)
    df1['MA200']=df1['Close'].rolling(200).mean().shift(-200)
    #Plot
    #df1['Close'].plot()
    #df1['MA40'].plot() #Fast signal
    #df1['MA200'].plot() #Slow signal
    ma20= df1['MA20'][0] 
    ma40= df1['MA40'][0]  
    ma200= df1['MA200'][0]
    close=df1['Close'][0]
    df1=df1[:220] #approximatly 1 year
    #df1=df1[:1000] #approximatly 5 years
    df1['LogReturn'] = np.log(df1['Close']).shift(-1) - np.log(df1['Close'])
    # Plot a histogram to show the distribution of log return of the stock. 
    # You can see it is very close to a normal distribution
    from scipy.stats import norm
    mu = df1['LogReturn'].mean()
    sigma = df1['LogReturn'].std(ddof=1)
    density = pd.DataFrame()
    density['x'] = np.arange(df1['LogReturn'].min()-0.01, df1['LogReturn'].max()+0.01, 0.001)
    density['pdf'] = norm.pdf(density['x'], mu, sigma)
    #df1['LogReturn'].hist(bins=50, figsize=(15, 8))
    #df1['LogReturn'].hist(bins=50, figsize=(3, 3))
    #plt.title(Symbol)
    #plt.plot(density['x'], density['pdf'], color='red')
    #plt.show()
    
    #figure(num=None, figsize=(3, 3), dpi=80, facecolor='w', edgecolor='k') 
    #df1['Close'].plot() 
    #plt.tight_layout() 
    #plt.title(Symbol)
    #plt.grid() 
    #plt.show()
    # probability that the stock price of microsoft will drop over 5% in a day
    prob_return1 = norm.cdf(-0.05, mu, sigma)
    # Now is your turn, calculate the probability that the stock price of microsoft will drop over 10% in a day
    prob_return1 = norm.cdf(-0.10, mu, sigma)
    # drop over 40% in 220 days
    mu220 = 220*mu
    sigma220 = (220**0.5) * sigma
    drop40 = norm.cdf(-0.4, mu220, sigma220)
    # drop over 20% in 220 days
    mu220 = 220*mu
    sigma220 = (220**0.5) * sigma
    drop20 = norm.cdf(-0.2, mu220, sigma220)
    # Value at risk(VaR)
    VaR = norm.ppf(0.05, mu, sigma)
    # Quatile 
    # 5% quantile
    q05 = norm.ppf(0.05, mu, sigma)
    # 95% quantile
    q95 = norm.ppf(0.95, mu, sigma)
    # This is your turn to calcuate the 25% and 75% Quantile of the return
    # 25% quantile
    q25 = norm.ppf(0.25, mu, sigma)
    # 75% quantile
    q75 = norm.ppf(0.75, mu, sigma) 
    data.loc[ii]=[Symbol,drop40, drop20,VaR,q05,q95,q25,q75,ma20,ma40,ma200,pricediff40,close]
    print(pricediff40)
    ii=ii+1

    
    
#print(data['Symbol'],data['dropping over 40% in 220 days'])
Filename='../../Data/Analysis.csv'
data.to_csv(Filename)


