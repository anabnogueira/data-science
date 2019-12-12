import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
import statsmodels
import math
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

register_matplotlib_converters()



def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                        'p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)






# loading data
data = pd.read_csv("submetering.csv", delimiter=",")
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d %H:%M:%S")
data.set_index("Date", inplace=True)
#print(data)

# selecting variable + checking missing values
ts = data["Global_active_power"]
#print("Number of missing values: ", ts.isnull().sum().sum())

# choosing time period
ts_filter = ts.loc['2009-01-01':'2009-12-31']
#print(ts_filter)
#print("Number of missing values: ", ts_filter.isnull().sum().sum())

# missing values imputation
ts_filter.interpolate(method='nearest', inplace=True)
#print(ts_filter)
#print("Number of missing values: ", ts_filter.isnull().sum().sum())

# aggregate + choose granularity
ts_daily = ts_filter.resample('D').mean()
plt.plot(ts_daily, linewidth=2, color='#137DD2')
plt.title("Daily granularity")
plt.show()



test_stationarity(ts_daily)


decomposition = seasonal_decompose(ts_daily, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.title("Components")
plt.plot(ts_daily, label='Original', linewidth=2, color='#EB507F')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend', linewidth=2, color='#137DD2')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality', linewidth=2, color='#FF9C34')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals', linewidth=2, color='#4EACC5')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



# additive time series?

plt.figure()
plt.title("moving_avg")
moving_avg = ts_daily.rolling(12).mean()
plt.plot(ts_daily)
plt.plot(moving_avg, color='red')
plt.show()



plt.figure()
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)

test_stationarity(ts_log_decompose)



#print(len(ts_daily))




#Creating train and test set 
train = ts_daily[0:355] 
test = ts_daily[355:]


fithw = ExponentialSmoothing(np.asarray(train), seasonal_periods=7, trend='add', seasonal='add').fit()
series = fithw.forecast(len(test))
y_hat_hw = pd.Series(series, test.index)

plt.figure(figsize=(16,8))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_hw, label='Holt-Winter')
plt.legend(loc='best')
plt.show()

print("\n\n")
rms = math.sqrt(mean_squared_error(test, y_hat_hw))
print("Root Mean Squared Error: ", rms)

mae = mean_absolute_error(test, y_hat_hw)
print("Mean Absolute Error: ", mae)


