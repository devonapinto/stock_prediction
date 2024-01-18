import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas_ta as ta
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import sys
symbol = "MSFT"
try:
    data = yf.download(symbol, start="2023-11-10", end="2023-11-16")
    if data.empty:
        raise ValueError("No data available for the specified date range.")
except Exception as e:
    print(f"Error occurred: {e}")
    sys.exit(1)
historical_prices = data['Close']
historical_mean = np.mean(historical_prices)
historical_std_dev = np.std(historical_prices)
company_data = yf.Ticker(symbol)
current_price = company_data.history(period="1d")["Close"].iloc[0] 
z_score = (current_price - historical_mean) / historical_std_dev
print(f"Current price of {symbol}: ${current_price}")
print("Z-Score:", z_score)
print(data.tail(10))
data['Next_Close'] = data['Close'].shift(-1)  
print(data['Next_Close'])
data.dropna(inplace=True)
X = data[['Open', 'High', 'Low', 'Close', 'Volume']] 
y = data['Next_Close'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse}")
latest_data = X.iloc[-1].values.reshape(1, -1)  
predicted_price = model.predict(latest_data)[0]
print(f"Predicted next day's closing price: {predicted_price}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
r_squared = r2_score(y_test, predictions)
print(f"R-squared: {r_squared}")
n = len(X_test)  
p = X_test.shape[1]  
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
print(f"Adjusted R-squared: {adjusted_r_squared}")
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.coef_})
print(feature_importance)
symbol = 'MSFT'
end_date = datetime.today()
start_date = end_date - timedelta(days=120) 
stock_data = yf.download(symbol, start=start_date, end=end_date)
stock_data.ta.macd(append=True)
stock_data.ta.rsi(append=True)
stock_data.ta.bbands(append=True)
stock_data.ta.obv(append=True)
stock_data.ta.sma(length=20, append=True)
stock_data.ta.ema(length=50, append=True)
stock_data.ta.stoch(append=True)
stock_data.ta.adx(append=True)
stock_data.ta.willr(append=True)
stock_data.ta.cmf(append=True)
stock_data.ta.psar(append=True)
stock_data['OBV_in_million'] =  stock_data['OBV']/1e7
stock_data['MACD_histogram_12_26_9'] =  stock_data['MACDh_12_26_9'] 
last_day_summary = stock_data.iloc[-1][['Adj Close',
    'MACD_12_26_9','MACD_histogram_12_26_9', 'RSI_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0','SMA_20', 'EMA_50','OBV_in_million', 'STOCHk_14_3_3', 
    'STOCHd_14_3_3', 'ADX_14',  'WILLR_14', 'CMF_20', 
    'PSARl_0.02_0.2', 'PSARs_0.02_0.2'
]]
sys_prompt = """
Assume the role as a leading Technical Analysis (TA) expert in the stock market, \
a modern counterpart to Charles Dow, John Bollinger, and Alan Andrews. \
Your mastery encompasses both stock fundamentals and intricate technical indicators. \
You possess the ability to decode complex market dynamics, \
providing clear insights and recommendations backed by a thorough understanding of interrelated factors. \
Your expertise extends to practical tools like the pandas_ta module, \
allowing you to navigate data intricacies with ease. \
As a TA authority, your role is to decipher market trends, make informed predictions, and offer valuable perspectives.

given {} TA data as below on the last trading day, what will be the next few days possible stock price movement? 

Summary of Technical Indicators for the tomorrow:
{}""".format(symbol,last_day_summary)
print(sys_prompt)
plt.figure(figsize=(14, 8))
plt.subplot(3, 3, 1)
plt.plot(stock_data.index, stock_data['Adj Close'], label='Adj Close', color='blue')
plt.plot(stock_data.index, stock_data['EMA_50'], label='EMA 50', color='green')
plt.plot(stock_data.index, stock_data['SMA_20'], label='SMA_20', color='orange')
plt.title("Price Trend")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  
plt.xticks(rotation=45, fontsize=8) 
plt.legend()
plt.subplot(3, 3, 2)
plt.plot(stock_data['OBV'], label='On-Balance Volume')
plt.title('On-Balance Volume (OBV) Indicator')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b%d')) 
plt.xticks(rotation=45, fontsize=8) 
plt.legend()
plt.subplot(3, 3, 3)
plt.plot(stock_data['MACD_12_26_9'], label='MACD')
plt.plot(stock_data['MACDh_12_26_9'], label='MACD Histogram')
plt.title('MACD Indicator')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  
plt.xticks(rotation=45, fontsize=8) 
plt.title("MACD")
plt.legend()
plt.subplot(3, 3, 4)
plt.plot(stock_data['RSI_14'], label='RSI')
plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  
plt.xticks(rotation=45, fontsize=8) 
plt.title('RSI Indicator')
plt.subplot(3, 3, 5)
plt.plot(stock_data.index, stock_data['BBU_5_2.0'], label='Upper BB')
plt.plot(stock_data.index, stock_data['BBM_5_2.0'], label='Middle BB')
plt.plot(stock_data.index, stock_data['BBL_5_2.0'], label='Lower BB')
plt.plot(stock_data.index, stock_data['Adj Close'], label='Adj Close', color='brown')
plt.title("Bollinger Bands")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b%d')) 
plt.xticks(rotation=45, fontsize=8)  
plt.legend()
plt.subplot(3, 3, 6)
plt.plot(stock_data.index, stock_data['STOCHk_14_3_3'], label='Stoch %K')
plt.plot(stock_data.index, stock_data['STOCHd_14_3_3'], label='Stoch %D')
plt.title("Stochastic Oscillator")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b%d')) 
plt.xticks(rotation=45, fontsize=8)  
plt.legend()
plt.subplot(3, 3, 7)
plt.plot(stock_data.index, stock_data['WILLR_14'])
plt.title("Williams %R")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b%d')) 
plt.xticks(rotation=45, fontsize=8) 
plt.subplot(3, 3, 8)
plt.plot(stock_data.index, stock_data['ADX_14'])
plt.title("Average Directional Index (ADX)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b%d')) 
plt.xticks(rotation=45, fontsize=8) 
plt.subplot(3, 3, 9)
plt.plot(stock_data.index, stock_data['CMF_20'])
plt.title("Chaikin Money Flow (CMF)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b%d')) 
plt.xticks(rotation=45, fontsize=8)  
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(data['value'])
plt.title('Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
plot_acf(data['value'], lags=30)
plt.title('ACF')
plt.show()
plot_pacf(data['value'], lags=30)
plt.title('PACF')
plt.show()