import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load Historical Data
# ---------------------------
# Example: Apple (AAPL) stock for last 5 years
stock_symbol = "AAPL"
data = yf.download(stock_symbol, start="2020-01-01", end="2025-01-01")

print(data.head())

# ---------------------------
# 2. Prepare Data
# ---------------------------
# We'll predict the "Close" price based on the previous day's Close
data['Prev_Close'] = data['Close'].shift(1)
data = data.dropna()

X = np.array(data['Prev_Close']).reshape(-1, 1)
y = np.array(data['Close'])

# ---------------------------
# 3. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ---------------------------
# 4. Train Linear Regression
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# 5. Make Predictions
# ---------------------------
predictions = model.predict(X_test)

# ---------------------------
# 6. Plot Results
# ---------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title(f"{stock_symbol} Stock Price Prediction (Linear Regression)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

# ---------------------------
# 7. Predict Next Day
# ---------------------------
last_close = data['Close'].iloc[-1]
next_day_price = model.predict(np.array([[last_close]]))
print(f"Predicted next day closing price for {stock_symbol}: ${next_day_price[0]:.2f}")
