# 📈 **NIFTY-50 Stock Price Prediction Using Time Series Forecasting**  


## **📌 Project Overview**  
This project aims to predict **NIFTY-50 stock prices** using **Time Series Forecasting techniques**. The dataset consists of **NIFTY-50 stock market data (2000-2021)**. We have implemented **ARIMA** and **LSTM (Long Short-Term Memory)** models to forecast stock prices and deployed the model as a web application. This project helps investors, traders, and analysts gain insights into future price movements, improving decision-making in the stock market.

---

## **🚀 Features**  
✅ **Data Preprocessing** – This involves multiple steps to clean and transform the raw stock market data:
   - Handling **missing values** by imputation or removal to maintain consistency.
   - Removing **outliers and anomalies** caused by market fluctuations or data errors.
   - **Scaling** the data using MinMaxScaler or StandardScaler to normalize values for better model performance.
   - **Splitting the data** into training and testing sets to evaluate model accuracy.

✅ **Exploratory Data Analysis (EDA)** – EDA is performed to understand the historical patterns and trends in stock prices:
   - **Moving Averages** help smooth fluctuations and highlight trends over time.
   - **Candlestick Charts** provide a visual representation of open, high, low, and closing prices for better analysis.
   - **Seasonal Decomposition** allows us to break down the stock prices into trend, seasonal, and residual components.
   - **Correlation Analysis** helps identify relationships between different stock market indicators.

✅ **Time Series Forecasting Models**:  
   - **ARIMA (AutoRegressive Integrated Moving Average)** – A statistical method that models the time series by capturing trends, seasonality, and noise. It is useful for stationary datasets and provides interpretable results.
   - **LSTM (Long Short-Term Memory Neural Networks)** – A deep learning model designed for sequential data that captures long-term dependencies in stock price movement. This makes it highly effective for predicting future stock prices.

✅ **Evaluation Metrics** – The performance of the forecasting models is assessed using:
   - **Root Mean Square Error (RMSE)** – Measures how much the predicted values deviate from actual values.
   - **Mean Absolute Error (MAE)** – Calculates the average absolute error between predicted and actual values.
   - **Mean Absolute Percentage Error (MAPE)** – Expresses forecast accuracy as a percentage, making it easier to interpret results.

✅ **Web App Deployment** – The trained model is deployed using **Flask** and **Streamlit**, providing users with an interactive interface where they can:
   - Input a date range for predictions.
   - Visualize stock price trends through dynamic plots.
   - Compare actual vs. predicted prices for performance evaluation.

---

## **📂 Dataset**  
- **Source**: [NIFTY-50 Stock Market Data (2000 - 2021) – Kaggle](https://www.kaggle.com/datasets)  
- **Features**:  
  - **Date** (Index)  
  - **Open Price** – The price at which trading begins for a given day.
  - **High Price** – The highest price recorded for the stock during the trading day.
  - **Low Price** – The lowest price recorded for the stock during the trading day.
  - **Close Price** – The final trading price of the stock at market close.
  - **Volume** – The number of shares traded during the day, indicating liquidity and market activity.

---

## **📊 Model Implementation**  

### **🔹 ARIMA Model**  
The **ARIMA (AutoRegressive Integrated Moving Average)** model is a widely used statistical method for time series forecasting. It consists of three main components:
- **AutoRegressive (AR)** – Uses past values to predict future values based on the assumption that past trends continue.
- **Integrated (I)** – Differencing is applied to remove non-stationarity and stabilize the time series data.
- **Moving Average (MA)** – Uses past forecast errors to refine predictions and improve accuracy.

**Implementation using `statsmodels`:**  

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(train, order=(5,1,0))
arima_model = model.fit()

# Forecast
arima_predictions = arima_model.forecast(steps=len(test))
```

**Pros:**
✅ Works well with stationary time series.
✅ Provides interpretable results with clear coefficients.
✅ Requires relatively low computational power compared to deep learning models.

**Cons:**
❌ Struggles with non-linear data patterns.
❌ Requires manual tuning of hyperparameters (p, d, q).
❌ Less effective for complex stock price movements.

---

### **🔹 LSTM Model**  
The **LSTM (Long Short-Term Memory)** model is a type of recurrent neural network (RNN) that is specifically designed to capture long-term dependencies in time series data. Unlike traditional RNNs, LSTM mitigates the vanishing gradient problem using its memory cell mechanism.

Key components of LSTM:
- **Forget Gate** – Decides which information should be discarded from memory.
- **Input Gate** – Determines which new information is relevant and should be added.
- **Output Gate** – Controls how much of the stored information is used to influence the output.

**Implementation using TensorFlow/Keras:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)
```

**Pros:**
✅ Handles long-term dependencies in stock price data effectively.
✅ Captures complex non-linear relationships, leading to more accurate predictions.
✅ Adaptable to different time series forecasting problems.

**Cons:**
❌ Requires large datasets and significant computational resources.
❌ Training time is significantly longer than statistical models like ARIMA.
❌ More prone to overfitting if not carefully regularized.

---

## **🔗 Connect With Me**  
💼 [LinkedIn](www.linkedin.com/in/jacksonjacobl)  
📧 Email: your-jackson24499@gmail.com  
