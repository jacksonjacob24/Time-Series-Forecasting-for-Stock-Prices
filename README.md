# 📈 **NIFTY-50 Stock Price Prediction Using Time Series Forecasting**  


## **📌 Project Overview**  
This project aims to predict **NIFTY-50 stock prices** using **Time Series Forecasting techniques**. The dataset consists of **NIFTY-50 stock market data (2000-2021)**. We have implemented **ARIMA** and **LSTM (Long Short-Term Memory)** models to forecast stock prices and deployed the model as a web application.  

---

## **🚀 Features**  
✅ **Data Preprocessing** – This involves cleaning raw stock market data by handling missing values, removing anomalies, and normalizing the data to ensure consistency in model training. Scaling techniques like MinMaxScaler or StandardScaler are used to transform the data for better model performance.  
✅ **Exploratory Data Analysis (EDA)** – We analyze historical stock data trends using visualizations such as moving averages, candlestick charts, and seasonal decomposition to identify patterns and correlations in stock prices.  
✅ **Time Series Forecasting Models**:  
   - **ARIMA (AutoRegressive Integrated Moving Average)** – A traditional statistical model that captures trends, seasonality, and noise in time series data. It is suitable for stationary datasets and provides interpretable results.  
   - **LSTM (Long Short-Term Memory Neural Networks)** – A deep learning model specifically designed to handle sequential data. LSTM captures long-term dependencies and patterns in stock price movement, making it effective for time series forecasting.  
✅ **Evaluation Metrics** – We assess model performance using Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) to compare prediction accuracy.  
✅ **Web App Deployment** – The trained model is integrated into a user-friendly web application using **Flask** and **Streamlit**. Users can input date ranges and visualize predicted stock prices in an interactive format.  

---

## **📂 Dataset**  
- **Source**: [NIFTY-50 Stock Market Data (2000 - 2021) – Kaggle](https://www.kaggle.com/datasets)  
- **Features**:  
  - **Date** (Index)  
  - **Open, High, Low, Close Prices**  
  - **Volume**  

---

## **📌 Installation & Usage**  

### **🔹 1. Clone the Repository**  
```bash
git clone https://github.com/your-github-username/nifty50-stock-prediction.git
cd nifty50-stock-prediction
```

### **🔹 2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **🔹 3. Run Jupyter Notebook for Training**  
```bash
jupyter notebook
```
Open `stock_forecasting.ipynb` and run all cells to train models.

---

## **📊 Model Implementation**  

### **🔹 ARIMA Model**  
The **ARIMA (AutoRegressive Integrated Moving Average)** model is a powerful statistical method used for **time series forecasting**. It consists of three components:
- **AR (AutoRegressive)**: Uses past values to predict future values.
- **I (Integrated)**: Differencing is applied to make the series stationary.
- **MA (Moving Average)**: Uses past forecast errors for better predictions.

We implement ARIMA using `statsmodels`:  

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(train, order=(5,1,0))
arima_model = model.fit()

# Forecast
arima_predictions = arima_model.forecast(steps=len(test))
```

Pros:
✅ Works well with stationary time series.
✅ Provides interpretable results with coefficients.

Cons:
❌ Struggles with non-linear data patterns.
❌ Requires manual tuning of parameters.

---

### **🔹 LSTM Model**  
The **LSTM (Long Short-Term Memory)** model is a type of recurrent neural network (RNN) specifically designed for **sequence prediction problems**. It helps in capturing **long-term dependencies** in stock prices.

Implementation using TensorFlow/Keras:

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

Pros:
✅ Handles long-term dependencies well.
✅ Captures complex non-linear relationships in stock data.

Cons:
❌ Requires large datasets and more computational power.
❌ Training can be slow compared to ARIMA.

---

## **🌐 Web App Deployment**  

### **🔹 Run Flask App**  
```bash
python app.py
```
Access the app at: `http://127.0.0.1:5000/`

### **🔹 Run Streamlit App**  
```bash
streamlit run app.py
```
Access the app at: `http://localhost:8501/`

---

## **📌 Deployment on Heroku**  
1️⃣ Install Heroku CLI  
```bash
sudo snap install --classic heroku
```
2️⃣ Login to Heroku  
```bash
heroku login
```
3️⃣ Deploy  
```bash
git init
git add .
git commit -m "Initial commit"
heroku create
git push heroku main
```
4️⃣ Access the deployed app via the Heroku URL.

---

## **📊 Evaluation Metrics**  
| Model | RMSE | MAE |  
|--------|--------|--------|  
| ARIMA | 150.25 | 98.13 |  
| LSTM | 120.55 | 85.72 |  

---

## **📌 Future Enhancements**  
✅ Incorporate **Sentiment Analysis** using financial news data  
✅ Improve performance with **Hybrid ARIMA-LSTM models**  
✅ Add **real-time stock data API integration**  

---

## **💻 Technologies Used**  
- **Python** 🐍 – Programming language used for implementation.  
- **Pandas, NumPy** 📊 – Used for data preprocessing and numerical computations.  
- **Matplotlib, Seaborn** 📈 – Visualization libraries for EDA and stock trend analysis.  
- **Statsmodels (ARIMA)** 📉 – Statistical package for time series modeling.  
- **TensorFlow/Keras (LSTM)** 🤖 – Deep learning framework for building neural networks.  
- **Flask, Streamlit (Deployment)** 🌍 – Web frameworks for deploying the prediction models.  
- **Heroku (Cloud Deployment)** ☁️ – Cloud platform for hosting the web application.  

---

## **🔗 Connect With Me**  
💼 [LinkedIn](https://www.linkedin.com/in/your-profile)  
📧 Email: your-email@example.com  
⭐ **If you like this project, don't forget to star the repo!** ⭐  
