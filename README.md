# 📈 **NIFTY-50 Stock Price Prediction Using Time Series Forecasting**  


## **📌 Project Overview**  
This project aims to predict **NIFTY-50 stock prices** using **Time Series Forecasting techniques**. The dataset consists of **NIFTY-50 stock market data (2000-2021)**. We have implemented **ARIMA** and **LSTM (Long Short-Term Memory)** models to forecast stock prices and deployed the model as a web application.  

---

## **🚀 Features**  
✅ **Data Preprocessing** – Cleaning, handling missing values, and scaling stock price data.  
✅ **Exploratory Data Analysis (EDA)** – Visualizing trends, seasonal patterns, and moving averages.  
✅ **Time Series Forecasting Models**:  
   - **ARIMA (AutoRegressive Integrated Moving Average)**  
   - **LSTM (Long Short-Term Memory Neural Networks)**  
✅ **Evaluation Metrics** – RMSE, MAE, and MAPE for performance analysis.  
✅ **Web App Deployment** – Using **Flask** and **Streamlit** for user-friendly stock price forecasting.  

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
- **Captures trends, seasonality, and residual noise**  
- Uses `statsmodels` for implementation  

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(train, order=(5,1,0))
arima_model = model.fit()

# Forecast
arima_predictions = arima_model.forecast(steps=len(test))
```

---

### **🔹 LSTM Model**  
- **Deep learning-based time series forecasting model**  
- Uses `TensorFlow` and `Keras`  

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
- **Python** 🐍  
- **Pandas, NumPy** 📊  
- **Matplotlib, Seaborn** 📈  
- **Statsmodels (ARIMA)** 📉  
- **TensorFlow/Keras (LSTM)** 🤖  
- **Flask, Streamlit (Deployment)** 🌍  
- **Heroku (Cloud Deployment)** ☁️  

---

## **🔗 Connect With Me**  
💼 [LinkedIn](linkedin.com/in/jacksonjacobl)  
📧 Email:  
