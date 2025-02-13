# Gold Price Prediction Using Data Science and Analysis  

This project leverages Data Science techniques to predict gold prices by analyzing historical trends and influential economic factors. It aims to provide accurate and insightful predictions, helping investors, analysts, and traders make informed decisions.

---

## Project Overview  

Gold is one of the most valuable and widely traded commodities, with prices influenced by various economic indicators like inflation, currency fluctuations, and geopolitical events. This project uses historical gold price data, along with macroeconomic variables, to build predictive models using Data Science and Machine Learning techniques.

---

## Objectives  

- Analyze historical trends and patterns in gold prices.  
- Identify key economic factors influencing gold prices.  
- Build and compare predictive models for accurate gold price forecasting.  
- Visualize data insights and model performance for better interpretability.  

---

## Features  

- **Data Collection and Cleaning:** Import and preprocess historical gold prices and relevant economic indicators.  
- **Exploratory Data Analysis (EDA):** Visualize historical trends, seasonality, and correlations.  
- **Feature Engineering:** Extract useful features like moving averages and volatility indices.  
- **Predictive Modeling:** Implement and compare models such as Linear Regression, LSTM, ARIMA, and XGBoost.  
- **Model Evaluation:** Evaluate models using metrics like MAE, RMSE, and R-Squared.  
- **Visualization and Insights:** Interactive dashboards for data visualization and price predictions.  

---

## Dataset  

The dataset used in this project includes:  
- **Gold Prices:** Historical daily/weekly/monthly gold prices (USD/Ounce).  
- **Economic Indicators:**  
  - Inflation Rates  
  - US Dollar Index (DXY)  
  - Interest Rates  
  - Crude Oil Prices  
  - Stock Market Indices (e.g., S&P 500)  

Data sources include:  
- [World Gold Council](https://www.gold.org/)  
- [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/)  
- [Yahoo Finance](https://finance.yahoo.com/)  

---

## Technologies Used  

- **Programming Language:** Python  
- **Data Analysis Libraries:** Pandas, NumPy  
- **Visualization Libraries:** Matplotlib, Seaborn, Plotly  
- **Machine Learning Models:** Scikit-learn, XGBoost, TensorFlow/Keras (for LSTM)  
- **Time Series Analysis:** Statsmodels (for ARIMA)  
- **Jupyter Notebook:** For interactive data exploration and model development  

---

## Getting Started  

### Prerequisites  
- **Python** (Version 3.8 or above)  
- **Jupyter Notebook** (Recommended for interactive analysis)  
- **Package Manager:** pip or conda  

### Installation  

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/your-username/gold-price-prediction.git
   cd gold-price-prediction
   ```

2. **Create Virtual Environment (Optional but recommended):**  
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: .\env\Scripts\activate
   ```

3. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**  
   ```bash
   jupyter notebook
   ```
   Open `Gold_Price_Prediction.ipynb` to start exploring the data and models.

---

## Data Analysis Workflow  

1. **Data Collection and Cleaning:**  
   - Import historical gold prices and economic indicators using Pandas.  
   - Clean and preprocess the data for time series analysis.  

2. **Exploratory Data Analysis (EDA):**  
   - Analyze historical price trends and patterns.  
   - Examine correlations between gold prices and economic variables.  

3. **Feature Engineering:**  
   - Calculate moving averages, price volatility, and other technical indicators.  
   - Create lag features for time series forecasting.  

4. **Predictive Modeling:**  
   - Implement and compare models:  
     - **Linear Regression:** Basic baseline model.  
     - **ARIMA:** For time series forecasting.  
     - **LSTM (Long Short-Term Memory):** Deep learning model for sequential data.  
     - **XGBoost:** Advanced ensemble model for regression.  

5. **Model Evaluation and Selection:**  
   - Evaluate models using metrics such as:  
     - Mean Absolute Error (MAE)  
     - Root Mean Square Error (RMSE)  
     - R-Squared (R²)  
   - Compare models and select the best-performing one.  

6. **Visualization and Insights:**  
   - Visualize historical trends, model predictions, and error distributions.  
   - Interactive dashboards for exploring model performance.  

---

## Sample Code  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load Data
df = pd.read_csv('data/gold_prices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Feature Engineering
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# Prepare Data
X = df[['MA50', 'MA200']].dropna()
y = df['Close'].shift(-1).dropna()
X, y = X.align(y, join='inner', axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'RMSE: {rmse}')

# Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red')
plt.title('Gold Price Prediction')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD/Ounce)')
plt.legend()
plt.show()
```

---

## Visualization Examples  

- **Historical Trend Analysis:** Line charts showing historical price movements.  
- **Correlation Heatmap:** Relationships between gold prices and economic indicators.  
- **Prediction vs Actual Plot:** Comparing model predictions with actual prices.  
- **Interactive Dashboard:** Explore predictions and insights using Plotly Dash.  

---

## Contributing  

We welcome contributions to this project! To contribute:  
1. Fork the repository.  
2. Create a new branch:  
   ```bash
   git checkout -b feature-name
   ```  
3. Make your changes and commit:  
   ```bash
   git commit -m "Add feature-name"
   ```  
4. Push to your forked repository:  
   ```bash
   git push origin feature-name
   ```  
5. Create a pull request and describe your changes.  

---

## License  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  

---

## Acknowledgments  

- Special thanks to the financial data community for open access to data sources.  
- Gratitude to contributors for continuous support and feedback.  

---


## Future Enhancements  

- Integration with real-time data APIs for dynamic forecasting.  
- Advanced deep learning models (e.g., Transformers).  
- Sentiment analysis of news headlines to enhance predictions.  

---

## Feedback and Support  

Your feedback is valuable! If you encounter any issues or have suggestions, feel free to open an issue on GitHub or reach out via email.  

---

Start predicting gold prices with confidence using **Gold Price Prediction** – Your gateway to financial insights!
