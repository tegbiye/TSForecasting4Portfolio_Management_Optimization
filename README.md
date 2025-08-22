# Time Series Forecasting for Portfolio Management Optimization 

### The objective is to apply time series forecasting to historical financial data to enhance portfolio management strategies. 

# Task 1: Preprocess and Explore the Data

### Key Insights from Preprocessing and Exploration of the Dataset

✅ The dataset covers historical financial data for three assets—Vanguard Total Bond Market ETF (BND), S&P 500 ETF (SPY), and Tesla (TSLA)—from July 1, 2015, to July 30, 2025, sourced via YFinance. 

✅ The analysis includes data cleaning (handling multi-index structure, interpolation of missing values, type conversions), computation of daily returns, rolling statistics, outlier detection, stationarity tests (ADF), and risk metrics (VaR and Sharpe Ratio). 

✅ Below, I document the key insights as requested, focusing on TSLA while contextualizing it against BND (low-risk bond ETF) and SPY (moderate-risk market benchmark). 

✅ Insights are derived from the notebook's outputs, including descriptive statistics, visualizations (e.g., rolling mean/std plots), and computed metrics.

![BND Banner](./images/BND_Close_vs_Time_series.png)

![SPY Banner](./images/SPY_Close_Time_Series.png)

![TSLA Banner](./images/TSLA_Close_Time_Series.png)

# Task 2: Develop Time Series Forecasting Models

With the split of the data as Train (2015-2023), Test (2024-2025)

And the models used are SARIMA, and LSTM

The metrics result are done with MAE, RMSE, MAPE

#### Results

SARIMA Metrics: MAE=82.78, RMSE=103.37, MAPE=29.18%
LSTM Metrics: MAE=10.17, RMSE=14.49, MAPE=3.69%

![BND Banner](./images/output_forecast_comparison.png)

#### Discussion

1. Mean Absolute Error (MAE)

    Measures the average absolute difference between the predicted and actual values.

    SARIMA: 82.78 → On average, SARIMA’s predictions are about $82.78 away from the actual stock price.

    LSTM: 10.17 → On average, LSTM’s predictions are about $10.17 away.

    👉 Lower is better → LSTM performs much better.
2. Root Mean Squared Error (RMSE)

    Similar to MAE but penalizes larger errors more (since errors are squared).

    SARIMA: 103.37

    LSTM: 14.49

    👉 Again, lower is better → LSTM clearly outperforms SARIMA.

3. Mean Absolute Percentage Error (MAPE)

    Expresses error as a percentage of the actual values → useful for interpretability.

    SARIMA: 29.18% → Predictions are off by ~29% on average.

    LSTM: 3.69% → Predictions are off by ~3.7% on average.

    👉 LSTM provides far more accurate percentage-based predictions.

The LSTM model is significantly more accurate than SARIMA for forecasting TSLA stock price, as seen by the much lower MAE, RMSE, and MAPE values.

SARIMA has much higher forecast uncertainty (shown in the wide gray cone in the plot), while LSTM follows the test data more closely.

# Project Structure


<pre>
TSForecasting4Portfolio_Management_Optimization/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data
├── images/
├── notebooks/
│   └── README.md
|   ├── forcast-model-ts.ipynb   # forecast model time series
|   └── preprocess-explore.ipynb   # Preprocess and explore
├── src/
│   └── __init__.py
|   ├── forecast.py        # helper function for forcasting
|   └── preprocess.py      # helper function for preprocess
├── tests/
|   ├── __init__.py
|   ├── test_preprocess.py   # unit test for preprocess
|   ├── test_forecast.py     # unit test for forecast
│   └── test_add.py         # Unit tests
├── requirements.txt        # required libs
├── .gitignore
├── LICENSE
└── README.md
</pre>


# Getting Started

Clone the repository

`git clone https://github.com/tegbiye/TSForecasting4Portfolio_Management_Optimization.git`

`cd TSForecasting4Portfolio_Management_Optimization`

Create environment using venv

`python -m venv .telenv`

Activate the environment

`.telenv\Scripts\activate` (Windows)

`source .telenv\bin\activate` (Linux / Mac)

Install Dependencies

`pip install -r requirements.txt`

📜 License This project is licensed un