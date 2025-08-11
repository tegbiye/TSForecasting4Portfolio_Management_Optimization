# Time Series Forecasting for Portfolio Management Optimization 

### The objective is to apply time series forecasting to historical financial data to enhance portfolio management strategies. 



# Project Structure


<pre>
TSForecasting4Portfolio_Management_Optimization/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data
├── notebooks/
│   └── README.md         
├── src/
│   └── __init__.py  
├── tests/
|   ├── __init__.py
│   └── test_add.py         # Unit tests
├── requirements.txt
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