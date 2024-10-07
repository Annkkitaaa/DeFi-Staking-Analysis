

# DeFi Staking Analysis Dashboard

This repository contains the code for a **DeFi Staking Analysis Dashboard** built using **Streamlit**, **matplotlib**, and **Seaborn** for data visualization, as well as **sklearn** for predictive modeling. The dashboard helps users analyze the historical prices and volatility of popular cryptocurrencies (ETH, SOL, MATIC), study their correlations, and build predictive models for price forecasting using linear regression.

### Live Demo
You can view the live demo of the dashboard here:
[DeFi Staking Analysis Dashboard](https://annkkitaaa-defi-staking-analysis-defi-analysis-ualsu0.streamlit.app/)

## Features
- **Data Fetching:** Real-time historical cryptocurrency data from Yahoo Finance (ETH, SOL, and MATIC).
- **Returns Calculation:** Computes daily percentage returns for each cryptocurrency.
- **Volatility Analysis:** Estimates rolling volatility for each coin using a 30-day window.
- **Correlation Matrix:** Visualizes the correlation between daily returns of ETH, SOL, and MATIC.
- **Predictive Modeling:** Uses historical price data (moving averages and volatility) to build a linear regression model to predict future prices.
- **Visualization:** Dynamic line charts, heatmaps, and interactive price prediction based on user inputs.

## Technologies Used
- **Python 3.x**: Core language used for scripting and data processing.
- **Streamlit**: Framework used to build an interactive web application.
- **Matplotlib & Seaborn**: Used for creating visualizations (price trends, correlation heatmap).
- **yfinance**: Library to fetch cryptocurrency data from Yahoo Finance.
- **scikit-learn**: For building predictive linear regression models.

## Installation

### 1. Clone the Repository
```
git clone https://github.com/Annkkitaaa/defi-staking-analysis.git
cd defi-staking-analysis
```

### 2. Set up a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

### 3. Install Required Libraries
Ensure all dependencies are installed by running the following:
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
yfinance
streamlit
```

### 4. Run the Application
Launch the Streamlit app by running:
```bash
streamlit run defi_analysis.py
```

Once the app is running, it will be accessible at `http://localhost:8501/` on your local machine.

## Usage

1. **Data Fetching**: The app fetches historical price data for Ethereum (ETH), Solana (SOL), and Polygon (MATIC) from Yahoo Finance between the years 2020-2023.
   
2. **Price Trends Visualization**: The dashboard displays a line chart showing the price trends for each cryptocurrency over time.

3. **Correlation Heatmap**: A heatmap shows the correlation between the daily returns of the three cryptocurrencies, helping users understand their interdependence.

4. **Volatility Analysis**: A rolling 30-day volatility is calculated for each coin and plotted as a line chart.

5. **Price Prediction**: The dashboard allows users to input values for features such as the 7-day and 30-day moving averages and volatility for ETH, SOL, or MATIC. It uses a trained linear regression model to predict the future price based on these values.

## Folder Structure
```plaintext
defi-staking-analysis/
│
├── defi_analysis.py          # Main Python script containing the dashboard logic
├── requirements.txt          # Required libraries for running the app
└── README.md                 # Project overview and instructions
```

## Future Improvements
- **Additional Cryptocurrencies**: Add support for more cryptocurrencies like Bitcoin (BTC), Cardano (ADA), etc.
- **Technical Indicators**: Introduce more technical indicators like RSI, MACD, and Bollinger Bands to enhance the predictive models.
- **Advanced Modeling**: Explore more complex machine learning models such as Random Forest, Gradient Boosting, or Neural Networks for more accurate predictions.
