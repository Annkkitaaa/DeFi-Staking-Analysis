import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import streamlit as st

# Data Collection and Preprocessing
def fetch_crypto_data(crypto, start_date, end_date):
    data = yf.download(crypto, start=start_date, end=end_date)
    return data['Close']

def preprocess_data(eth_data, sol_data, matic_data):
    df = pd.DataFrame({
        'ETH': eth_data,
        'SOL': sol_data,
        'MATIC': matic_data
    })
    df['Date'] = df.index
    df = df.reset_index(drop=True)
    return df

# Data Analysis
def calculate_returns(df):
    for coin in ['ETH', 'SOL', 'MATIC']:
        df[f'{coin}_Returns'] = df[coin].pct_change()
    return df

def calculate_volatility(df):
    for coin in ['ETH', 'SOL', 'MATIC']:
        df[f'{coin}_Volatility'] = df[f'{coin}_Returns'].rolling(window=30).std() * np.sqrt(365)
    return df

def calculate_correlation(df):
    return df[['ETH_Returns', 'SOL_Returns', 'MATIC_Returns']].corr()

# Predictive Modeling
def create_features(df):
    for coin in ['ETH', 'SOL', 'MATIC']:
        df[f'{coin}_MA7'] = df[coin].rolling(window=7).mean()
        df[f'{coin}_MA30'] = df[coin].rolling(window=30).mean()
    return df.dropna()

def train_model(df, coin):
    features = [f'{coin}_MA7', f'{coin}_MA30', f'{coin}_Volatility']
    X = df[features]
    y = df[coin]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Visualization
def plot_price_trends(df):
    plt.figure(figsize=(12, 6))
    for coin in ['ETH', 'SOL', 'MATIC']:
        plt.plot(df['Date'], df[coin], label=coin)
    plt.title('Price Trends')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    return plt

def plot_correlation_heatmap(corr):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    return plt

# Main Analysis Function
def run_analysis():
    # Fetch data
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    eth_data = fetch_crypto_data('ETH-USD', start_date, end_date)
    sol_data = fetch_crypto_data('SOL-USD', start_date, end_date)
    matic_data = fetch_crypto_data('MATIC-USD', start_date, end_date)

    # Preprocess data
    df = preprocess_data(eth_data, sol_data, matic_data)
    df = calculate_returns(df)
    df = calculate_volatility(df)
    df = create_features(df)

    # Analysis
    correlation = calculate_correlation(df)

    # Modeling
    eth_model, X_test, y_test = train_model(df, 'ETH')
    sol_model, _, _ = train_model(df, 'SOL')
    matic_model, _, _ = train_model(df, 'MATIC')

    return df, correlation, eth_model, sol_model, matic_model

# Streamlit Dashboard
def run_dashboard():
    st.title('DeFi Staking Analysis Dashboard')

    df, correlation, eth_model, sol_model, matic_model = run_analysis()

    st.header('Price Trends')
    st.pyplot(plot_price_trends(df))

    st.header('Correlation Analysis')
    st.pyplot(plot_correlation_heatmap(correlation))

    st.header('Volatility Comparison')
    st.line_chart(df[['ETH_Volatility', 'SOL_Volatility', 'MATIC_Volatility']])

    st.header('Predictive Modeling')
    coin = st.selectbox('Select Cryptocurrency', ['ETH', 'SOL', 'MATIC'])
    model = {'ETH': eth_model, 'SOL': sol_model, 'MATIC': matic_model}[coin]
    features = [f'{coin}_MA7', f'{coin}_MA30', f'{coin}_Volatility']
    
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f'Enter {feature}')
    
    if st.button('Predict Price'):
        prediction = model.predict(pd.DataFrame(user_input, index=[0]))
        st.write(f'Predicted {coin} Price: ${prediction[0]:.2f}')

if __name__ == "__main__":
    run_dashboard()