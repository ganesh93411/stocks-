import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for styling
# Set page config
st.set_page_config(page_title="Apple Stock Prediction")

# Add custom CSS to style the top border and text
st.markdown(
    """
    <style>
    .top-border {
        background-color: blue;   /* Border color */
        color: white;  /* Text color */
        font-size: 30px;  /* Text size - increase for larger text */
        font-family: 'Arial', sans-serif;  /* Font style - change to your preferred font */
        font-weight: bold;  /* Text weight */
        padding: 20px 0;  /* Increase padding to make the border taller */
        text-align: center;  /* Center the text horizontally */
        width: 100%;  /* Full width of the page */
    }
    </style>
    """, unsafe_allow_html=True
)

# Add the top border with text
st.markdown('<div class="top-border"> Stock Prediction </div>', unsafe_allow_html=True)

# title
st.title(" Apple Stock Prediction ")

# Display the image at the top with a fixed height and width (rectangular shape)
st.image("https://th.bing.com/th/id/OIP.Ac_eVqBSQ7enrDC6jLQucQAAAA?rs=1&pid=ImgDetMain", width=800)

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    stock = pd.read_csv(uploaded_file)
    stock['Date'] = pd.to_datetime(stock['Date'], dayfirst=True, errors='coerce')
    stock = stock.dropna(subset=['Date'])  # Drop rows with invalid dates
    stock.set_index('Date', inplace=True)

    st.write("### Data Overview")
    st.write(stock)
    
    st.write("### Data Summary")
    st.write(stock.describe())
    
    st.write("### Missing Values")
    st.write(stock.isnull().sum())
    
    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(stock.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Stock Closing Price Over Time
    st.write("### Stock Closing Price Over Time")
    fig, ax = plt.subplots()
    ax.plot(stock.index, stock['Close'], label='Closing Price', color='brown')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Stock Closing Price Over Time')
    ax.legend()
    st.pyplot(fig)
    
    # Moving Averages
    stock['20MA'] = stock['Close'].rolling(window=20).mean()
    stock['50MA'] = stock['Close'].rolling(window=50).mean()
    
    st.write("### Moving Averages")
    fig, ax = plt.subplots()
    ax.plot(stock.index, stock['Close'], label='Closing Price', color='brown')
    ax.plot(stock.index, stock['20MA'], label='20-Day MA', linestyle='dashed')
    ax.plot(stock.index, stock['50MA'], label='50-Day MA', linestyle='dotted')
    ax.set_title('Stock Price with Moving Averages')
    ax.legend()
    st.pyplot(fig)
    
    # Prepare data
    stock['Days'] = np.arange(len(stock))
    X = stock[['Days']]
    y = stock['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model selection
    model_choice = st.selectbox("Select a model to train:", ["Random Forest", "Linear Regression", "XGBoost"])
    
    if st.button("Train Model"):
        model_dict = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression(),
            "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        }
        
        model = model_dict[model_choice]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        accuracy = r2 * 100  # Convert R² score to percentage accuracy
        
        st.write(f"### {model_choice} Model Performance")
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")
        st.write(f"RMSE: {rmse}")
        st.write(f"R² Score: {r2}")
        st.write(f"Accuracy: {accuracy}%")
        
        # Predict next 30 days
        future_days = np.arange(len(stock), len(stock) + 30).reshape(-1, 1)
        future_predictions = model.predict(future_days)
        
        # Display future predictions
        st.write("### Next 30 Days Stock Price Prediction")
        future_dates = pd.date_range(stock.index[-1], periods=30, freq='D')
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
        st.write(future_df)
        
        # Plot future predictions
        fig, ax = plt.subplots()
        ax.plot(stock.index, stock['Close'], label='Historical Close', color='blue')
        ax.plot(future_dates, future_predictions, label='Predicted Close', color='red', linestyle='dashed')
        ax.set_title('Stock Price Prediction for Next 30 Days')
        ax.legend()
        st.pyplot(fig)
        
