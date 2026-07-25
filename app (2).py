import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# --- Configuration (from notebook's best model) ---
# These values are derived from the best model found in the notebook
BEST_MODEL_TYPE = "SimpleRNN"
BEST_WINDOW_SIZE = 24
MODEL_FILENAME = f'{BEST_MODEL_TYPE}_window_{BEST_WINDOW_SIZE}_best_model.h5'
SCALER_FILENAME = 'scaler.pkl'
DATA_FILENAME = 'monthly_milk_production.csv'

# --- Data Loading ---
@st.cache_data
def load_data():
    # Ensure the CSV is available in the same directory or adjust path
    df = pd.read_csv(DATA_FILENAME)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
    df.set_index('Date', inplace=True)
    return df

@st.cache_resource
def load_model_and_scaler():
    # Load scaler
    if not os.path.exists(SCALER_FILENAME):
        st.error(f"Scaler file not found: {SCALER_FILENAME}. Please ensure it's in the same directory as app.py")
        st.stop()
    scaler = joblib.load(SCALER_FILENAME)

    # Load the Keras model
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"Model file not found: {MODEL_FILENAME}. Please ensure it's in the same directory as app.py")
        st.stop()
    model = tf.keras.models.load_model(MODEL_FILENAME)
    return model, scaler

# --- Helper Function (from notebook) ---
def create_dataset(data, window_size):
    X_data, y_data = [], []
    for i in range(window_size, len(data)):
        X_data.append(data[i-window_size:i, 0])
        y_data.append(data[i, 0])
    return np.array(X_data), np.array(y_data)

# --- Forecasting Logic ---
def forecast_future(model, scaler, df_historical, window_size, future_months=12):
    # Scale historical data
    df_scaled = scaler.transform(df_historical) # Use transform, not fit_transform for new data

    # Get the last 'window_size' observations from the scaled data to start forecasting
    last_window = df_scaled[-window_size:]

    # Create a list to store future predictions (scaled)
    future_predictions_scaled = []

    current_input = last_window.reshape(1, window_size, 1)

    for _ in range(future_months):
        # Predict the next value and extract the scalar
        next_prediction_scaled = model.predict(current_input, verbose=0).item()
        future_predictions_scaled.append(next_prediction_scaled)

        # Update the input sequence: remove the first element and add the new prediction
        next_prediction_reshaped = np.array([[next_prediction_scaled]])
        current_input = np.append(current_input[:, 1:, :], next_prediction_reshaped.reshape(1, 1, 1), axis=1)

    # Inverse transform the future predictions to the original scale
    future_predictions_original = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

    # Create a date range for the forecasted months
    last_date = df_historical.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=future_months + 1, freq='MS')[1:]

    forecast_df = pd.DataFrame(future_predictions_original, index=forecast_dates, columns=['Forecasted Production'])
    return forecast_df

# --- Streamlit App ---
st.title('Monthly Milk Production Forecasting')

st.write('''
This application forecasts monthly milk production for the next 12 months
using a pre-trained Recurrent Neural Network model.
''')

# Load data and model
df_historical = load_data()
best_model, scaler = load_model_and_scaler()

st.subheader('Historical Milk Production')
st.line_chart(df_historical['Production'])

st.subheader(f'Forecasted Milk Production for Next 12 Months using {BEST_MODEL_TYPE} Model (Window Size: {BEST_WINDOW_SIZE})')

if st.button('Generate Forecast'):
    with st.spinner('Generating forecast...'):
        forecast_df = forecast_future(best_model, scaler, df_historical, BEST_WINDOW_SIZE)

        st.line_chart(forecast_df)

        st.write('### Combined Historical and Forecasted Production')
        combined_df = pd.concat([df_historical['Production'], forecast_df['Forecasted Production']])
        st.line_chart(combined_df)

        st.write('### Forecast Data')
        st.dataframe(forecast_df)

st.sidebar.header('About')
st.sidebar.info('''
This app uses a Sequential Keras model (SimpleRNN, LSTM, or GRU)
trained on historical monthly milk production data.
The model was optimized using GridSearchCV to find the best hyperparameters.
''')
