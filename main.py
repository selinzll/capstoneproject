import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
def load_data(file):
   df = pd.read_csv(file, parse_dates=['Date'])
   df.set_index('Date', inplace=True)
   df = df.asfreq('D')
   df['Revenue'] = df['Revenue'].ffill()
   return df

def group_data(df, view_mode):
   if view_mode == "Monthly":
       return df.resample('ME').sum()
   elif view_mode == "Quarterly":
       return df.resample('QE').sum()
   return df
def time_series_split(df, test_size=0.2):
   df = df.sort_index()
   n_rows = len(df)
   split_point = int(n_rows * (1 - test_size))
   train_df = df.iloc[:split_point]
   test_df = df.iloc[split_point:]
   return train_df, test_df
def train_evaluate_linear_regression(train_df, test_df):
   train_df = train_df.copy()
   test_df = test_df.copy()

   train_df['Days'] = (train_df.index - train_df.index.min()).days
   X_train = train_df[['Days']]
   y_train = train_df['Revenue']

   model = LinearRegression()
   model.fit(X_train, y_train)

   test_df['Days'] = (test_df.index - train_df.index.min()).days
   X_test = test_df[['Days']]
   y_test = test_df['Revenue']

   predictions = model.predict(X_test)

   mse = mean_squared_error(y_test, predictions)
   rmse = np.sqrt(mse)
   r2 = r2_score(y_test, predictions)

   return model, predictions, mse, rmse, r2, y_test.index

def linear_regression_final_forecast(df, forecast_days):
   df = df.copy()
   df['Days'] = (df.index - df.index.min()).days
   X = df[['Days']]
   y = df['Revenue']

   model = LinearRegression()
   model.fit(X, y)

   future_days = np.arange(X['Days'].max() + 1, X['Days'].max() + forecast_days + 1).reshape(-1, 1)
   future_dates = [df.index.max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
   forecast = model.predict(future_days)

   forecast_df = pd.DataFrame({'Forecasted Revenue': forecast}, index=future_dates)
   return forecast_df
def train_evaluate_knn_regression(train_df, test_df, n_neighbors):
   train_df = train_df.copy()
   test_df = test_df.copy()

   train_df['Days'] = (train_df.index - train_df.index.min()).days
   X_train = train_df[['Days']]
   y_train = train_df['Revenue']

   model = KNeighborsRegressor(n_neighbors=n_neighbors)
   model.fit(X_train, y_train)

   test_df['Days'] = (test_df.index - train_df.index.min()).days
   X_test = test_df[['Days']]
   y_test = test_df['Revenue']

   predictions = model.predict(X_test)

   mse = mean_squared_error(y_test, predictions)
   rmse = np.sqrt(mse)
   r2 = r2_score(y_test, predictions)

   return model, predictions, mse, rmse, r2, y_test.index

def knn_regression_final_forecast(df, forecast_days, n_neighbors):
   df = df.copy()
   df['Days'] = (df.index - df.index.min()).days
   X = df[['Days']]
   y = df['Revenue']

   model = KNeighborsRegressor(n_neighbors=n_neighbors)
   model.fit(X, y)

   future_days = np.arange(X['Days'].max() + 1, X['Days'].max() + forecast_days + 1).reshape(-1, 1)
   future_dates = [df.index.max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
   forecast = model.predict(future_days)

   forecast_df = pd.DataFrame({'Forecasted Revenue': forecast}, index=future_dates)
   return forecast_df
def time_series_forecast(df, forecast_days=30, trend='add', seasonal='add', seasonal_periods=12):
   if len(df) < 2 * seasonal_periods:
       st.warning(f"Not enough data for two full seasonal cycles ({2 * seasonal_periods} data points required). "
                  f"Adjusting seasonal periods to 1 if default 12 is too long.")
       if len(df) < 24 and seasonal_periods == 12:
           seasonal_periods = 1
       if len(df) < 2 * seasonal_periods:
            raise ValueError("Still not enough data for seasonal model, even with adjusted periods.")

   model = ExponentialSmoothing(df['Revenue'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
   fit = model.fit()
   forecast = fit.forecast(forecast_days)

   forecast_df = pd.DataFrame({'Forecasted Revenue': forecast}, index=forecast.index)
   return forecast_df, fit

st.title("Predict Your Money")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
forecast_days = st.slider("How many days to forecast?", min_value=7, max_value=180, value=30, step=7)
model_choice = st.selectbox("Choose a prediction model:",
                          ["Linear Regression", "Time Series Forecasting - Exponential Smoothing", "K-Nearest Neighbors"])
view_mode = st.radio("View data by:", ["Daily", "Monthly", "Quarterly"])

n_neighbors = None
if model_choice == "K-Nearest Neighbors":
   n_neighbors = st.slider("Number of Neighbors (k) for K-NN", min_value=1, max_value=20, value=5, step=1)

test_size = st.slider("Test Set Size for Evaluation (for Linear/KNN)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)


if uploaded_file:
   try:
       df = load_data(uploaded_file)

       if len(df) < 20 and (model_choice == "Linear Regression" or model_choice == "K-Nearest Neighbors"):
           st.warning("Not enough data for a meaningful train-test split. Consider using more data or adjust test size.")
           pass

       train_df, test_df = time_series_split(df, test_size=test_size)

       st.subheader("Model Evaluation on Test Set")
       metrics_display = {}
       forecast_df = None

       if model_choice == "Linear Regression":
           model, predictions, mse, rmse, r2, test_index = train_evaluate_linear_regression(train_df, test_df)
           metrics_display["Linear Regression"] = {"MSE": mse, "RMSE": rmse, "R-squared": r2}
           forecast_df = linear_regression_final_forecast(df, forecast_days)

           test_predictions_df = pd.DataFrame({'Predicted Revenue': predictions}, index=test_index)
           actual_test_df = test_df[['Revenue']]

       elif model_choice == "Time Series Forecasting - Exponential Smoothing":

           forecast_df, es_fit = time_series_forecast(df, forecast_days=forecast_days)
           metrics_display["Exponential Smoothing"] = {"AIC": es_fit.aic, "BIC": es_fit.bic}
           test_predictions_df = pd.DataFrame()
           actual_test_df = pd.DataFrame()

       elif model_choice == "K-Nearest Neighbors":
           if n_neighbors is None:
               n_neighbors = 5
           model, predictions, mse, rmse, r2, test_index = train_evaluate_knn_regression(train_df, test_df, n_neighbors)
           metrics_display["K-Nearest Neighbors"] = {"MSE": mse, "RMSE": rmse, "R-squared": r2}
           forecast_df = knn_regression_final_forecast(df, forecast_days, n_neighbors)

           test_predictions_df = pd.DataFrame({'Predicted Revenue': predictions}, index=test_index)
           actual_test_df = test_df[['Revenue']]


       if metrics_display:
           st.write("---")
           st.write(f"### Evaluation Metrics for {model_choice} on Test Set")
           for model_name, metrics in metrics_display.items():
               for metric, value in metrics.items():
                   st.write(f"- {metric}: {value:.4f}")
           st.write("---")


       st.subheader("Revenue Forecast & Test Set Performance")

       past_grouped = group_data(df[['Revenue']], view_mode)

       future_grouped = pd.DataFrame()
       if forecast_df is not None and 'Forecasted Revenue' in forecast_df.columns:
           future_grouped = group_data(forecast_df[['Forecasted Revenue']], view_mode)

       fig = go.Figure()

       fig.add_trace(go.Scatter(
           x=past_grouped.index, y=past_grouped['Revenue'],
           mode='lines', name='Past Revenue',
           line=dict(color='royalblue')
       ))

       if not future_grouped.empty:
           fig.add_trace(go.Scatter(
               x=future_grouped.index, y=future_grouped['Forecasted Revenue'],
               mode='lines', name='Forecasted Revenue',
               line=dict(color='firebrick', dash='dash')
           ))

       if not test_predictions_df.empty and not actual_test_df.empty and (model_choice == "Linear Regression" or model_choice == "K-Nearest Neighbors"):
           fig.add_trace(go.Scatter(
               x=group_data(actual_test_df, view_mode).index,
               y=group_data(actual_test_df, view_mode)['Revenue'],
               mode='lines+markers', name='Actual Test Revenue',
               line=dict(color='darkgreen'),
               marker=dict(symbol='circle', size=4)
           ))
           fig.add_trace(go.Scatter(
               x=group_data(test_predictions_df, view_mode).index,
               y=group_data(test_predictions_df, view_mode)['Predicted Revenue'],
               mode='lines+markers', name='Predicted Test Revenue',
               line=dict(color='orange', dash='dot'),
               marker=dict(symbol='square', size=4)
           ))


       fig.update_layout(
           title='Revenue Forecast & Test Set Performance',
           xaxis_title='Date',
           yaxis_title='Revenue',
           legend=dict(x=0, y=1, xanchor='left'),
           template='plotly_white',
           hovermode='x unified'
       )
       st.plotly_chart(fig, use_container_width=True)

   except Exception as e:
       st.error(f"An error occurred: {e}")
       st.exception(e)