import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np

def load_and_prepare_data(file_path="apple_5yr_one.csv"):
    df=pd.read_csv(file_path)
    df=df[["Date","Close"]]
    df['Date']=pd.to_datetime(df['Date'])
    df=df.rename(columns={"Date":"ds","Close":"y"})
    df=df.sort_values('ds')
    return df

def train_model_and_forecast(df,periods=90):
    model=Prophet()
    model.fit(df)
    future=model.make_future_dataframe(periods=periods)
    forecast=model.predict(future)
    return model,forecast

def plot_forecast(model,forecast):
    fig=model.plot(forecast)
    plt.title("Stock Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

def evaluate_forecast(df,forecast):
    merged=pd.merge(df,forecast[['ds','yhat']],on="ds",how='inner')

    y_true=merged['y']
    y_pred=merged['yhat']

    mae=mean_absolute_error(y_true=y_true,y_pred=y_pred)
    rmse=np.sqrt(mean_squared_error(y_true=y_true,y_pred=y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"📊 Evaluation Metrics:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")


if __name__=="__main__":
    df=load_and_prepare_data()
    model,forecast=train_model_and_forecast(df)
    plot_forecast(model,forecast)
    evaluate_forecast(df,forecast)