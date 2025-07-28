import pandas as pd
from prophet import Prophet
import streamlit as st
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
from io import BytesIO

@st.cache_data
def load_data(file_path="apple_5yr_one.csv"):
    df=pd.read_csv(file_path)
    df=df[["Date","Close"]]
    df=df.rename(columns={"Date":"ds","Close":"y"})
    df['ds']=pd.to_datetime(df['ds'])
    df=df.sort_values("ds")

    return df

def train_model_and_forecast(df,periods):
    model=Prophet()
    model.fit(df)
    future=model.make_future_dataframe(periods=periods)
    forecast=model.predict(future)

    return model,forecast

def evaluate_forecast(df,forecast):
    merged=pd.merge(df,forecast[['ds','yhat']],on='ds',how='inner')
    y_true=merged['y']
    y_pred=merged['yhat']
    mae=mean_absolute_error(y_true,y_pred)
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    mape=np.mean(np.abs((y_true - y_pred) / y_true))*100

    return mae,rmse,mape

def get_csv_download_link(df):
    buffer=BytesIO() # temporary memory so that fle need no to be stored in disk before offering to download
    df.to_csv(buffer,index=False) # data in df goes to buffer in csv format
    return buffer # the csv data in bufer is retured

def main():
    st.set_page_config(page_title="📈 Apple Stock Forecasting App",layout="wide")
    st.markdown("""
        <style>
            /* Set Open Sans font and gradient background */
            html, body, .main, .block-container {
                background-image: linear-gradient(to top, #a8edea 0%, #fed6e3 100%);
                font-family: 'Open Sans', sans-serif !important;
                color: #222 !important;
            }

            /* Fix title and subtitle text colors */
            h1, h2, h3, h4, h5, h6, p, span, label, div, .stTextInput label {
                color: #222 !important;
            }

            /* Forecast slider styling */
            .stSlider > div {
                background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%) !important;
                padding: 10px;
                border-radius: 10px;
            }

            /* Button color fix */
            .stButton > button {
                background-color: #ffe1e1 !important;
                color: #222 !important;
                font-weight: 600;
                border: 1px solid #f3b6b6;
                border-radius: 8px;
                padding: 0.6rem 1.5rem;
                transition: 0.3s ease-in-out;
            }

            ..stButton > button:hover {
                background-color: #fbdada !important;
                border-color: #f1a9a9;
                color: #111 !important;
            }

            /* Download Button - off-white */
            button[title="📥 Download Forecast CSV"] {
                background-color: #f7f7f7 !important;
                color: #333 !important;
                border: 1px solid #ddd !important;
                border-radius: 6px !important;
                font-weight: 500 !important;
            }

            button[title="📥 Download Forecast CSV"]:hover {
                background-color: #e6e6e6 !important;
                color: #111 !important;
                border: 1px solid #ccc !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("📈 Apple Stock Price Forecasting App")
    st.markdown("Predict the future of Apple stock prices with **Facebook Prophet**, trained on real 5-year historical data.")

    df=load_data()

    st.subheader("Forecast Settings")
    periods=st.slider("📅 Select number of future days to forecast",min_value=30,max_value=365,value=90,step=15)

    if st.button("🔮 Generate Forecast"):
        with st.spinner("Training model and forecasting"):
            model,forecast=train_model_and_forecast(df,periods)
            mae,rmse,mape=evaluate_forecast(df,forecast)

            # forecast Plot
            st.subheader("📊 Forecast Plot")
            fig=model.plot(forecast)
            ax=fig.gca()
            ax.set_title("Apple Stock Price Forecast",fontsize=16)
            ax.set_xlabel("Date",fontsize=12)
            ax.set_ylabel("Predicted Close Price (USD)", fontsize=12)
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)

            # Evaluation
            st.subheader("🧮 Model Evaluation")
            st.markdown(
                f"""
            - **MAE (Mean Absolute Error):** `{mae:.2f}`
            - **RMSE (Root Mean Squared Error):** `{rmse:.2f}`
            - **MAPE (Mean Absolute Percentage Error):** `{mape:.2f}%`
            """
            )

            # Forecast Table
            st.subheader("📄 Forecasted Data")
            forecast_display=forecast[["ds",'yhat','yhat_lower','yhat_upper']].tail(periods)
            st.dataframe(forecast_display,use_container_width=True)

            # download button
            csv_data=get_csv_download_link(forecast_display)
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_data,
                file_name="Forecast.csv",
                mime="text/csv"
            )

if __name__=="__main__":
    main()