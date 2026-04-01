import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

plt.style.use("seaborn-v0_8")

st.set_page_config(page_title="Stock Market AI Dashboard", layout="wide")

# ---------------- LOGIN ----------------

def login():

    st.title("📈 Stock Prediction Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state["login"] = True
        else:
            st.error("Invalid username or password")


# ---------------- DASHBOARD ----------------

def dashboard():

    st.title("📊 AI Stock Market Dashboard")

    st.sidebar.header("Stock Controls")

    stocks = ["AAPL","GOOGL","MSFT","AMZN","TSLA","META","NVDA","AMD"]
    ticker = st.sidebar.selectbox("Select Stock", stocks)

    start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end = st.sidebar.date_input("End Date", datetime.date.today())

    # ✅ Fix future date issue
    if end > datetime.date.today():
        st.sidebar.warning("Future date not allowed. Reset to today.")
        end = datetime.date.today()

    show_ma100 = st.sidebar.checkbox("Show 100-Day MA", True)
    show_ma200 = st.sidebar.checkbox("Show 200-Day MA", True)

    if ticker:

        # 🔄 Load Data
        with st.spinner("Fetching stock data..."):
            df = yf.download(ticker, start=start, end=end)

        # ✅ FIX 1: Handle empty data
        if df.empty:
            st.error("❌ No data found. Try changing date range or stock.")
            return

        df = df.reset_index()

        tab1,tab2,tab3,tab4 = st.tabs([
            "📊 Data",
            "📈 Statistics",
            "📉 Charts",
            "🎓 Learn Stocks"
        ])

        # -------- DATA --------
        with tab1:
            st.dataframe(df)

        # -------- STATS --------
        with tab2:
            st.write(df.describe())

        # -------- CHARTS --------
        with tab3:

            ma100 = df.Close.rolling(100).mean()
            ma200 = df.Close.rolling(200).mean()

            fig = plt.figure(figsize=(10,4))
            plt.plot(df.Close,label="Closing Price")

            if show_ma100:
                plt.plot(ma100,label="MA100")

            if show_ma200:
                plt.plot(ma200,label="MA200")

            plt.legend()
            plt.title("Stock Trend")
            st.pyplot(fig)

            # ✅ FIX 2: Ensure enough data for LSTM
            if len(df) < 200:
                st.warning("⚠️ Not enough data for prediction (need at least 200 rows)")
                return

            try:
                # ----- LSTM -----
                data_training = df.Close[0:int(len(df)*0.7)]
                data_testing = df.Close[int(len(df)*0.7):]

                # ✅ FIX 3: Avoid empty training data
                if len(data_training) == 0:
                    st.error("Training data is empty ❌")
                    return

                scaler = MinMaxScaler()
                data_training_array = scaler.fit_transform(
                    np.array(data_training).reshape(-1,1)
                )

                # Load model
                model = load_model("keras_model.h5")

                past_100_days = data_training.tail(100)
                final_df = pd.concat([past_100_days,data_testing],ignore_index=True)

                input_data = scaler.transform(final_df.values.reshape(-1,1))

                x_test = []
                y_test = []

                for i in range(100, input_data.shape[0]):
                    x_test.append(input_data[i-100:i])
                    y_test.append(input_data[i,0])

                x_test, y_test = np.array(x_test), np.array(y_test)

                # Prediction
                y_predicted = model.predict(x_test)

                scale_factor = 1 / scaler.scale_[0]

                y_predicted = y_predicted * scale_factor
                y_test = y_test * scale_factor

                # Plot prediction
                fig2 = plt.figure(figsize=(10,4))
                plt.plot(y_test,label="Actual")
                plt.plot(y_predicted,label="Predicted")
                plt.legend()
                plt.title("Prediction vs Actual")

                st.pyplot(fig2)

                # Recommendation
                st.subheader("AI Recommendation")

                if y_predicted[-1] > y_predicted[0]*1.02:
                    st.success("📈 Uptrend → Consider Buying")

                elif y_predicted[-1] < y_predicted[0]*0.98:
                    st.error("📉 Downtrend → Risky to Buy")

                else:
                    st.warning("⚖️ Sideways → Hold")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

        # -------- LEARNING --------
        with tab4:

            st.write("### 📘 Learn Stock Market Basics")

            st.info("A stock represents ownership in a company.")
            st.info("Moving Average helps identify trends.")
            st.info("Bull Market = rising prices.")
            st.info("Bear Market = falling prices.")


# ---------------- APP FLOW ----------------

if "login" not in st.session_state:
    st.session_state["login"] = False

if st.session_state["login"]:
    dashboard()
else:
    login()