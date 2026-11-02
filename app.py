import streamlit as st
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("data/weatherAUS.csv")
joblib = joblib.load('model/aussie_rain.joblib')
model = joblib['model']
numeric_cols = joblib['numeric_cols']
categorical_cols = joblib['categorical_cols']
encoded_cols = joblib['encoded_cols']
imputer = joblib['imputer']
scaler = joblib['scaler']
encoder = joblib['encoder']

COLUMN_CONFIG = {
    "Date": st.column_config.DateColumn(pinned=True),
    "Humidity9am": st.column_config.ProgressColumn(min_value=0, max_value=10, color="auto", format="compact", width=100),
    "Humidity3pm": st.column_config.ProgressColumn(min_value=0, max_value=100, color="auto", format="compact", width=100),
    "Pressure3pm": st.column_config.NumberColumn(format="%.2f hPa"),
    "Pressure9am": st.column_config.NumberColumn(format="%.2f hPa"),
    "Temp3pm": st.column_config.NumberColumn(format="%.1f °C"),
    "Temp9am": st.column_config.NumberColumn(format="%.1f °C"),
    "MinTemp": st.column_config.NumberColumn(format="%.1f °C"),
    "MaxTemp": st.column_config.NumberColumn(format="%.1f °C"),
    "WindSpeed3pm": st.column_config.NumberColumn(format="%.2f km/h"),
    "WindSpeed9am": st.column_config.NumberColumn(format="%.2f km/h"),
    "WindGustSpeed": st.column_config.NumberColumn(format="%.2f km/h"),
    "RainToday": st.column_config.MultiselectColumn(options=set(data["RainToday"].dropna().unique().tolist()), color="auto"),
    "RainTomorrow": st.column_config.MultiselectColumn(options=set(data["RainTomorrow"].dropna().unique().tolist()), color="auto"),
    "WindGustDir": st.column_config.MultiselectColumn(options=set(data["WindGustDir"].dropna().unique().tolist()), color="auto"),
    "WindDir3pm": st.column_config.MultiselectColumn(options=set(data["WindDir3pm"].dropna().unique().tolist()), color="auto"),
    "WindDir9am": st.column_config.MultiselectColumn(options=set(data["WindDir9am"].dropna().unique().tolist()), color="auto")
}

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)] * 100
    return pred, prob

st.sidebar.header('Forecasting rain in Australia')
st.sidebar.markdown('It is the simple model of forecasting rain in Australia for TOMORROW based on Logistic Regression')
st.sidebar.image('images/aus.png')

tab1, tab2, tab3 = st.tabs(["Analytics", "Raw Data", "Settings"])

with tab1:
    st.header("Analytics View")
    st.dataframe(data.describe().round(2))
    rainy_days = data[data['RainToday'] == 'Yes'].groupby('Location').size().reset_index()
    rainy_days.columns = ["Location", "Rainy Days"]
    st.bar_chart(rainy_days, x="Location", y="Rainy Days", color="Location")

with tab2:
    st.header("Dataset")
    st.markdown("Source/custodian: Australian Bureau of Meteorology")
    st.markdown("This dataset comprises about 10 years of daily weather observations from numerous locations across Australia. 'RainTomorrow' is the target variable to predict. It answers the crucial question: will it rain the next day? ('Yes' or 'No'). The threshold used to assign 'Yes' is if the precipitation for that day exceeded 1 mm in the 24 hours to 9 a.m. ")
    st.markdown(f"The count of rows = {data.shape[0]}")
    st.image('images/fields.png')
    for loc in data['Location'].unique().tolist():
        st.space()
        st.subheader(f"Dataset for {loc}")
        st.dataframe(data[data['Location'] == loc].drop(columns=['Location']), height=500, column_config=COLUMN_CONFIG, hide_index=True)

with tab3:
    st.header("Configurations")
    st.markdown('Please, enter the data for TODAY')
    single_input = {}
    c1, c2 = st.columns(2)
    cols = [c1, c2]
    i = 0

    am = ['Temp9am', 'WindDir9am', 'WindSpeed9am', 'Humidity9am', 'Pressure9am', 'Cloud9am']
    pm = ['Temp3pm', 'WindDir3pm', 'WindSpeed3pm', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm']
    general = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindGustDir', 'RainToday']
    
    sections = {
        "9 AM Observations": (am, False),
        "3 PM Observations": (pm, False),
    }

    single_input = {}
    
    cols = st.columns(2)
    i = 0 
    for col in general:
        target_col = cols[i % 2]

        if col in numeric_cols:
            s = data[col].dropna()
            if len(s) > 0:
                vmin, vmax = float(s.min()), float(s.max())
                dflt = float(s.median())
                with target_col:
                    single_input[col] = st.slider(label=col, min_value=vmin, max_value=vmax, value=dflt, key=f"input_{col}")
                i += 1

        elif col in categorical_cols:
            opts = sorted(data[col].dropna().unique().tolist())
            with target_col:
                if col == "RainToday":
                    single_input[col] = st.radio(label=col, options=opts, horizontal=True, key=f"input_{col}")
                else:
                    single_input[col] = st.selectbox(label=col, options=opts, key=f"input_{col}")
            i += 1
                    
    for title, (feature_list, is_expanded) in sections.items():
        with st.expander(title, expanded=is_expanded):
            cols = st.columns(2)
            i = 0 
            for col in feature_list:
                target_col = cols[i % 2]

                if col in numeric_cols:
                    s = data[col].dropna()
                    if len(s) > 0:
                        vmin, vmax = float(s.min()), float(s.max())
                        dflt = float(s.median())
                        with target_col:
                            single_input[col] = st.slider(label=col, min_value=vmin, max_value=vmax, value=dflt, key=f"input_{col}")
                        i += 1

                elif col in categorical_cols:
                    opts = sorted(data[col].dropna().unique().tolist())
                    with target_col:
                        if col == "RainToday":
                            single_input[col] = st.radio(label=col, options=opts, horizontal=True, key=f"input_{col}")
                        else:
                            single_input[col] = st.selectbox(label=col, options=opts, key=f"input_{col}")
                    i += 1  
                    
st.space()
if st.sidebar.button("Predict"):
    result, prob = predict_input(single_input)
    st.sidebar.markdown(f"Will it rain tomorrow? {result}")
    st.sidebar.markdown(f"Probability is {prob:.2f}%")
