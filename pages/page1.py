import streamlit as st
import pandas as pd

st.title("Exploration des Données")
uploaded_file = st.file_uploader("Data/netflix.csv", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data)
    st.write("Statistiques descriptives :")
    st.write(data.describe())
