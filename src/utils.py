import streamlit as st

def display_metrics(metrics):
    st.write("**Évaluation du modèle :**")
    for metric, value in metrics.items():
        st.write(f"- {metric.capitalize()} : {value:.2f}")




