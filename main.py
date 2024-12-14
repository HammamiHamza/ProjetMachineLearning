import streamlit as st
from src.data_processing import load_and_process_data
from src.model_training import train_and_save_model, load_model
from src.utils import display_metrics
from src.model_training import train_and_save_model, load_model

st.title("Application Machine Learning avec Streamlit")

# Upload du dataset
uploaded_file = st.file_uploader("Chargez un fichier CSV :", type="csv")
if uploaded_file:
    data = load_and_process_data(uploaded_file)
    st.write("Aperçu des données :")
    st.dataframe(data.head())

    # Entraînement du modèle
    if st.button("Entraîner le modèle"):
# Entraînement et sauvegarde du modèle avec la colonne correcte "Outcome"
        model, metrics = train_and_save_model(data, target_column="Outcome", algorithm="random_forest", save_path="models/random_forest.pkl")
        st.success("Modèle entraîné et sauvegardé avec succès !")
        display_metrics(metrics)

    # Prédictions interactives
    try:
        model = load_model("models/model.pkl")
        st.subheader("Tester des prédictions")
        inputs = {col: st.number_input(f"Entrez la valeur pour {col}") for col in data.columns[:-1]}
        if st.button("Prédire"):
            prediction = model.predict([list(inputs.values())])
            st.write(f"Prédiction : {prediction[0]}")
    except FileNotFoundError:
        st.error("Modèle non trouvé. Veuillez entraîner le modèle d'abord.")
    except EOFError:
        st.error("Fichier de modèle corrompu. Réentraînez le modèle.")



