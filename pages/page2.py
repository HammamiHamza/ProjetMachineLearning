import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_training import load_model



import streamlit as st
from src.model_training import load_model

st.title("Tester des Prédictions")

# Chargement du modèle avec gestion des erreurs
model = load_model("models/model.pkl")

if model:
    st.success("Modèle chargé avec succès !")

    # Saisie des features par l'utilisateur
    inputs = {
        f"Feature {i}": st.number_input(f"Entrez la valeur pour Feature {i}", value=0.0)
        for i in range(8)
    }

    # Prédiction
    if st.button("Prédire"):
        try:
            prediction = model.predict([list(inputs.values())])
            st.write(f"Prédiction : {prediction[0]}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
else:
    st.error("Modèle non trouvé ou non chargé. Veuillez entraîner un modèle d'abord.")
