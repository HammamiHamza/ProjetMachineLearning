import pandas as pd

def load_and_process_data(uploaded_file):
    # Charger le fichier CSV en dataframe
    data = pd.read_csv(uploaded_file)

    # Vérification basique des données
    if data.isnull().sum().any():
        data = data.fillna(data.mean())  # Remplir les valeurs manquantes avec la moyenne
    return data




