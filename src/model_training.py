import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# src/model_training.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



def train_and_save_model(data, target_column, save_path, algorithm='random_forest'):
    # Convertir l'argument 'algorithm' en minuscules
    algorithm = algorithm.lower()

    if algorithm == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif algorithm == 'logistic_regression':
        model = LogisticRegression(random_state=42)
    elif algorithm == 'svm':
        model = SVC(random_state=42)
    else:
        raise ValueError(f"Algorithm '{algorithm}' not supported.")

    # Séparer les données en caractéristiques (X) et cible (y)
    if target_column not in data.columns:
        raise ValueError(f"Column '{target_column}' not found in data.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Entraîner le modèle
    model.fit(X, y)

    # Prédictions et évaluation
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    metrics = {"accuracy": accuracy}
    

    # Sauvegarder le modèle
    joblib.dump(model, save_path)
    
    return model, metrics


def load_model(save_path):
    """
    Charge un modèle depuis un fichier en gérant les erreurs possibles.
    """
    try:
        if os.path.exists(save_path):
            return joblib.load(save_path)
        else:
            print(f"Fichier non trouvé : {save_path}")
            return None
    except (EOFError, FileNotFoundError) as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")
        return None

# Exemple d'utilisation de la fonction
data = pd.read_csv('E:/GLSI-3eme-année-s1/Machine Learning/ProjectMachine Learning/Data/Netflix_Movies_and_TV_Shows.csv')
model, metrics = train_and_save_model(data, target_column="Outcome", algorithm="random_forest", save_path="models/random_forest.pkl")

print(f"Métriques : {metrics}")










