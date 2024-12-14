import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Charger le dataset
#DATA_PATH = 'E:/GLSI-3eme-année-s1/Machine Learning/ProjectMachine Learning/Data/Netflix_Movies_and_TV_Shows.csv'
#df = pd.read_csv('Netflix_Movies_and_TV_Shows.csv')
df = pd.read_csv('Data/Netflix_Movies_and_TV_Shows.csv')

# Nettoyer les données
df['duration_numeric'] = df['Duration'].str.extract('(\d+)').astype(float)  # Convertir la durée en numérique
df.dropna(subset=['Type'], inplace=True)  # Supprimer les valeurs nulles dans 'type'

# Titre principal
st.title("Tableau de bord interactif : Netflix Movies & TV Shows")

# Sidebar pour les filtres
st.sidebar.header("Filtres")
type_filter = st.sidebar.multiselect(
    "Filtrer par type de contenu",
    options=df['Type'].unique(),
    default=df['Type'].unique()
)

year_filter = st.sidebar.slider(
    "Filtrer par année de sortie",
    int(df['Release Year'].min()),
    int(df['Release Year'].max()),
    (2000, 2020)
)

# Appliquer les filtres
filtered_df = df[(df['Type'].isin(type_filter)) & (df['Release Year'].between(*year_filter))]

# Affichage des données filtrées
st.write("### Aperçu des données filtrées")
st.dataframe(filtered_df)

# Graphique 1 : Répartition des types de contenu
st.write("### Répartition des types de contenu")
type_counts = filtered_df['Type'].value_counts()
st.bar_chart(type_counts)

# Graphique 2 : Durée moyenne par année
st.write("### Durée moyenne des contenus par année")
avg_duration = filtered_df.groupby('Release Year')['duration_numeric'].mean().dropna()
st.line_chart(avg_duration)

# Graphique 3 : Relation entre année et durée
st.write("### Relation entre année et durée")
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['Release Year'], filtered_df['duration_numeric'], alpha=0.5, color='blue')
plt.title("Relation entre année de sortie et durée")
plt.xlabel("Année de sortie")
plt.ylabel("Durée (minutes)")
st.pyplot(plt)
