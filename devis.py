import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from ta import add_all_ta_features
from ta.utils import dropna

# Fonction pour télécharger les données
def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Fonction pour calculer les indicateurs techniques
def calculate_technical_indicators(data):
    data = dropna(data)
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    return data

# Fonction pour préparer les données pour la classification
def prepare_data_classification(data):
    features = data.drop(columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"])
    target = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)  # 1: hausse, 0: baisse
    return features, target

# Fonction pour préparer les données pour la régression
def prepare_data_regression(data):
    features = data.drop(columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"])
    target = data["Close"].shift(-1)
    features = features[:-1]
    target = target[:-1]
    return features, target

# Fonction pour afficher les graphiques
def plot_data(data):
    st.line_chart(data['Close'])

# Fonction pour appliquer PCA
def apply_pca(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    return principal_components

# Fonction pour détecter les opportunités de trading
def detect_trading_opportunities(data, model):
    scaler = StandardScaler()
    features, target = prepare_data_classification(data)
    features_scaled = scaler.fit_transform(features)
    principal_components = apply_pca(features)
    
    model.fit(principal_components, target)
    predictions = model.predict(principal_components)
    
    data['Prediction'] = np.nan
    data['Prediction'][-len(predictions):] = predictions
    
    buy_signals = data[(data['Prediction'] == 1) & (data['Prediction'].shift(1) == 0)]
    sell_signals = data[(data['Prediction'] == 0) & (data['Prediction'].shift(1) == 1)]
    
    return buy_signals, sell_signals

# Fonction pour prédire les prix futurs
def predict_future_prices(data, model):
    features, target = prepare_data_regression(data)
    model.fit(features, target)
    future_predictions = model.predict(features[-5:])
    return future_predictions

# Fonction principale
def main():
    st.set_page_config(page_title="Plateforme de Trading", layout="wide")
    
    st.title("Plateforme de Détection des Opportunités de Trading")

    # Sidebar pour les paramètres
    st.sidebar.header("Paramètres de l'analyse")
    ticker = st.sidebar.text_input("Entrer le symbole de l'action (ex. AAPL)", value='AAPL')
    start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("2023-01-01"))
    model_choice = st.sidebar.selectbox("Choisissez le modèle de machine learning", ["LDA", "SVM", "Logistic Regression", "KNN", "Decision Tree"])
    
    if st.sidebar.button("Télécharger les données"):
        data = download_data(ticker, start=start_date, end=end_date)
        
        # Afficher les données brutes
        st.subheader("Données Brutes")
        st.dataframe(data.head())

        # Calculer et afficher les indicateurs techniques
        data_ta = calculate_technical_indicators(data)
        st.subheader("Données avec Indicateurs Techniques")
        st.dataframe(data_ta.head())

        # Préparer les données
        features, target = prepare_data_classification(data_ta)

        # Afficher le graphique des prix de clôture
        st.subheader("Graphique des Prix de Clôture")
        plot_data(data)

        # Appliquer PCA et afficher les composantes principales
        principal_components = apply_pca(features)
        st.subheader("Composantes Principales")
        st.write(principal_components[:5])

        # Choisir le modèle de machine learning
        models = {
            "LDA": LDA(),
            "SVM": SVC(),
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
        model = models[model_choice]

        # Détection des opportunités de trading
        buy_signals, sell_signals = detect_trading_opportunities(data_ta, model)
        
        st.subheader("Opportunités de Trading")
        st.write("Signaux d'achat")
        st.dataframe(buy_signals[['Close']])
        st.write("Signaux de vente")
        st.dataframe(sell_signals[['Close']])
        
        # Afficher les signaux d'achat et de vente sur le graphique
        st.subheader("Graphique des Signaux de Trading")
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Close'], label='Prix de Clôture', color='blue')
        ax.scatter(buy_signals.index, buy_signals['Close'], label='Acheter', marker='^', color='green')
        ax.scatter(sell_signals.index, sell_signals['Close'], label='Vendre', marker='v', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix')
        ax.legend()
        st.pyplot(fig)

        # Prédiction des prix futurs
        st.subheader("Prédiction des Prix Futurs")
        regression_model = RandomForestRegressor()
        future_prices = predict_future_prices(data_ta, regression_model)
        st.write("Prix prédits pour les 5 prochains jours:", future_prices)

if __name__ == "__main__":
    main()
