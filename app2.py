import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, mean_squared_error
from ta import add_all_ta_features
from ta.utils import dropna
import pywt

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
    target = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    features, target = features.iloc[:-1, :], target[:-1]  # Align features with target
    return features, target

# Fonction pour préparer les données pour la régression
def prepare_data_regression(data):
    features = data.drop(columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"])
    target = data["Close"].shift(-1).dropna()
    features = features.iloc[:-1, :]  # Align features with target
    return features, target

# Fonction pour appliquer PCA
def apply_pca(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_features)
    return pca_components

# Fonction pour appliquer t-SNE
def apply_tsne(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_components = tsne.fit_transform(scaled_features)
    return tsne_components

# Fonction pour appliquer la transformée en ondelettes
def apply_wavelet_transform(data):
    wavelet = 'db1'
    coeffs = pywt.wavedec(data['Close'], wavelet, level=3)
    coeffs_flattened = np.concatenate(coeffs)
    return coeffs_flattened

# Fonction pour prédire les prix futurs
def predict_future_prices(features, target, model):
    model.fit(features, target)
    future_predictions = model.predict(features[-5:])
    return future_predictions

# Fonction pour détecter les opportunités de trading
def detect_trading_opportunities(data, model, features, target):
    model.fit(features, target)
    buy_signals = data[(data['Close'] > data['Open']) & (data['Close'] > data['Close'].shift(1))]
    sell_signals = data[(data['Close'] < data['Open']) & (data['Close'] < data['Close'].shift(1))]
    return buy_signals, sell_signals

# Fonction principale
def main():
    st.set_page_config(page_title="Plateforme de Trading", layout="wide")
    
    st.title("Plateforme de Détection des Opportunités de Trading et Prédiction des Prix")

    # Sidebar pour les paramètres
    st.sidebar.header("Paramètres de l'analyse")
    ticker = st.sidebar.text_input("Entrer le symbole de l'action (ex. AAPL)", value='AAPL')
    start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("2023-01-01"))
    model_choice = st.sidebar.selectbox("Choisissez le modèle de machine learning pour la classification", ["LDA", "SVM", "Logistic Regression", "KNN", "Decision Tree"])
    regression_model_choice = st.sidebar.selectbox("Choisissez le modèle de régression", ["SVR", "Linear Regression", "Decision Tree Regressor"])
    
    if st.sidebar.button("Télécharger les données"):
        data = download_data(ticker, start=start_date, end=end_date)
        
        # Afficher les données brutes
        st.subheader("Données Brutes")
        st.dataframe(data.head())

        # Calculer et afficher les indicateurs techniques
        data_ta = calculate_technical_indicators(data)
        st.subheader("Données avec Indicateurs Techniques")
        st.dataframe(data_ta.head())

        # Préparer les données pour la classification
        features, target = prepare_data_classification(data_ta)

        # Afficher le graphique des prix de clôture
        st.subheader("Graphique des Prix de Clôture")
        st.line_chart(data['Close'])

        # Appliquer PCA et t-SNE et afficher les composantes principales
        pca_components = apply_pca(features)
        tsne_components = apply_tsne(features)
        st.subheader("Composantes Principales (PCA)")
        st.write(pca_components[:5])
        st.subheader("Composantes t-SNE")
        st.write(tsne_components[:5])

        # Choisir le modèle de machine learning pour la classification
        models = {
            "LDA": LDA(),
            "SVM": SVC(),
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
        model = models[model_choice]

        # Fitting the model before prediction
        model.fit(pca_components, target)
        
        buy_signals, sell_signals = detect_trading_opportunities(data_ta, model, pca_components, target)

        st.write(classification_report(target, model.predict(pca_components)))

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

        # Préparer les données pour la régression
        reg_features, reg_target = prepare_data_regression(data_ta)

        # Appliquer la transformée en ondelettes
        wavelet_features = apply_wavelet_transform(data_ta)
        wavelet_features = wavelet_features.reshape(-1, 1)  # reshape pour s'adapter à l'entrée du modèle

        # Choisir le modèle de régression
        regression_models = {
            "SVR": SVR(),
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }
        regression_model = regression_models[regression_model_choice]

        # Prédire les prix futurs
        future_prices = predict_future_prices(wavelet_features[:len(reg_target)], reg_target, regression_model)
        
        st.subheader("Prédiction des Prix Futurs")
        st.write("Prix prédits pour les 5 prochains jours:", future_prices)
        
        # Afficher les prédictions des prix futurs sur le graphique
        st.subheader("Graphique des Prix avec Prédictions")
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Close'], label='Prix de Clôture', color='blue')
        future_index = pd.date_range(start=data.index[-1], periods=6)
        ax.plot(future_index, np.concatenate([[data['Close'].iloc[-1]], future_prices]), label='Prédictions', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
