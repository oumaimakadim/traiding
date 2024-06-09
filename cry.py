import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, mean_squared_error
from ta import add_all_ta_features
from ta.utils import dropna
import pywt
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

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

# Lightning module for classification with LLM
class LightningLLMClassifier(pl.LightningModule):
    def __init__(self, model_name, num_labels):
        super(LightningLLMClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def training_step(self, batch, batch_idx):
        inputs = self.tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")
        labels = batch['label']
        outputs = self.forward(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
        loss = outputs.loss
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

# Fonction principale
def main():
    st.set_page_config(page_title="Plateforme de Trading", layout="wide")
    
    st.title("Plateforme de Détection des Opportunités de Trading et Prédiction des Prix")

    # Documentation
    st.markdown("""
    ## Introduction
    Bienvenue sur la plateforme de détection des opportunités de trading et de prédiction des prix ! Cet outil est conçu pour vous aider à analyser les données du marché boursier, à détecter des opportunités de trading potentielles et à prédire les prix futurs des actions à l'aide de modèles avancés de machine learning. Décomposons chaque composant pour vous aider à prendre des décisions d'investissement éclairées.

    ## Analyse des Données et Indicateurs Techniques
    La plateforme télécharge les données historiques des actions pour le symbole de ticker de votre choix à partir de Yahoo Finance. Les données incluent des métriques telles que les prix d'ouverture, de clôture, les plus hauts et les plus bas, le volume, et les prix ajustés de clôture.

    **Indicateurs Techniques :**
    - **Moyennes Mobiles (MA) :** Moyenne les prix de clôture sur une période spécifiée pour lisser les données de prix et identifier les tendances.
    - **Indice de Force Relative (RSI) :** Mesure la vitesse et le changement des mouvements de prix pour identifier les conditions de surachat ou de survente.
    - **Convergence-Divergence des Moyennes Mobiles (MACD) :** Montre la relation entre deux moyennes mobiles pour indiquer des signaux d'achat ou de vente.
    - **Bandes de Bollinger :** Composées d'une bande centrale (moyenne mobile simple) et de bandes supérieure et inférieure (écarts types) pour indiquer la volatilité.

    ## Réduction de Dimensionnalité
    Pour simplifier les données et visualiser leurs principales caractéristiques, nous utilisons :
    - **Analyse en Composantes Principales (PCA) :** Réduit les données à deux composantes principales, capturant la plus grande variance dans les données.
    - **t-Distributed Stochastic Neighbor Embedding (t-SNE) :** Réduit les données à deux dimensions, préservant la structure locale et les relations entre les points.

    ## Modèles de Classification
    Les modèles de classification prédisent si le prix de l'action va augmenter ou diminuer en fonction des données historiques. Choisissez parmi plusieurs modèles :
    - **Analyse Discriminante Linéaire (LDA) :** Projette les données dans un espace de dimension inférieure tout en maximisant la séparabilité des classes.
    - **Machines à Vecteurs de Support (SVM) :** Trouve l'hyperplan optimal qui sépare différentes classes dans les données.
    - **Régression Logistique :** Modélise la probabilité d'un résultat binaire en fonction d'une ou plusieurs variables prédictives.
    - **K-Nearest Neighbors (KNN) :** Classe un point de données en fonction de la majorité de la classe parmi ses voisins les plus proches.
    - **Arbre de Décision :** Divise les données en branches pour prédire la variable cible.
    - **Perceptron :** Un modèle de réseau neuronal simple qui peut classer les points de données en classes binaires.
    - **Lightning LLM :** Utilise un grand modèle de langage (par exemple, BERT) pour des prédictions plus avancées et contextuelles.

    ## Signaux de Trading
    Après avoir ajusté le modèle de classification, la plateforme détecte les opportunités de trading potentielles :
    - **Signaux d'Achat :** Indiquent quand acheter l'action, basés sur des critères spécifiques (par exemple, le prix de clôture est supérieur au prix d'ouverture).
    - **Signaux de Vente :** Indiquent quand vendre l'action, basés sur des critères spécifiques (par exemple, le prix de clôture est inférieur au prix d'ouverture).

    ## Prédiction des Prix
    Pour prédire les prix futurs des actions, la plateforme utilise des modèles de régression. Vous pouvez choisir parmi :
    - **Régression à Vecteurs de Support (SVR) :** Étend SVM pour prédire des valeurs continues.
    - **Régression Linéaire :** Modélise la relation entre la cible et les variables prédictives comme une fonction linéaire.
    - **Régression par Arbre de Décision :** Divise les données en branches pour prédire des valeurs continues.

    La plateforme applique également la Transformée en Ondelettes pour extraire des caractéristiques supplémentaires des données de séries temporelles.

    ## Visualisation
    La plateforme visualise les prix historiques des actions, les indicateurs techniques et les prix futurs prédits, vous aidant à comprendre les tendances et à prendre des décisions d'investissement éclairées.

    En utilisant ces outils analytiques avancés, vous pouvez obtenir des informations sur les tendances du marché boursier, identifier des opportunités de trading potentielles et prendre des décisions d'investissement plus éclairées.
    """)

    # Sidebar pour les paramètres
    st.sidebar.header("Paramètres de l'analyse")
    asset_type = st.sidebar.selectbox("Choisissez le type d'actif", ["Action", "Cryptomonnaie"])
    if asset_type == "Action":
        ticker = st.sidebar.selectbox("Choisissez le symbole de l'action", ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA", "FB", "NVDA", "NFLX", "BRK-A", "V"])
    else:
        ticker = st.sidebar.selectbox("Choisissez le symbole de la cryptomonnaie", ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD", "DOT1-USD", "LINK-USD", "XLM-USD", "DOGE-USD"])

    start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("2023-01-01"))
    model_choice = st.sidebar.selectbox("Choisissez le modèle de machine learning pour la classification", ["LDA", "SVM", "Logistic Regression", "KNN", "Decision Tree", "Perceptron", "Lightning LLM"])
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
            "Decision Tree": DecisionTreeClassifier(),
            "Perceptron": Perceptron()
        }
        
        if model_choice != "Lightning LLM":
            model = models[model_choice]
            model.fit(pca_components, target)
            buy_signals, sell_signals = detect_trading_opportunities(data_ta, model, pca_components, target)
            st.write(classification_report(target, model.predict(pca_components)))
        else:
            # Prepare DataLoader for Lightning
            # Create dummy text data and labels for illustration
            texts = ["example text 1"] * len(features)  # Replace with actual text data
            labels = target.tolist()
            dataset = [{'text': text, 'label': label} for text, label in zip(texts, labels)]
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Initialize and train Lightning model
            model = LightningLLMClassifier(model_name='bert-base-uncased', num_labels=2)
            trainer = pl.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)
            trainer.fit(model, train_loader)
            
            st.write("Lightning LLM model trained successfully.")
        
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
