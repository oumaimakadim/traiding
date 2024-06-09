import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import yfinance as yf


# Fonction pour télécharger les données financières avec yfinance
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data


# Fonction pour préparer les données
def prepare_data(data):
    data = data.dropna()
    if data.empty:
        raise ValueError("Les données téléchargées sont vides. Vérifiez les tickers et la période sélectionnée.")
    data['return'] = data.pct_change().mean(axis=1)
    data['volatility'] = data.pct_change().std(axis=1)
    data['volume'] = data.mean(axis=1)  # Volume moyen simulé pour cet exemple
    data['trend'] = (data['return'] > 0).astype(int)
    data['target'] = (data['return'] > data['return'].mean()).astype(int)  # Créer une cible binaire pour rendement

    # Convertir les valeurs continues en classes discrètes
    data['volatility_class'] = pd.qcut(data['volatility'], q=3, labels=['Low', 'Medium', 'High'])
    data['volume_class'] = pd.qcut(data['volume'], q=3, labels=['Low', 'Medium', 'High'])

    return data


# Définir le modèle de prédiction avec PyTorch Lightning
class PerformancePredictor(pl.LightningModule):
    def __init__(self, input_dim):
        super(PerformancePredictor, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        return self.output(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Télécharger les données
st.title('Analyse et Prédiction des Performances des Actions Internationales')
tickers = st.text_input('Entrer les symboles des actions (séparés par des espaces)', 'AAPL GOOGL MSFT AMZN')
tickers = tickers.split()
start_date = st.date_input('Date de début', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('Date de fin', value=pd.to_datetime('2022-01-01'))

if st.button('Charger les données'):
    try:
        data = load_data(tickers, start=start_date, end=end_date)
        if data.empty:
            st.error("Les données téléchargées sont vides. Vérifiez les tickers et la période sélectionnée.")
        else:
            data = prepare_data(data)
            st.write(data)

            # Encoder les classes catégorielles
            le_volatility = LabelEncoder()
            data['volatility_class_encoded'] = le_volatility.fit_transform(data['volatility_class'])

            le_volume = LabelEncoder()
            data['volume_class_encoded'] = le_volume.fit_transform(data['volume_class'])

            # Réduction de Dimensionnalité avec PCA
            X = data.drop(
                columns=['target', 'trend', 'volatility', 'volume', 'return', 'volatility_class', 'volume_class'])
            y = data['target']
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            tsne = TSNE(n_components=2)
            X_tsne = tsne.fit_transform(X)

            # Calculer dynamiquement le nombre de composantes pour LDA
            n_classes = len(np.unique(y))
            n_features = X.shape[1]
            n_components_lda = min(n_features, n_classes - 1)
            lda = LDA(n_components=n_components_lda)
            X_lda = lda.fit_transform(X, y)

            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Classification de la Volatilité avec Decision Tree
            tree = DecisionTreeClassifier()
            tree.fit(X_train, data['volatility_class_encoded'].loc[X_train.index])
            y_pred_tree = tree.predict(X_test)

            # Classification du Rendement avec SVM
            svm = SVC()
            svm.fit(X_train, y_train)
            y_pred_svm = svm.predict(X_test)

            # Classification du Volume avec KNN
            knn = KNeighborsClassifier()
            knn.fit(X_train, data['volume_class_encoded'].loc[X_train.index])
            y_pred_knn = knn.predict(X_test)

            # Classification de la Tendance avec Logistic Regression
            log_reg = LogisticRegression(max_iter=1000)
            log_reg.fit(X_train, data['trend'].loc[X_train.index])
            y_pred_log_reg = log_reg.predict(X_test)

            # Prédiction des Performances Futures avec PyTorch Lightning
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            y_tensor = torch.tensor(y.values, dtype=torch.long)
            dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

            model = PerformancePredictor(input_dim=X.shape[1])
            trainer = pl.Trainer(max_epochs=20)
            trainer.fit(model, train_loader)

            # Affichage des données sous forme de graphiques
            st.header("Visualisations")
            st.subheader('Réduction de Dimensionnalité avec PCA')
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)
            st.pyplot(fig)

            st.subheader('Exploration des Données avec t-SNE')
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)
            st.pyplot(fig)

            st.subheader('Séparation des Classes avec LDA')
            fig, ax = plt.subplots()
            if n_components_lda > 1:
                scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
            else:
                scatter = ax.scatter(X_lda[:, 0], np.zeros_like(X_lda[:, 0]), c=y, cmap='viridis')
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)
            st.pyplot(fig)

            # Explications des modèles
            # Classification de la Volatilité avec Decision Tree
            st.subheader("Classification de la Volatilité avec Decision Tree")
            st.write("Le modèle Decision Tree est utilisé pour classer les niveaux de volatilité des actions.")
            st.text(classification_report(data['volatility_class_encoded'].loc[X_test.index], y_pred_tree))

            # Classification du Rendement avec SVM
            st.subheader("Classification du Rendement avec SVM")
            st.write(
                "Le modèle SVM (Support Vector Machine) est utilisé pour prédire la performance future des actions.")
            st.text(classification_report(y_test, y_pred_svm))

            # Classification du Volume avec KNN
            st.subheader("Classification du Volume avec KNN")
            st.write("Le modèle KNN (K-Nearest Neighbors) est utilisé pour classer les volumes des actions.")
            st.text(classification_report(data['volume_class_encoded'].loc[X_test.index], y_pred_knn))

            # Classification de la Tendance avec Logistic Regression
            st.subheader("Classification de la Tendance avec Logistic Regression")
            st.write("Le modèle Logistic Regression est utilisé pour prédire la tendance future des actions.")
            st.text(classification_report(data['trend'].loc[X_test.index], y_pred_log_reg))

            # Prédiction des Performances Futures avec PyTorch Lightning
            st.subheader("Prédiction des Performances Futures avec PyTorch Lightning")
            st.write("Le modèle PyTorch Lightning est utilisé pour prédire les performances futures des actions.")

            # Prédictions avec le modèle PyTorch Lightning
            predictions = model.forward(X_tensor).argmax(dim=1)

            # Ajouter les prédictions au DataFrame
            data['predictions'] = predictions.numpy()

            # Afficher les prédictions
            st.write("Prédictions :")
            st.write(data[['return', 'predictions']].head(10))

            # Recommandations personnalisées
            st.subheader("Recommandations personnalisées pour l'optimisation du portefeuille...")

            # Exemple de recommandations basées sur les classifications et prédictions
            recommandations = []

            # Classification de la Volatilité avec Decision Tree
            if any(y_pred_tree):
                recommandations.append("La classification de la volatilité avec Decision Tree indique des variations importantes. \
                Il pourrait être prudent de surveiller de près ces actions et de prendre des décisions en conséquence.")

            # Classification du Rendement avec SVM
            if any(y_pred_svm):
                recommandations.append("La classification du rendement avec SVM montre des perspectives positives pour certaines actions. \
                Il peut être intéressant d'investir davantage dans ces actions pour obtenir des rendements plus élevés.")

            # Classification du Volume avec KNN
            if any(y_pred_knn):
                recommandations.append("La classification du volume avec KNN indique une forte activité sur certaines actions. \
                Il pourrait être avantageux d'examiner de plus près ces actions pour profiter des mouvements du marché.")

            # Classification de la Tendance avec Logistic Regression
            if any(y_pred_log_reg):
                recommandations.append("La classification de la tendance avec Logistic Regression suggère des tendances claires sur certains marchés. \
                Il peut être judicieux d'aligner les stratégies d'investissement en fonction de ces tendances.")

            # Prédiction des Performances Futures avec PyTorch Lightning
            if any(predictions):
                recommandations.append("Les prédictions futures avec PyTorch Lightning indiquent des performances potentielles pour les actions. \
                Il serait judicieux de prendre en compte ces prédictions lors de la planification d'investissements à long terme.")

            # Affichage des recommandations
            for recommandation in recommandations:
                st.write(recommandation)

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
