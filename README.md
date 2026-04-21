# Projet ML Retail — Analyse comportementale et prédiction du churn

## 1. Présentation du projet

Ce projet a pour objectif de construire une chaîne complète de Machine Learning appliquée à l’analyse comportementale de la clientèle dans un contexte retail / e-commerce de cadeaux.

L’idée principale est de prédire le **churn client** à partir de variables transactionnelles, temporelles, démographiques et comportementales. En complément, le projet inclut également :

- une phase de **préparation et nettoyage des données**
- une phase de **segmentation exploratoire** via PCA + KMeans
- une phase de **régression** pour prédire `MonetaryTotal`
- une phase de **déploiement** à travers une interface Flask
- un script de **prédiction réutilisable** sur de nouvelles données

Le projet suit donc une logique de bout en bout :

**Données brutes → Prétraitement → Modélisation → Évaluation → Inférence → Déploiement**

---

## 2. Objectifs

Les objectifs principaux sont les suivants :

- prédire si un client va churn ou non
- analyser les variables importantes
- réduire la dimensionnalité des données avec PCA
- explorer une segmentation client avec KMeans
- entraîner un modèle de régression sur `MonetaryTotal`
- produire une solution d’inférence réutilisable
- rendre le modèle testable via une interface web Flask

---

## 3. Structure du projet

```text
projet_ml_retail/
│
├── data/
│   ├── raw/               # Données brutes originales
│   ├── processed/         # Sorties de prédiction / données inférées
│   └── train_test/        # Fichiers X_train, X_test, y_train, y_test
│
├── notebooks/             # Réservé aux essais exploratoires (peut rester vide)
│
├── src/
│   ├── preprocessing.py   # Nettoyage, feature engineering, split, SMOTE
│   ├── train_model.py     # PCA, KMeans, classification, régression, CV
│   ├── predict.py         # Inférence sur nouvelles données
│   └── utils.py           # Fonctions utilitaires
│
├── models/                # Modèles et artefacts sauvegardés
│
├── app/
│   ├── app.py             # Backend Flask
│   ├── templates/
│   │   └── index.html     # Interface HTML
│   └── static/
│       └── css/
│           └── style.css  # Style CSS
│
├── reports/               # Figures, métriques, rapports intermédiaires
│
├── requirements.txt       # Dépendances Python
└── README.md              # Documentation du projet
4. Description des dossiers
data/raw/

Contient les données originales, non modifiées.

Exemple :

retail_customers_COMPLETE_CATEGORICAL.csv
data/train_test/

Contient les fichiers produits après prétraitement et split :

X_train.csv
X_test.csv
y_train.csv
y_test.csv
data/processed/

Contient les fichiers de sortie générés par l’inférence :

predictions_test.csv
predictions_full.csv
src/

Contient les scripts principaux du pipeline ML.

models/

Contient les artefacts sauvegardés, par exemple :

preprocessor.joblib
best_model.joblib
best_regressor.joblib
pca.joblib
kmeans.joblib
outlier_bounds.json
app/

Contient l’application Flask utilisée pour tester le modèle via interface web.

reports/

Contient les résultats et visualisations générés pendant l’entraînement :

heatmap de corrélation
PCA
KMeans
matrice de confusion
résultats CV
rapports texte / JSON / CSV
notebooks/

Dossier réservé aux notebooks exploratoires.
Dans cette version du projet, l’implémentation finale est centralisée dans src/, donc ce dossier peut rester vide.

5. Pipeline du projet
Étape 1 — Prétraitement (preprocessing.py)

Le script :

charge les données brutes
supprime les colonnes inutiles ou à risque de fuite
parse les dates
crée les features
transforme l’adresse IP en variables exploitables
gère les outliers avec IQR
supprime certaines variables redondantes
split train/test
applique SMOTE / SMOTENC sur le train
sauvegarde les jeux train_test
sauvegarde le préprocesseur
Étape 2 — Entraînement (train_model.py)

Le script :

charge X_train, X_test, y_train, y_test
génère heatmap et VIF
applique une PCA
exécute KMeans
entraîne plusieurs modèles de classification
optimise certains hyperparamètres avec GridSearchCV et Optuna
entraîne des modèles de régression
exécute une validation croisée
sauvegarde le meilleur modèle de classification
sauvegarde le meilleur modèle de régression
Étape 3 — Inférence (predict.py)

Le script :

recharge le préprocesseur et le modèle
applique le même pipeline de préparation qu’en entraînement
aligne les colonnes attendues
effectue la prédiction
sauvegarde les sorties dans un fichier CSV
Étape 4 — Déploiement (app/app.py)

L’application Flask permet de :

tester manuellement un profil client
ou charger une ligne réelle de X_test
afficher la classe prédite
afficher la probabilité de churn
6. Prétraitement appliqué

Le prétraitement final inclut :

suppression des colonnes inutiles
suppression de colonnes à risque de leakage
parsing de RegistrationDate
feature engineering sur LastLoginIP
création de variables dérivées
clipping IQR sur certaines colonnes numériques
imputation :
médiane pour les variables numériques
mode pour les variables catégorielles
KNNImputer pour Age
normalisation via StandardScaler
encodage catégoriel via OneHotEncoder
rééquilibrage des classes avec SMOTENC
7. Variables supprimées

Certaines variables ont été supprimées pour améliorer la qualité du pipeline, notamment :

Colonnes inutiles ou identifiants
CustomerID
NewsletterSubscribed
Colonnes à risque de fuite
ChurnRiskCategory
colonnes contenant churnrisk
Recency
TenureRatio
MonetaryPerDay
CustomerType
RFMSegment
Colonnes redondantes
CancelledTransactions
UniqueInvoices
UniqueDescriptions
AvgLinesPerInvoice
MonetaryMin
MonetaryMax
MinQuantity
8. Modèles utilisés
Classification

Modèles testés :

Logistic Regression
SVM
Random Forest

Optimisation appliquée :

GridSearchCV sur Logistic Regression
GridSearchCV sur SVM
GridSearchCV sur Random Forest
Optuna sur Logistic Regression
Régression

Modèles testés :

Ridge
RandomForestRegressor
Clustering
KMeans appliqué sur les données transformées par PCA
Réduction de dimension
PCA automatique avec seuil de variance expliquée
9. Résultats finaux
Classification

Le meilleur modèle final est une Logistic Regression optimisée, avec des performances très élevées sur la prédiction du churn.

Exemple de résultats finaux :

Accuracy ≈ 0.994
F1-score ≈ 0.991
ROC-AUC ≈ 0.999
Régression

Le meilleur modèle de régression est :

RandomForestRegressor

Exemple de performance :

R² ≈ 0.769
Validation croisée

La validation croisée a montré une bonne stabilité des modèles, surtout en classification.

10. KMeans et PCA

L’analyse non supervisée a été utilisée comme étape exploratoire.

PCA a permis de réduire la dimension tout en conservant l’essentiel de l’information
KMeans a été utilisé pour tester différents nombres de clusters
dans la version finale retenue, le meilleur k obtenu par le code final est k = 2

La partie clustering reste exploratoire et secondaire par rapport à la classification supervisée.

11. Déploiement Flask

L’application Flask propose deux modes :

Mode manuel

L’utilisateur saisit un sous-ensemble de variables importantes :

Frequency
MonetaryTotal
CustomerTenureDays
UniqueProducts
ReturnRatio
Age
SupportTicketsCount
SatisfactionScore
Gender
Country
Mode dataset

L’utilisateur renseigne directement l’index d’une ligne de X_test.csv, ce qui permet de tester une observation réelle complète.

Remarque importante

Le mode manuel reste une version simplifiée.
Le modèle complet attend davantage de variables. Donc :

le mode dataset est le plus fidèle au modèle réel
le mode manuel est surtout destiné à la démonstration
12. Commandes principales
1. Prétraitement
python src/preprocessing.py
2. Entraînement
python src/train_model.py
3. Prédiction sur nouvelles données
python src/predict.py --input data/raw/retail_customers_COMPLETE_CATEGORICAL.csv --output data/processed/predictions_full.csv
4. Lancer l’application Flask
python app/app.py

Puis ouvrir dans le navigateur :

http://127.0.0.1:5000
13. Dépendances

Le projet utilise principalement :

Python
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
joblib
flask
optuna
statsmodels

Installation possible via :

pip install -r requirements.txt
14. Points forts du projet
pipeline ML complet
bon découpage du code
preprocessing cohérent
gestion du déséquilibre avec SMOTENC
comparaison de plusieurs modèles
tuning avancé
validation croisée
script d’inférence réutilisable
déploiement Flask
15. Limites du projet
performances très élevées à interpréter avec prudence
partie clustering surtout exploratoire
écart entre modèle complet et formulaire simplifié Flask
interface Flask non encore totalement alignée sur toutes les variables du modèle
16. Pistes d’amélioration
enrichir l’interface Flask
créer un modèle spécifique pour le mode manuel
ajouter une explication locale des prédictions
intégrer SHAP
tester sur un autre dataset
améliorer la calibration probabiliste
ajouter une API REST complète
17. Auteur

Yassine Meftah

Projet réalisé dans le cadre de l’atelier Machine Learning 2025–2026.