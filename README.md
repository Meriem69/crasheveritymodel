# 🚗 Prédiction de la Gravité des Accidents Routiers

Projet MLOps complet : entraînement, tracking MLflow, API FastAPI et déploiement Docker.

---

## 📋 Description du projet

Ce projet prédit la gravité d'un accident de la route à partir de données BAAC (Bulletin d'Analyse des Accidents Corporels). Il intègre **MLflow** pour le tracking complet des expériences de machine learning.

**Variable cible — `grav` (4 classes) :**
| Valeur | Signification |
|--------|---------------|
| 1 | Indemne |
| 2 | Blessé léger |
| 3 | Blessé hospitalisé |
| 4 | Tué |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
│                                                          │
│   ┌──────────────┐         ┌──────────────────────┐     │
│   │   API        │ ──────► │   MLflow Server      │     │
│   │   FastAPI    │         │   :5000              │     │
│   │   :8000      │         │   sqlite + artefacts │     │
│   └──────────────┘         └──────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

**Flux de travail MLOps :**
```
Notebook (modeling.ipynb)
    │
    ▼
Entraînement 4 modèles ──► MLflow Tracking (runs, params, métriques, artefacts)
    │
    ▼
Tuning (manuel → GridSearch → Optuna)
    │
    ▼
Meilleur modèle ──► Model Registry (versioning)
    │
    ▼
API FastAPI ──► charge le modèle depuis Registry ──► Prédictions
```

---

## 🚀 Lancement rapide

### Prérequis
- Python 3.11+
- Docker et Docker Compose
- Git

### Option 1 — Avec Docker (recommandé)

```bash
# Cloner le projet
git clone <url-du-repo>
cd crashseveritymodel

# Lancer tous les services
docker-compose up --build

# Vérifier que tout tourne
docker-compose ps
```

Services disponibles :
- **MLflow UI** : http://localhost:5000
- **API FastAPI** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs

### Option 2 — En local (sans Docker)

```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancer le serveur MLflow
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts

# Dans un autre terminal, lancer l'API
uvicorn app.main:app --reload --port 8000
```

### Option 3 — Google Colab

Ouvrir `modeling.ipynb` dans Google Colab. Le notebook est configuré pour utiliser SQLite en local (pas besoin de serveur séparé).

---

## 📊 MLflow — Guide d'utilisation

### Lancer le serveur MLflow

```bash
# En local
mlflow server --host 127.0.0.1 --port 5000

# Avec Docker (inclus dans docker-compose)
docker-compose up mlflow
```

### Accéder à l'interface web

Ouvrir http://localhost:5000 dans un navigateur.

### Structure des expériences

| Expérience | Contenu | Étape |
|------------|---------|-------|
| `crashseveritymodel` | 4 runs (LogReg, RandomForest, XGBoost, LightGBM) | Jour 1 |
| `tuning-lightgbm` | 3 configs manuelles + run avec artefacts | Jour 2 |
| `gridsearch-lightgbm` | 8 combinaisons GridSearchCV | Jour 3 |
| `optuna-lightgbm` | 10 essais Optuna | Jour 3 |

### Ce qui est loggé dans chaque run

- **Paramètres** : `n_estimators`, `max_depth`, `learning_rate`, etc.
- **Métriques** : `accuracy`, `f1_weighted`, `precision_weighted`, `recall_weighted`
- **Artefacts** : matrice de confusion (PNG), feature names (TXT)
- **Modèle** : sérialisé et stocké dans le Registry

### Charger un modèle depuis le Registry

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # ou http://localhost:5000

# Charger la dernière version
model = mlflow.pyfunc.load_model("models:/XGBoost_accidents_model/latest")

# Faire des prédictions
predictions = model.predict(X_test_scaled)
```

### Reproduire les expériences

```bash
# Lancer le notebook complet
jupyter notebook modeling.ipynb

# Ou sur Colab : importer modeling_mlflow_complet.ipynb
```

---

## 📁 Structure du projet

```
crashseveritymodel/
│
├── modeling_mlflow_complet.ipynb  # Notebook principal (MLflow intégré)
├── docker-compose.yml             # Orchestration des services Docker
├── Dockerfile                     # Image Docker de l'API
├── requirements.txt               # Dépendances Python
├── README.md                      # Ce fichier
│
├── app/                           # Code de l'API FastAPI
│   ├── main.py                    # Point d'entrée de l'API
│   ├── model.py                   # Chargement du modèle MLflow
│   └── schemas.py                 # Schémas de données (Pydantic)
│
├── data/
│   └── df_accidents_clean.csv     # Dataset nettoyé (BAAC)
│
├── mlflow.db                      # Base MLflow (généré automatiquement)
├── mlartifacts/                   # Artefacts MLflow (généré automatiquement)
└── mlruns/                        # Runs MLflow (généré automatiquement)
```

---

## 📈 Résultats

| Modèle | Accuracy | F1 weighted |
|--------|----------|-------------|
| LogisticRegression | 44.7% | 0.468 |
| RandomForest | 57.3% | 0.582 |
| XGBoost | 59.7% | 0.538 |
| LightGBM (base) | 61.2% | 0.577 |
| LightGBM (tuning manuel) | 64.3% | - |
| LightGBM (GridSearchCV) | 64.3% | - |
| **LightGBM (Optuna)** | **64.7%** | - |

**Meilleurs paramètres trouvés par Optuna :**
```
n_estimators  : 355
max_depth     : 11
learning_rate : 0.062
```

---

## ⚙️ Variables d'environnement

| Variable | Valeur par défaut | Description |
|----------|-------------------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | URL du serveur MLflow |
| `MODEL_NAME` | `XGBoost_accidents_model` | Nom du modèle dans le Registry |
| `MODEL_VERSION` | `latest` | Version du modèle à charger |

---

## 🔧 Dépendances principales

```
mlflow>=2.0
lightgbm
xgboost
optuna
scikit-learn
pandas
numpy
fastapi
uvicorn
```

---

## 👥 Équipe

Projet réalisé dans le cadre du programme **AI & Data Science Developer — Simplon by Microsoft**

---

## 📚 Ressources

- [Documentation MLflow](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
