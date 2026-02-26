from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import numpy as np
import uvicorn
import os
import mlflow
import mlflow.sklearn

# =====================
# BASE DE DONNÉES
# =====================
import psycopg2
from psycopg2.extras import RealDictCursor


def get_db_connection():
    """Connexion à PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "db"),
            database=os.getenv("DB_NAME", "accidents_db"),
            user=os.getenv("DB_USER", "admin"),
            password=os.getenv("DB_PASSWORD", "password123"),
        )
        return conn
    except Exception as e:
        print(f"Erreur connexion BDD: {e}")
        return None


def init_db():
    """Créer la table historique si elle n'existe pas"""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS historique (
                    id SERIAL PRIMARY KEY,
                    date_prediction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    age INTEGER,
                    sexe VARCHAR(10),
                    vehicule VARCHAR(50),
                    meteo VARCHAR(50),
                    luminosite VARCHAR(50),
                    type_route VARCHAR(50),
                    resultat VARCHAR(20),
                    probabilite FLOAT
                )
            """)
            conn.commit()
            cur.close()
            conn.close()
            print("✅ Table historique créée/vérifiée")
        except Exception as e:
            print(f"Erreur création table: {e}")


def save_prediction(age, sexe, vehicule, meteo, luminosite, type_route, resultat, probabilite):
    """Sauvegarder une prédiction dans la BDD"""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO historique (age, sexe, vehicule, meteo, luminosite, type_route, resultat, probabilite)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (age, sexe, vehicule, meteo, luminosite, type_route, resultat, probabilite),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Erreur sauvegarde: {e}")


def get_historique():
    """Récupérer l'historique des prédictions"""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT * FROM historique ORDER BY date_prediction DESC LIMIT 50")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return rows
        except Exception as e:
            print(f"Erreur lecture historique: {e}")
            return []
    return []


# =====================
# MLFLOW — Chargement du modèle
# =====================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Charger le modèle depuis le fichier pkl (fallback si MLflow pas dispo)
try:
    model = mlflow.pyfunc.load_model("models:/XGBoost_accidents_model/latest")
    print("✅ Modèle chargé depuis MLflow Registry")
except Exception as e:
    print(f"⚠️ MLflow Registry non disponible, chargement depuis pkl : {e}")
    model = joblib.load("modele_final.pkl")
    print("✅ Modèle chargé depuis modele_final.pkl")

# =====================
# APPLICATION FASTAPI
# =====================

app = FastAPI(title="API Prédiction Accidents", version="2.0")

templates = Jinja2Templates(directory="templates")

SEXE_MAP = {1: "Homme", 2: "Femme"}
VEHICULE_MAP = {1: "Vélo", 2: "Cyclomoteur", 7: "Voiture", 13: "Poids lourd", 31: "Moto", 50: "Trottinette"}
METEO_MAP = {1: "Normale", 2: "Pluie légère", 3: "Pluie forte", 4: "Neige"}
LUM_MAP = {1: "Jour", 2: "Crépuscule", 3: "Nuit sans éclairage", 4: "Nuit avec éclairage"}
ROUTE_MAP = {1: "Autoroute", 2: "Nationale", 3: "Départementale", 4: "Communale"}


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True, "version": "2.0", "mlflow_uri": MLFLOW_TRACKING_URI}


@app.get("/historique", response_class=HTMLResponse)
async def historique(request: Request):
    data = get_historique()
    return templates.TemplateResponse("historique.html", {"request": request, "historique": data})


class AccidentData(BaseModel):
    lum: int
    agg: int
    int_: int = Field(alias='int')
    atm: int
    col: int
    catr: int
    catv: int
    heure: int
    jour_semaine: int
    weekend: int
    sexe: int
    age: int
    secu1: int
    terre_plein: int


@app.post("/predict")
def predict(data: AccidentData):
    input_data = np.array([[
        data.lum, data.agg, data.int_, data.atm, data.col,
        data.catr, data.catv, data.heure, data.jour_semaine,
        data.weekend, data.sexe, data.age, data.secu1, data.terre_plein
    ]])

    # Logger la prédiction dans MLflow
    with mlflow.start_run(run_name="prediction_api"):
        mlflow.log_params({
            "lum": data.lum, "atm": data.atm, "catv": data.catv,
            "sexe": data.sexe, "age": data.age
        })

        if hasattr(model, "predict_proba"):
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]
        else:
            prediction = model.predict(input_data)[0]
            proba = 0.0

        resultat = "GRAVE" if prediction == 1 else "PAS GRAVE"
        mlflow.log_metric("probabilite_grave", round(proba * 100, 1))
        mlflow.set_tag("source", "api")

    save_prediction(
        age=data.age,
        sexe=SEXE_MAP.get(data.sexe, "Inconnu"),
        vehicule=VEHICULE_MAP.get(data.catv, "Autre"),
        meteo=METEO_MAP.get(data.atm, "Inconnue"),
        luminosite=LUM_MAP.get(data.lum, "Inconnue"),
        type_route=ROUTE_MAP.get(data.catr, "Inconnue"),
        resultat=resultat,
        probabilite=round(proba * 100, 1),
    )

    return {
        "prediction": int(prediction),
        "label": resultat,
        "probabilite_grave": round(proba * 100, 1),
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    lum: int = Form(...),
    agg: int = Form(...),
    intersection: int = Form(...),
    atm: int = Form(...),
    col: int = Form(...),
    catr: int = Form(...),
    catv: int = Form(...),
    heure: int = Form(...),
    jour_semaine: int = Form(...),
    weekend: int = Form(...),
    sexe: int = Form(...),
    age: int = Form(...),
    secu1: int = Form(...),
    terre_plein: int = Form(...),
):
    input_data = np.array([[
        lum, agg, intersection, atm, col, catr, catv,
        heure, jour_semaine, weekend, sexe, age, secu1, terre_plein
    ]])

    if hasattr(model, "predict_proba"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
    else:
        prediction = model.predict(input_data)[0]
        proba = 0.0

    if prediction == 1:
        resultat = "⚠️ ACCIDENT GRAVE"
        resultat_db = "GRAVE"
        couleur = "danger"
    else:
        resultat = "✅ ACCIDENT PAS GRAVE"
        resultat_db = "PAS GRAVE"
        couleur = "success"

    save_prediction(
        age=age,
        sexe=SEXE_MAP.get(sexe, "Inconnu"),
        vehicule=VEHICULE_MAP.get(catv, "Autre"),
        meteo=METEO_MAP.get(atm, "Inconnue"),
        luminosite=LUM_MAP.get(lum, "Inconnue"),
        type_route=ROUTE_MAP.get(catr, "Inconnue"),
        resultat=resultat_db,
        probabilite=round(proba * 100, 1),
    )

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": resultat, "couleur": couleur, "proba": f"{proba * 100:.1f}"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

