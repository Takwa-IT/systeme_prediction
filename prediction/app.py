from pathlib import Path
import json
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_pickle(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Fichier manquant : {path}")
    return joblib.load(path)


def load_json(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Fichier manquant : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


label_encoder = load_pickle("label_encoder.pkl")

quality_model_no_lab = load_pickle("quality_model_no_lab.pkl")
yield_model_no_lab = load_pickle("yield_model_no_lab.pkl")

quality_model_with_lab = load_pickle("quality_model_with_lab.pkl")
yield_model_with_lab = load_pickle("yield_model_with_lab.pkl")

metadata = load_json("metadata.json")


app = FastAPI(
    title="API IA Huilerie - Dual Mode",
    description="Prédiction de la qualité, du rendement et de la quantité d'huile avec ou sans analyse labo",
    version="3.0.0",
)


class PredictionInput(BaseModel):
    variete: str = Field(..., example="Chemlali")
    region: str = Field(..., example="Mahdia")
    methode_recolte: str = Field(..., example="Manuelle")
    type_sol: str = Field(..., example="Calcaire")
    poids_olives_kg: float = Field(..., gt=0, example=5000)
    maturite_niveau_1_5: int = Field(..., ge=1, le=5, example=3)
    duree_stockage_jours: int = Field(..., ge=0, example=1)
    temps_depuis_recolte_heures: float = Field(..., ge=0, example=10)
    temperature_malaxage_c: float = Field(..., gt=0, example=26.0)
    duree_malaxage_min: float = Field(..., gt=0, example=34)
    vitesse_decanteur_tr_min: float = Field(..., gt=0, example=3300)
    humidite_pourcent: float = Field(..., ge=0, example=18.0)
    acidite_olives_pourcent: float = Field(..., ge=0, example=0.3)
    taux_feuilles_pourcent: float = Field(..., ge=0, example=0.8)
    lavage_effectue: str = Field(..., example="Oui")
    type_machine: str = Field(..., example="Moderne_2_phases")
    pression_extraction_bar: float = Field(..., gt=0, example=105)
    controle_temperature: str = Field(..., example="Oui")

    # Variables labo optionnelles
    acidite_huile_pourcent: Optional[float] = Field(default=None, example=0.45)
    indice_peroxyde_meq_o2_kg: Optional[float] = Field(
        default=None, example=8.5)
    polyphenols_mg_kg: Optional[float] = Field(default=None, example=380)
    k232: Optional[float] = Field(default=None, example=1.75)
    k270: Optional[float] = Field(default=None, example=0.14)


@app.get("/")
def root():
    return {"message": "API IA Huilerie active"}


@app.get("/model-info")
def model_info():
    return metadata


@app.get("/feature-importance/{mode}")
def feature_importance(mode: str):
    if mode not in ["no_lab", "with_lab"]:
        raise HTTPException(
            status_code=400, detail="Mode invalide. Utiliser 'no_lab' ou 'with_lab'.")
    return {
        "mode": mode,
        "quality_feature_importance_top15": metadata["modes"][mode]["quality_feature_importance_top15"],
        "yield_feature_importance_top15": metadata["modes"][mode]["yield_feature_importance_top15"],
    }


def has_lab_analysis(data: PredictionInput) -> bool:
    return all(
        value is not None
        for value in [
            data.acidite_huile_pourcent,
            data.indice_peroxyde_meq_o2_kg,
            data.polyphenols_mg_kg,
            data.k232,
            data.k270,
        ]
    )


def select_models(mode: str):
    if mode == "with_lab":
        return (
            quality_model_with_lab,
            yield_model_with_lab,
            metadata["modes"]["with_lab"]["feature_columns"],
        )
    return (
        quality_model_no_lab,
        yield_model_no_lab,
        metadata["modes"]["no_lab"]["feature_columns"],
    )


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        mode = "with_lab" if has_lab_analysis(data) else "no_lab"

        quality_model, yield_model, feature_columns = select_models(mode)

        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_columns]

        quality_encoded = quality_model.predict(input_df)[0]
        quality_class = label_encoder.inverse_transform([quality_encoded])[0]

        quality_probability = None
        if hasattr(quality_model.named_steps["model"], "predict_proba"):
            probas = quality_model.predict_proba(input_df)[0]
            quality_probability = float(max(probas))

        rendement_pred = float(yield_model.predict(input_df)[0])
        poids = float(data.poids_olives_kg)
        quantite_recalculee = poids * (rendement_pred / 100.0) * 0.92

        return {
            "mode_prediction": mode,
            "qualite_predite": quality_class,
            "probabilite_qualite": round(quality_probability, 4) if quality_probability is not None else None,
            "rendement_predit_pourcent": round(rendement_pred, 2),
            "quantite_huile_recalculee_litres": round(quantite_recalculee, 2),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
