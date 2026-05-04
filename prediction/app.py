from pathlib import Path
import json
from typing import Dict, List, Optional

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


def normalize_value(value) -> str:
    return (
        str(value if value is not None else "")
        .strip()
        .lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ç", "c")
        .replace(" ", "_")
        .replace("-", "_")
    )


def normalize_type_machine(value) -> str:
    normalized = normalize_value(value)
    mapping = {
        "moderne_2_phases": "2_phase",
        "moderne_3_phases": "3_phase",
        "traditionnelle": "presse",
        "2_phases": "2_phase",
        "3_phases": "3_phase",
        "centrifugation_2_phases": "2_phase",
        "centrifugation_3_phases": "3_phase",
        "presse_hydraulique": "presse",
        "presse": "presse",
    }
    return mapping.get(normalized, normalized)


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


class GuideStepInput(BaseModel):
    code_etape: str = Field(..., example="nettoyage_lavage")
    nom: Optional[str] = Field(default=None, example="Nettoyage / Lavage")
    machine_id: Optional[int] = Field(default=None, example=12)
    machine_type: Optional[str] = Field(default=None, example="soufflerie")
    parametres: Dict[str, str | int | float |
                     bool] = Field(default_factory=dict)


class PredictionInput(BaseModel):
    variete: str = Field(..., example="Chemlali")
    region: str = Field(..., example="Mahdia")
    methode_recolte: str = Field(..., example="manuelle")
    type_sol: str = Field(..., example="calcaire")
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
    lavage_effectue: str = Field(..., example="oui")
    type_machine: str = Field(..., example="3_phase")
    type_broyeur: Optional[str] = Field(default=None, example="marteaux")
    type_malaxeur: Optional[str] = Field(default=None, example="vertical")
    type_nettoyage: Optional[str] = Field(default=None, example="soufflerie")
    type_separation: Optional[str] = Field(
        default=None, example="decanteur_3_phases")
    nombre_etapes: Optional[int] = Field(default=None, ge=1, example=7)
    presence_ajout_eau: Optional[int] = Field(
        default=None, ge=0, le=1, example=1)
    presence_presse: Optional[int] = Field(default=None, ge=0, le=1, example=0)
    presence_separateur: Optional[int] = Field(
        default=None, ge=0, le=1, example=1)
    controle_temperature: str = Field(..., example="oui")
    guide_steps: List[GuideStepInput] = Field(default_factory=list)

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


def infer_type_nettoyage(payload: dict) -> str:
    leaves = float(payload.get("taux_feuilles_pourcent", 0) or 0)
    humidity = float(payload.get("humidite_pourcent", 0) or 0)
    lavage = normalize_value(payload.get("lavage_effectue", ""))

    if leaves >= 1.2:
        return "soufflerie"
    if lavage == "oui" and humidity >= 17:
        return "laveuse_eau"
    return "separateur_feuilles"


def infer_type_machine(payload: dict) -> str:
    value = normalize_type_machine(payload.get("type_machine"))
    if value in {"2_phase", "3_phase", "presse"}:
        return value
    if int(payload.get("presence_presse", 0) or 0) == 1:
        return "presse"
    if int(payload.get("presence_ajout_eau", 0) or 0) == 1:
        return "3_phase"
    return "2_phase"


def derive_payload_features(data: PredictionInput) -> dict:
    payload = data.model_dump()
    steps = payload.pop("guide_steps", [])

    payload["type_machine"] = infer_type_machine(payload)
    payload["type_broyeur"] = normalize_value(payload.get("type_broyeur")) or (
        "meule" if payload["type_machine"] == "presse" else "marteaux"
    )
    payload["type_malaxeur"] = normalize_value(payload.get("type_malaxeur")) or (
        "vertical" if payload["type_machine"] == "3_phase" else "horizontal"
    )
    payload["type_nettoyage"] = normalize_value(payload.get(
        "type_nettoyage")) or infer_type_nettoyage(payload)
    payload["type_separation"] = normalize_value(payload.get("type_separation")) or {
        "3_phase": "decanteur_3_phases",
        "2_phase": "decanteur_2_phases",
        "presse": "decantation_naturelle",
    }[payload["type_machine"]]
    payload["nombre_etapes"] = int(payload.get("nombre_etapes") or (
        {"3_phase": 7, "2_phase": 6, "presse": 6}[payload["type_machine"]]))
    payload["presence_ajout_eau"] = int(payload.get("presence_ajout_eau") if payload.get(
        "presence_ajout_eau") is not None else (1 if payload["type_machine"] == "3_phase" else 0))
    payload["presence_presse"] = int(payload.get("presence_presse") if payload.get(
        "presence_presse") is not None else (1 if payload["type_machine"] == "presse" else 0))
    payload["presence_separateur"] = int(payload.get("presence_separateur") if payload.get(
        "presence_separateur") is not None else (1 if payload["type_machine"] in {"3_phase", "presse"} else 0))

    if steps:
        payload["nombre_etapes"] = len(steps)
        for step in steps:
            code = normalize_value(step.get("code_etape"))
            machine_type = normalize_type_machine(step.get("machine_type"))
            if code in {"nettoyage", "nettoyage_lavage", "lavage"} and machine_type:
                payload["type_nettoyage"] = machine_type
            if code in {"broyage", "broyage_meule"} and machine_type:
                payload["type_broyeur"] = machine_type
            if code in {"malaxage"} and machine_type:
                payload["type_malaxeur"] = machine_type
            if code in {"separation", "decantation", "decanteur_2_phases", "decanteur_3_phases"} and machine_type:
                payload["type_separation"] = machine_type
            if code in {"ajout_eau", "ajout_d_eau"}:
                payload["presence_ajout_eau"] = 1
            if code in {"presse", "extraction_presse"}:
                payload["presence_presse"] = 1
            if code in {"separation", "decantation", "decanteur_2_phases", "decanteur_3_phases"}:
                payload["presence_separateur"] = 1

    return payload


def select_models(mode: str):
    if mode == "with_lab":
        return (
            quality_model_with_lab,
            yield_model_with_lab,
            metadata["modes"]["with_lab"]["feature_columns"],
            metadata["modes"]["with_lab"]["categorical_features"],
            metadata["modes"]["with_lab"]["numeric_features"],
        )
    return (
        quality_model_no_lab,
        yield_model_no_lab,
        metadata["modes"]["no_lab"]["feature_columns"],
        metadata["modes"]["no_lab"]["categorical_features"],
        metadata["modes"]["no_lab"]["numeric_features"],
    )


def build_input_frame(
    data: PredictionInput,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> pd.DataFrame:
    payload = derive_payload_features(data)
    frame = pd.DataFrame([payload])

    # Remplir les colonnes manquantes avec des valeurs par défaut appropriées
    default_values = {
        "nombre_etapes": 6,
        "presence_ajout_eau": 0,
        "presence_presse": 0,
        "presence_separateur": 0,
        "type_broyeur": "marteaux",
        "type_malaxeur": "horizontal",
        "type_nettoyage": "separateur_feuilles",
        "type_separation": "decanteur_2_phases",
    }

    for column in feature_columns:
        if column not in frame.columns:
            # Utiliser les valeurs par défaut si disponibles
            default_val = default_values.get(column, 0)
            frame[column] = default_val

    # Garder les colonnes catégorielles en texte; ne convertir que les colonnes numériques.
    for col in categorical_columns:
        if col in frame.columns:
            frame[col] = frame[col].fillna("").astype(str)

    for col in numeric_columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

    return frame[feature_columns]


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        mode = "with_lab" if has_lab_analysis(data) else "no_lab"
        quality_model, yield_model, feature_columns, categorical_columns, numeric_columns = select_models(
            mode)

        # Log pour debugging
        print(f"\n=== DEBUG PREDICTION ===")
        print(f"Mode: {mode}")
        print(f"Features requises: {feature_columns}")

        input_df = build_input_frame(
            data, feature_columns, categorical_columns, numeric_columns)

        print(f"Input DataFrame shape: {input_df.shape}")
        print(f"Input DataFrame dtypes:\n{input_df.dtypes}")
        print(f"Input DataFrame values:\n{input_df}")
        print(f"=== FIN DEBUG ===\n")

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
        import traceback
        print(f"\n!!! ERREUR PREDICTION !!!")
        print(traceback.format_exc())
        print(f"!!! FIN ERREUR !!!\n")
        raise HTTPException(status_code=500, detail=str(exc))
