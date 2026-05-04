from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_huilerie_multietapes_5000.csv"
ENRICHED_DATASET_PATH = BASE_DIR / "data" / "dataset_huilerie_multietapes.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


TARGET_QUALITY = "classe_qualite"
TARGET_YIELD = "rendement_extraction_pourcent"
WEIGHT_COLUMN = "poids_olives_kg"

CATEGORICAL_FEATURES = [
    "variete",
    "region",
    "methode_recolte",
    "type_sol",
    "lavage_effectue",
    "type_machine",
    "type_broyeur",
    "type_malaxeur",
    "type_nettoyage",
    "type_separation",
    "controle_temperature",
]

NUMERIC_FEATURES = [
    "poids_olives_kg",
    "maturite_niveau_1_5",
    "duree_stockage_jours",
    "temps_depuis_recolte_heures",
    "temperature_malaxage_c",
    "duree_malaxage_min",
    "vitesse_decanteur_tr_min",
    "humidite_pourcent",
    "acidite_olives_pourcent",
    "taux_feuilles_pourcent",
    "pression_extraction_bar",
    "nombre_etapes",
    "presence_ajout_eau",
    "presence_presse",
    "presence_separateur",
]

LAB_FEATURES = [
    "acidite_huile_pourcent",
    "indice_peroxyde_meq_o2_kg",
    "polyphenols_mg_kg",
    "k232",
    "k270",
]

NO_LAB_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
WITH_LAB_FEATURES = NO_LAB_FEATURES + LAB_FEATURES


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Le dataset est vide.")
    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")


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


def derive_type_nettoyage(row: pd.Series) -> str:
    leaves = float(row.get("taux_feuilles_pourcent", 0) or 0)
    humidity = float(row.get("humidite_pourcent", 0) or 0)
    lavage = normalize_value(row.get("lavage_effectue", ""))

    if leaves >= 1.2:
        return "soufflerie"
    if lavage == "oui" and humidity >= 17:
        return "laveuse_eau"
    return "separateur_feuilles"


def derive_enriched_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["type_machine"] = enriched["type_machine"].map(
        normalize_type_machine)
    enriched["type_broyeur"] = enriched["type_machine"].map(
        lambda value: "meule" if value == "presse" else "marteaux"
    )
    enriched["type_malaxeur"] = enriched["type_machine"].map(
        lambda value: "vertical" if value == "3_phase" else "horizontal"
    )
    enriched["type_nettoyage"] = enriched.apply(derive_type_nettoyage, axis=1)
    enriched["type_separation"] = enriched["type_machine"].map(
        lambda value: {
            "3_phase": "decanteur_3_phases",
            "2_phase": "decanteur_2_phases",
            "presse": "decantation_naturelle",
        }.get(value, "decanteur_2_phases")
    )
    enriched["nombre_etapes"] = enriched["type_machine"].map(
        lambda value: {"3_phase": 7, "2_phase": 6, "presse": 6}.get(value, 0)
    )
    enriched["presence_ajout_eau"] = enriched["type_machine"].map(
        lambda value: 1 if value == "3_phase" else 0
    )
    enriched["presence_presse"] = enriched["type_machine"].map(
        lambda value: 1 if value == "presse" else 0
    )
    enriched["presence_separateur"] = enriched["type_machine"].map(
        lambda value: 1 if value in {"3_phase", "presse"} else 0
    )

    return enriched


def ensure_feature_schema(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    prepared = df.copy()
    for column in feature_columns:
        if column not in prepared.columns:
            prepared[column] = np.nan
    return prepared[feature_columns]


def build_preprocessor(X: pd.DataFrame, include_lab_features: bool = False):
    categorical_features = [
        col for col in CATEGORICAL_FEATURES if col in X.columns]
    numeric_candidates = NUMERIC_FEATURES + \
        (LAB_FEATURES if include_lab_features else [])
    numeric_features = [col for col in numeric_candidates if col in X.columns]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    return preprocessor, categorical_features, numeric_features


def optimize_classifier(X_train, y_train, preprocessor):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [None, 10, 15, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="f1_weighted",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def optimize_regressor(X_train, y_train, preprocessor):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [None, 10, 15, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="r2",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_classifier(model, X_test, y_test, label_encoder, title):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {title} ===")
    print(f"Accuracy : {acc:.4f}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))


def evaluate_regressor(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"\n=== {title} ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


def get_feature_names_from_pipeline(fitted_pipeline: Pipeline) -> list[str]:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return [f"feature_{i}" for i in range(len(fitted_pipeline.named_steps["model"].feature_importances_))]


def extract_feature_importance(fitted_pipeline: Pipeline, top_n: int = 15) -> list[dict]:
    model = fitted_pipeline.named_steps["model"]
    importances = model.feature_importances_
    feature_names = get_feature_names_from_pipeline(fitted_pipeline)

    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            "feature": feature,
            "importance": round(float(importance), 6),
        }
        for feature, importance in pairs[:top_n]
    ]


def train_mode(df, feature_columns, mode_name, label_encoder, include_lab_features: bool = False):
    X = df[feature_columns].copy()
    y_quality = label_encoder.transform(df[TARGET_QUALITY].copy())
    y_yield = df[TARGET_YIELD].copy()

    preprocessor, categorical_features, numeric_features = build_preprocessor(
        X,
        include_lab_features=include_lab_features,
    )

    X_train, X_test, yq_train, yq_test, yy_train, yy_test = train_test_split(
        X,
        y_quality,
        y_yield,
        test_size=0.2,
        random_state=42,
        stratify=y_quality,
    )

    print(f"\n########## MODE {mode_name.upper()} ##########")

    print("\nOptimisation classification...")
    quality_search = optimize_classifier(X_train, yq_train, preprocessor)
    print("Meilleurs paramètres classification :")
    print(quality_search.best_params_)

    print("\nOptimisation régression rendement...")
    yield_search = optimize_regressor(X_train, yy_train, preprocessor)
    print("Meilleurs paramètres régression :")
    print(yield_search.best_params_)

    evaluate_classifier(
        quality_search.best_estimator_,
        X_test,
        yq_test,
        label_encoder,
        f"Classification qualité - {mode_name}",
    )

    evaluate_regressor(
        yield_search.best_estimator_,
        X_test,
        yy_test,
        f"Régression rendement - {mode_name}",
    )

    quality_importance = extract_feature_importance(
        quality_search.best_estimator_)
    yield_importance = extract_feature_importance(yield_search.best_estimator_)

    return {
        "quality_model": quality_search.best_estimator_,
        "yield_model": yield_search.best_estimator_,
        "quality_best_params": quality_search.best_params_,
        "yield_best_params": yield_search.best_params_,
        "quality_feature_importance": quality_importance,
        "yield_feature_importance": yield_importance,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }


def save_artifacts(label_encoder, no_lab_results, with_lab_results):
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    joblib.dump(no_lab_results["quality_model"],
                MODELS_DIR / "quality_model_no_lab.pkl")
    joblib.dump(no_lab_results["yield_model"],
                MODELS_DIR / "yield_model_no_lab.pkl")

    joblib.dump(with_lab_results["quality_model"],
                MODELS_DIR / "quality_model_with_lab.pkl")
    joblib.dump(with_lab_results["yield_model"],
                MODELS_DIR / "yield_model_with_lab.pkl")

    metadata = {
        "target_quality": TARGET_QUALITY,
        "target_yield": TARGET_YIELD,
        "weight_column": WEIGHT_COLUMN,
        "modes": {
            "no_lab": {
                "feature_columns": no_lab_results["feature_columns"],
                "categorical_features": no_lab_results["categorical_features"],
                "numeric_features": no_lab_results["numeric_features"],
                "quality_best_params": no_lab_results["quality_best_params"],
                "yield_best_params": no_lab_results["yield_best_params"],
                "quality_feature_importance_top15": no_lab_results["quality_feature_importance"],
                "yield_feature_importance_top15": no_lab_results["yield_feature_importance"],
            },
            "with_lab": {
                "feature_columns": with_lab_results["feature_columns"],
                "categorical_features": with_lab_results["categorical_features"],
                "numeric_features": with_lab_results["numeric_features"],
                "quality_best_params": with_lab_results["quality_best_params"],
                "yield_best_params": with_lab_results["yield_best_params"],
                "quality_feature_importance_top15": with_lab_results["quality_feature_importance"],
                "yield_feature_importance_top15": with_lab_results["yield_feature_importance"],
            },
        },
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== Sauvegarde terminée ===")
    for path in [
        MODELS_DIR / "label_encoder.pkl",
        MODELS_DIR / "quality_model_no_lab.pkl",
        MODELS_DIR / "yield_model_no_lab.pkl",
        MODELS_DIR / "quality_model_with_lab.pkl",
        MODELS_DIR / "yield_model_with_lab.pkl",
        MODELS_DIR / "metadata.json",
    ]:
        print(path)


def export_enriched_dataset(source_path: Path = DATA_PATH, target_path: Path = ENRICHED_DATASET_PATH) -> pd.DataFrame:
    raw_df = load_dataset(source_path)
    enriched_df = derive_enriched_columns(raw_df)
    enriched_df.to_csv(target_path, index=False)
    return enriched_df


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement IA huile d'olive multi-étapes")
    parser.add_argument("--export-only", action="store_true",
                        help="Génère seulement le dataset enrichi")
    parser.add_argument("--data", type=Path,
                        default=DATA_PATH, help="Chemin du CSV source")
    parser.add_argument("--output", type=Path,
                        default=ENRICHED_DATASET_PATH, help="Chemin du CSV enrichi")
    args = parser.parse_args()

    df = export_enriched_dataset(args.data, args.output)

    validate_columns(
        df,
        [TARGET_QUALITY, TARGET_YIELD, WEIGHT_COLUMN] +
        NO_LAB_FEATURES + LAB_FEATURES,
    )

    if args.export_only:
        print(f"Dataset enrichi exporté vers {args.output}")
        return

    label_encoder = LabelEncoder()
    label_encoder.fit(df[TARGET_QUALITY])

    no_lab_results = train_mode(df, NO_LAB_FEATURES, "no_lab", label_encoder)
    with_lab_results = train_mode(
        df,
        WITH_LAB_FEATURES,
        "with_lab",
        label_encoder,
        include_lab_features=True,
    )

    save_artifacts(label_encoder, no_lab_results, with_lab_results)


if __name__ == "__main__":
    main()
