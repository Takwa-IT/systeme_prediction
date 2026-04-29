# 🫒 Module IA – Prédiction de la Qualité et du Rendement de l’Huile d’Olive

Ce projet implémente un module intelligent permettant de prédire :
- la classe de qualité
- le rendement d’extraction
- la quantité d’huile produite

## Installation

python -m venv .venv

# Windows PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

## Entraînement

python train_dual_mode_models.py

## Lancer API

python -m uvicorn app:app --host 0.0.0.0 --port 7500 --reload

## Accès

http://127.0.0.1:7500/docs
