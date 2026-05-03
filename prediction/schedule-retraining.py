"""
Script d'orchestration pour réentraînement automatique du modèle
Utilise les valeurs réelles du feedback loop pour amélioration continue
Exécution: python schedule_retraining.py
"""

import subprocess
import requests
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import time
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
BACKEND_URL = "http://localhost:8000"
IA_URL = "http://localhost:7500"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / "retraining.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RetrainingOrchestrator:
    """Orchestre le processus de réentraînement automatique"""
    
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.ia_url = IA_URL
        self.retraining_count = 0
    
    def fetch_real_values_data(self, days_back=30):
        """Récupère les données réelles du backend"""
        try:
            depuis = (datetime.now() - timedelta(days=days_back)).isoformat()
            jusqu = datetime.now().isoformat()
            
            url = f"{self.backend_url}/api/execution-productions/valeurs-reelles/export-retrain"
            params = {"depuis": depuis, "jusqu": jusqu}
            
            logger.info(f"📥 Téléchargement des valeurs réelles depuis {days_back} jours...")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                data = result.get('data', [])
                logger.info(f"✅ {len(data)} valeurs réelles récupérées")
                return data
            else:
                logger.error(f"❌ Erreur {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du téléchargement: {e}")
            return None
    
    def check_if_retraining_needed(self, real_values_data):
        """Vérifie si le réentraînement est justifié"""
        if not real_values_data or len(real_values_data) < 20:
            logger.info("⏭️ Pas assez de données réelles pour réentraînement (min 20)")
            return False
        
        # Vérifier le nombre d'exécutions avec déviation importante
        important_deviations = sum(
            1 for rv in real_values_data 
            if rv.get("isSignificantDeviation", False)
        )
        
        if important_deviations / len(real_values_data) > 0.3:
            logger.info(f"⚠️ {important_deviations} déviations importantes détectées")
            return True
        
        logger.info("✅ Performance acceptable, pas de réentraînement nécessaire")
        return False
    
    def create_retraining_dataset(self, real_values_data):
        """Crée un dataset CSV pour réentraînement"""
        try:
            logger.info("📊 Création du dataset de réentraînement...")
            
            # Convertir données réelles en DataFrame
            rows = []
            for rv in real_values_data:
                rows.append({
                    'execution_id': rv.get('executionProductionId'),
                    'parametre_nom': rv.get('nomParametre'),
                    'valeur_reelle': rv.get('valeurReelle'),
                    'valeur_estimee': rv.get('valeurEstimee'),
                    'deviation_pourcent': rv.get('deviation', 0),
                    'qualite_deviation': rv.get('qualiteDeviation', ''),
                    'date_creation': rv.get('dateCreation'),
                })
            
            df_reals = pd.DataFrame(rows)
            
            # Charger dataset synthétique existant
            existing_data_path = DATA_DIR / "dataset_huilerie_avec_realvalues_5000.csv"
            if existing_data_path.exists():
                df_existing = pd.read_csv(existing_data_path)
                logger.info(f"📋 {len(df_existing)} lignes dataset existant")
                
                # Combiner (prioriser données réelles)
                df_combined = pd.concat([df_reals, df_existing], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['execution_id'], keep='first')
                
            else:
                df_combined = df_reals
            
            # Sauvegarder dataset mixte
            mixed_dataset_path = DATA_DIR / f"dataset_mixed_{datetime.now().strftime('%Y%m%d')}.csv"
            df_combined.to_csv(mixed_dataset_path, index=False)
            
            logger.info(f"✅ Dataset créé: {mixed_dataset_path} ({len(df_combined)} lignes)")
            return str(mixed_dataset_path)
            
        except Exception as e:
            logger.error(f"❌ Erreur création dataset: {e}")
            return None
    
    def run_retraining(self, dataset_path):
        """Lance le réentraînement du modèle"""
        try:
            logger.info("🚀 Lancement du réentraînement...")
            
            cmd = [
                "python",
                str(BASE_DIR / "train_dual_mode_models.py"),
                "--data", dataset_path,
                "--output", str(MODELS_DIR)
            ]
            
            logger.info(f"   Commande: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Réentraînement réussi!")
                logger.info(f"   Output: {result.stdout[-500:]}")  # Derniers 500 chars
                self.retraining_count += 1
                return True
            else:
                logger.error(f"❌ Réentraînement échoué!")
                logger.error(f"   Erreur: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Réentraînement dépassé (timeout 30 min)")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur réentraînement: {e}")
            return False
    
    def verify_model_update(self):
        """Vérifie que les modèles ont bien été mis à jour"""
        try:
            # Vérifier métadata.json récent
            metadata_path = MODELS_DIR / "metadata.json"
            if metadata_path.exists():
                mtime = metadata_path.stat().st_mtime
                mod_time = datetime.fromtimestamp(mtime)
                age_minutes = (datetime.now() - mod_time).total_seconds() / 60
                
                if age_minutes < 5:
                    logger.info(f"✅ Modèles mis à jour il y a {age_minutes:.0f} min")
                    return True
            
            logger.error("❌ Modèles non mis à jour")
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification modèles: {e}")
            return False
    
    def create_backup_current_models(self):
        """Sauvegarde les modèles actuels avant réentraînement"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = MODELS_DIR / f"backup_{timestamp}"
            backup_dir.mkdir(exist_ok=True)
            
            for pkl_file in MODELS_DIR.glob("*.pkl"):
                import shutil
                shutil.copy(pkl_file, backup_dir / pkl_file.name)
            
            logger.info(f"📦 Modèles sauvegardés: {backup_dir}")
            return str(backup_dir)
            
        except Exception as e:
            logger.error(f"❌ Erreur backup: {e}")
            return None
    
    def run_retraining_cycle(self):
        """Cycle complet de réentraînement"""
        logger.info("=" * 60)
        logger.info("🔄 CYCLE DE RÉENTRAÎNEMENT AUTOMATIQUE")
        logger.info(f"   Timestamp: {datetime.now()}")
        logger.info("=" * 60)
        
        try:
            # 1. Récupérer données réelles
            real_values = self.fetch_real_values_data(days_back=30)
            if not real_values:
                logger.info("⏭️ Cycle annulé: pas de données réelles")
                return
            
            # 2. Vérifier si réentraînement nécessaire
            if not self.check_if_retraining_needed(real_values):
                logger.info("⏭️ Cycle annulé: performance acceptable")
                return
            
            # 3. Sauvegarder modèles actuels
            backup = self.create_backup_current_models()
            
            # 4. Créer dataset mixte
            dataset_path = self.create_retraining_dataset(real_values)
            if not dataset_path:
                logger.error("❌ Impossible de créer le dataset")
                return
            
            # 5. Lancer réentraînement
            success = self.run_retraining(dataset_path)
            
            if success:
                # 6. Vérifier mise à jour modèles
                if self.verify_model_update():
                    logger.info("✅ CYCLE DE RÉENTRAÎNEMENT RÉUSSI!")
                    logger.info(f"   Cycles réussis: {self.retraining_count}")
                else:
                    logger.error("❌ Modèles non mise à jour correctement")
                    if backup:
                        logger.info(f"   Modèles sauvegardés: {backup}")
            else:
                logger.error("❌ Réentraînement échoué")
                if backup:
                    logger.info(f"   Modèles sauvegardés: {backup}")
        
        except Exception as e:
            logger.error(f"❌ Erreur cycle: {e}")
        
        finally:
            logger.info("=" * 60)
            logger.info("")


def schedule_retraining_jobs():
    """Configure les jobs de réentraînement"""
    
    orchestrator = RetrainingOrchestrator()
    
    # Schedule: Réentraînement hebdomadaire le lundi à 2h du matin
    schedule.every().monday.at("02:00").do(orchestrator.run_retraining_cycle)
    
    logger.info("📅 Schedule configuré:")
    logger.info("   • Réentraînement chaque lundi à 02:00")
    logger.info("   • Modèles sauvegardés avant chaque cycle")
    logger.info("")
    
    # Boucle de scheduler
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Vérifier chaque minute
        except KeyboardInterrupt:
            logger.info("🛑 Scheduler arrêté")
            break
        except Exception as e:
            logger.error(f"❌ Erreur scheduler: {e}")
            time.sleep(60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "schedule":
        # Mode scheduler (background)
        schedule_retraining_jobs()
    else:
        # Mode cycle unique
        orchestrator = RetrainingOrchestrator()
        orchestrator.run_retraining_cycle()
