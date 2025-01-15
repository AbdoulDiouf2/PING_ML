import pandas as pd
import os
from datetime import datetime
import logging

def setup_logging(log_file="Logs/execution_logs.log"):
    """
    Configure le système de logging.
    
    Args:
        log_file (str): Nom du fichier de log
    """
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='a'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*50)
    logging.info("🚀 Début d'une nouvelle session de traitement multi-années")

def concatener_fichiers_mensuels(dossier_path, annee):
    """
    Concatène tous les fichiers mensuels d'une année en un seul fichier CSV.
    """
    all_dfs = []
    fichiers_traites = 0
    fichiers_erreur = 0
    
    logging.info(f"📅 Traitement de l'année {annee}")
    logging.info(f"📂 Dossier: {dossier_path}")
    
    for i in range(1, 13):
        nom_fichier = f"{i}-{'janvier' if i==1 else 'fevrier' if i==2 else 'mars' if i==3 else 'avril' if i==4 else 'mai' if i==5 else 'juin' if i==6 else 'juillet' if i==7 else 'aout' if i==8 else 'septembre' if i==9 else 'octobre' if i==10 else 'novembre' if i==11 else 'decembre'}{annee}.xlsx"
        chemin_complet = os.path.join(dossier_path, nom_fichier)
        
        try:
            logging.info(f"📖 Lecture du fichier {nom_fichier}...")
            df = pd.read_excel(chemin_complet)
            
            colonnes_attendues = ['ETBDES', 'ARTDES', 'DATE', 'QUANTITE']
            if not all(col in df.columns for col in colonnes_attendues):
                logging.warning(f"⚠️ Colonnes manquantes dans {nom_fichier}")
                fichiers_erreur += 1
                continue
                
            all_dfs.append(df)
            fichiers_traites += 1
            logging.info(f"✅ Fichier {nom_fichier} traité avec succès ({len(df)} lignes)")
            
        except Exception as e:
            logging.error(f"❌ Erreur lors de la lecture de {nom_fichier}: {str(e)}")
            fichiers_erreur += 1
    
    if not all_dfs:
        error_msg = f"❌ Aucun fichier n'a pu être traité correctement pour l'année {annee}"
        logging.error(error_msg)
        return None
    
    logging.info(f"🔄 Concaténation des fichiers de {annee}...")
    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final = df_final.sort_values(['DATE', 'ETBDES', 'ARTDES'])
    
    nom_fichier_sortie = f"donnees_completes_{annee}.csv"
    df_final.to_csv(nom_fichier_sortie, index=False)
    
    logging.info(f"✨ Résumé pour {annee} ✨")
    logging.info(f"📊 Fichiers traités avec succès: {fichiers_traites}")
    logging.info(f"⚠️ Fichiers en erreur: {fichiers_erreur}")
    logging.info(f"📝 Nombre total de lignes: {len(df_final)}")
    logging.info(f"💾 Fichier sauvegardé: {nom_fichier_sortie}")
    
    return df_final

def traiter_toutes_annees(dossier_base="Logistique-new", annees=[2022, 2023, 2024]):
    """
    Traite les données pour plusieurs années et crée un fichier consolidé.
    
    Args:
        dossier_base (str): Chemin du dossier base
        annees (list): Liste des années à traiter
    """
    setup_logging()
    all_years_dfs = []
    stats_globales = {
        "annees_traitees": 0,
        "annees_erreur": 0,
        "total_lignes": 0
    }
    
    logging.info(f"🎯 Début du traitement multi-années: {annees}")
    
    for annee in annees:
        dossier = os.path.join(dossier_base, str(annee))
        try:
            df_annee = concatener_fichiers_mensuels(dossier, str(annee))
            if df_annee is not None:
                all_years_dfs.append(df_annee)
                stats_globales["annees_traitees"] += 1
                stats_globales["total_lignes"] += len(df_annee)
            else:
                stats_globales["annees_erreur"] += 1
        except Exception as e:
            logging.error(f"💥 Erreur critique pour l'année {annee}: {str(e)}")
            stats_globales["annees_erreur"] += 1
    
    if all_years_dfs:
        # Création du fichier consolidé toutes années
        df_final_global = pd.concat(all_years_dfs, ignore_index=True)
        df_final_global = df_final_global.sort_values(['DATE', 'ETBDES', 'ARTDES'])
        
        nom_fichier_global = f"donnees_completes_{min(annees)}-{max(annees)}.csv"
        df_final_global.to_csv(nom_fichier_global, index=False)
        
        # Résumé global
        logging.info("🌟"*30 + " RÉSUMÉ GLOBAL " + "🌟"*30)
        logging.info(f"📊 Années traitées avec succès: {stats_globales['annees_traitees']}")
        logging.info(f"⚠️ Années en erreur: {stats_globales['annees_erreur']}")
        logging.info(f"📝 Nombre total de lignes (toutes années): {stats_globales['total_lignes']}")
        logging.info(f"💾 Fichier consolidé: {nom_fichier_global}")
        logging.info("🌟"*75)
    else:
        logging.error("💥 Aucune année n'a pu être traitée correctement")

# Exemple d'utilisation
if __name__ == "__main__":
    try:
        traiter_toutes_annees("Logistique-new", [2022, 2023, 2024])
    except Exception as e:
        logging.error(f"💥 Erreur critique globale: {str(e)}")