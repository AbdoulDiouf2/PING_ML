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
    # Créer le format du log avec horodatage
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='a'),
            logging.StreamHandler()  # Pour afficher aussi dans la console
        ]
    )
    
    # Ajouter une ligne de séparation pour nouvelle session
    logging.info("="*50)
    logging.info("🚀 Début d'une nouvelle session de traitement")

def concatener_fichiers_mensuels(dossier_path, annee):
    """
    Concatène tous les fichiers mensuels d'une année en un seul fichier CSV.
    
    Args:
        dossier_path (str): Chemin vers le dossier contenant les fichiers
        annee (str): Année des données (ex: '2022')
    """
    # Initialiser le logging
    setup_logging()
    
    # Liste pour stocker tous les DataFrames
    all_dfs = []
    fichiers_traites = 0
    fichiers_erreur = 0
    
    logging.info(f"📅 Début du traitement pour l'année {annee}")
    logging.info(f"📂 Dossier source: {dossier_path}")
    
    # Parcourir tous les fichiers du dossier
    for i in range(1, 13):  # Pour les 12 mois
        # Construire le nom du fichier
        nom_fichier = f"{i}-{'janvier' if i==1 else 'fevrier' if i==2 else 'mars' if i==3 else 'avril' if i==4 else 'mai' if i==5 else 'juin' if i==6 else 'juillet' if i==7 else 'aout' if i==8 else 'septembre' if i==9 else 'octobre' if i==10 else 'novembre' if i==11 else 'decembre'}{annee}.xlsx"
        chemin_complet = os.path.join(dossier_path, nom_fichier)
        
        try:
            # Lire le fichier Excel
            logging.info(f"📖 Lecture du fichier {nom_fichier}...")
            df = pd.read_excel(chemin_complet)
            
            # Vérifier les colonnes attendues
            colonnes_attendues = ['ETBDES', 'ARTDES', 'DATE', 'QUANTITE']
            if not all(col in df.columns for col in colonnes_attendues):
                logging.warning(f"⚠️ Colonnes manquantes dans {nom_fichier}")
                fichiers_erreur += 1
                continue
                
            # Ajouter au DataFrame
            all_dfs.append(df)
            fichiers_traites += 1
            logging.info(f"✅ Fichier {nom_fichier} traité avec succès ({len(df)} lignes)")
            
        except Exception as e:
            logging.error(f"❌ Erreur lors de la lecture de {nom_fichier}: {str(e)}")
            fichiers_erreur += 1
    
    if not all_dfs:
        error_msg = "❌ Aucun fichier n'a pu être traité correctement"
        logging.error(error_msg)
        raise Exception(error_msg)
    
    # Concaténer tous les DataFrames
    logging.info("🔄 Concaténation des fichiers...")
    df_final = pd.concat(all_dfs, ignore_index=True)
    
    # Trier par date
    df_final = df_final.sort_values(['DATE', 'ETBDES', 'ARTDES'])
    
    # Sauvegarder en CSV
    nom_fichier_sortie = f"donnees_completes_{annee}.csv"
    df_final.to_csv(nom_fichier_sortie, index=False)
    
    # Log final avec statistiques
    logging.info("✨"*30 + " RÉSUMÉ " + "✨"*30)
    logging.info(f"📊 Fichiers traités avec succès: {fichiers_traites}")
    logging.info(f"⚠️ Fichiers en erreur: {fichiers_erreur}")
    logging.info(f"📝 Nombre total de lignes dans le fichier final: {len(df_final)}")
    logging.info(f"💾 Fichier de sortie: {nom_fichier_sortie}")
    logging.info("✨"*70)
    
    return df_final

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacer par le chemin de votre dossier contenant les fichiers
    dossier = r"Logistique-new/2024"
    try:
        df = concatener_fichiers_mensuels(dossier, "2024")
    except Exception as e:
        logging.error(f"💥 Erreur critique lors du traitement: {str(e)}")