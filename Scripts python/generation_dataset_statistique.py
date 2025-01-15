import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

def setup_logging(log_file="Logs/analyse_stats.log"):
    """Configure le système de logging pour l'analyse statistique"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.info("🔍 Début de l'analyse statistique")

def convertir_dates(df):
    """
    Convertit la colonne DATE en format datetime
    """
    try:
        # Essai avec le format JJ/MM/AAAA
        df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')
    except:
        try:
            # Essai avec le format AAAA-MM-JJ
            df['DATE'] = pd.to_datetime(df['DATE'])
        except Exception as e:
            logging.error(f"❌ Erreur lors de la conversion des dates: {str(e)}")
            # Afficher quelques exemples de dates pour le diagnostic
            logging.error(f"Exemples de dates dans le fichier: {df['DATE'].head().tolist()}")
            raise e
    return df

def analyser_donnees(fichier_input):
    """
    Analyse statistique complète du dataset
    
    Args:
        fichier_input (str): Chemin vers le fichier CSV à analyser
    """
    setup_logging()
    try:
        # Lecture des données
        logging.info(f"📖 Lecture du fichier {fichier_input}")
        df = pd.read_csv(fichier_input)
        
        # Afficher les informations sur les colonnes pour le diagnostic
        logging.info("\n📋 STRUCTURE DES DONNÉES")
        logging.info(f"Colonnes présentes: {df.columns.tolist()}")
        logging.info(f"Types des colonnes:\n{df.dtypes}")
        
        # Conversion des dates
        logging.info("\n🕒 Conversion des dates")
        df = convertir_dates(df)
        
        # 1. Statistiques générales
        logging.info("\n📊 STATISTIQUES GÉNÉRALES")
        logging.info(f"Nombre total de lignes: {len(df):,}")
        logging.info(f"Période couverte: de {df['DATE'].min().strftime('%d/%m/%Y')} à {df['DATE'].max().strftime('%d/%m/%Y')}")
        logging.info(f"Nombre d'établissements uniques: {df['ETBDES'].nunique():,}")
        logging.info(f"Nombre d'articles uniques: {df['ARTDES'].nunique():,}")
        
        # 2. Statistiques des quantités
        logging.info("\n📦 ANALYSE DES QUANTITÉS")
        stats_quantite = df['QUANTITE'].describe()
        for stat, valeur in stats_quantite.items():
            logging.info(f"{stat}: {valeur:,.2f}")
        
        # 3. Analyse par établissement
        stats_etb = df.groupby('ETBDES').agg({
            'QUANTITE': ['count', 'sum', 'mean', 'std']
        }).round(2)
        stats_etb.columns = ['count', 'sum', 'mean', 'std']
        stats_etb = stats_etb.reset_index().sort_values('sum', ascending=False)
        
        logging.info("\n🏢 TOP 5 DES ÉTABLISSEMENTS PAR VOLUME")
        for _, row in stats_etb.head().iterrows():
            logging.info(f"Établissement: {row['ETBDES']}")
            logging.info(f"  - Nombre de commandes: {row['count']:,}")
            logging.info(f"  - Quantité totale: {row['sum']:,.0f}")
            logging.info(f"  - Moyenne par commande: {row['mean']:,.2f}")
        
        # 4. Analyse temporelle
        df['mois'] = df['DATE'].dt.strftime('%Y-%m')
        stats_temps = df.groupby('mois').agg({
            'QUANTITE': ['count', 'sum', 'mean']
        }).round(2)
        stats_temps.columns = ['count', 'sum', 'mean']
        stats_temps = stats_temps.reset_index()
        
        logging.info("\n📅 ÉVOLUTION MENSUELLE")
        logging.info("Tendance des 3 derniers mois:")
        for _, row in stats_temps.tail(3).iterrows():
            logging.info(f"Mois: {row['mois']}")
            logging.info(f"  - Nombre de commandes: {row['count']:,}")
            logging.info(f"  - Quantité totale: {row['sum']:,.0f}")
        
        # 5. Analyse des articles
        stats_articles = df.groupby('ARTDES').agg({
            'QUANTITE': ['count', 'sum', 'mean']
        }).round(2)
        stats_articles.columns = ['count', 'sum', 'mean']
        stats_articles = stats_articles.reset_index().sort_values('sum', ascending=False)
        
        logging.info("\n📦 TOP 5 DES ARTICLES LES PLUS COMMANDÉS")
        for _, row in stats_articles.head().iterrows():
            logging.info(f"Article: {row['ARTDES']}")
            logging.info(f"  - Nombre de commandes: {row['count']:,}")
            logging.info(f"  - Quantité totale: {row['sum']:,.0f}")
        
        # Génération des fichiers de statistiques détaillées
        stats_etb.to_csv('statistiques_etablissements.csv', index=False)
        stats_articles.to_csv('statistiques_articles.csv', index=False)
        stats_temps.to_csv('statistiques_temporelles.csv', index=False)
        
        logging.info("\n💾 FICHIERS GÉNÉRÉS")
        logging.info("- statistiques_etablissements.csv")
        logging.info("- statistiques_articles.csv")
        logging.info("- statistiques_temporelles.csv")
        
        # 6. Détection des anomalies
        seuil_quantite = stats_quantite['mean'] + 3 * stats_quantite['std']
        anomalies = df[df['QUANTITE'] > seuil_quantite]
        
        if not anomalies.empty:
            logging.info("\n⚠️ DÉTECTION DES ANOMALIES")
            logging.info(f"Nombre de commandes potentiellement anormales: {len(anomalies)}")
            anomalies.to_csv('commandes_anormales.csv', index=False)
            logging.info("Liste sauvegardée dans 'commandes_anormales.csv'")
        
        # 7. Calculer les variations
        df['jour_semaine'] = df['DATE'].dt.strftime('%A')  # Nom du jour en anglais
        stats_jour = df.groupby('jour_semaine')['QUANTITE'].mean().round(2)
        
        logging.info("\n📊 VARIATIONS PAR JOUR DE LA SEMAINE")
        for jour, moyenne in stats_jour.items():
            logging.info(f"{jour}: {moyenne:,.2f} unités en moyenne")
            
        return df
        
    except Exception as e:
        logging.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        # Modifier le nom du fichier selon votre cas
        df = analyser_donnees("Final_data/donnees_completes_logistique_2022-2024.csv")
    except Exception as e:
        logging.error(f"💥 Erreur critique: {str(e)}")