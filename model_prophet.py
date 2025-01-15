import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PredicteurTemporel:
    def __init__(self, chemin_donnees='Final_data/donnees_completes_logistique_formatted.csv'):
        """Initialise le prédicteur avec Prophet"""
        self.models = {}  # Dictionnaire pour stocker les modèles par établissement/article
        
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            # Chargement des données
            self.df_historique = pd.read_csv(chemin_donnees, low_memory=False)
            self.df_historique['DATE'] = pd.to_datetime(self.df_historique['DATE'])
            self.logger.info("✅ Prédicteur temporel initialisé")
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
            raise e

    def preparer_donnees_prophet(self, df, etablissement=None, article=None):
        """Prépare les données pour Prophet"""
        try:
            # Filtrage des données
            df_filtered = df.copy()
            if etablissement:
                df_filtered = df_filtered[df_filtered['ETBDES'] == etablissement]
            if article:
                df_filtered = df_filtered[df_filtered['ARTDES'] == article]
            
            # Agrégation par date
            df_agg = df_filtered.groupby('DATE')['QUANTITE'].sum().reset_index()
            
            # Renommage des colonnes pour Prophet
            df_prophet = df_agg.rename(columns={
                'DATE': 'ds',
                'QUANTITE': 'y'
            })
            
            return df_prophet
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la préparation des données: {str(e)}")
            raise e

    def entrainer_modele(self, etablissement=None, article=None):
        """Entraîne un modèle Prophet pour une combinaison établissement/article"""
        try:
            # Création de la clé unique pour le modèle
            model_key = f"{etablissement}_{article}"
            
            # Préparation des données
            df_prophet = self.preparer_donnees_prophet(
                self.df_historique, 
                etablissement, 
                article
            )
            
            # Configuration du modèle Prophet avec paramètres optimisés
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,  # Flexibilité des changements de tendance
                seasonality_prior_scale=10,    # Force de la saisonnalité
                seasonality_mode='multiplicative'
            )
            
            # Entraînement du modèle
            model.fit(df_prophet)
            
            # Stockage du modèle
            self.models[model_key] = model
            
            self.logger.info(f"✅ Modèle entraîné pour {model_key}")
            return model
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
            raise e

    def predire(self, dates_prediction, etablissement=None, article=None):
        """Prédit les quantités pour des dates futures"""
        try:
            model_key = f"{etablissement}_{article}"
            
            # Vérification si le modèle existe, sinon l'entraîner
            if model_key not in self.models:
                self.entrainer_modele(etablissement, article)
            
            model = self.models[model_key]
            
            # Conversion des dates en format compatible avec Prophet
            if isinstance(dates_prediction, pd.DatetimeIndex):
                dates_prediction = dates_prediction.to_pydatetime()
            
            # Création du DataFrame de prédiction
            future_dates = pd.DataFrame({'ds': dates_prediction})
            
            # Prédiction
            forecast = model.predict(future_dates)
            
            # Préparation des résultats
            resultats = []
            for _, row in forecast.iterrows():
                resultat = {
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'prediction': max(0, round(row['yhat'], 2)),
                    'intervalle_confiance': {
                        'min': max(0, round(row['yhat_lower'], 2)),
                        'max': round(row['yhat_upper'], 2)
                    },
                    'tendance': round(row['trend'], 2),
                    'composante_hebdomadaire': round(row.get('weekly', 0), 2),
                    'composante_annuelle': round(row.get('yearly', 0), 2)
                }
                
                if etablissement:
                    resultat['etablissement'] = etablissement
                if article:
                    resultat['article'] = article
                    
                resultats.append(resultat)
            
            return resultats
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            raise e

    def evaluer_performances(self, date_debut, date_fin, etablissement=None, article=None):
        """
        Évalue les performances du modèle sur une période donnée
        """
        try:
            # Convertir les dates en datetime si nécessaire
            date_debut = pd.to_datetime(date_debut)
            date_fin = pd.to_datetime(date_fin)
            
            # Créer la période de test
            dates_test = pd.date_range(start=date_debut, end=date_fin)
            
            # Obtenir les données réelles d'abord
            df_reel = self.preparer_donnees_prophet(
                self.df_historique[self.df_historique['DATE'].isin(dates_test)],
                etablissement,
                article
            )
            
            if df_reel.empty:
                self.logger.warning("⚠️ Pas de données réelles disponibles pour cette période")
                return None
            
            # Ne garder que les dates pour lesquelles nous avons des données réelles
            dates_disponibles = df_reel['ds'].tolist()
            
            # Obtenir les prédictions pour ces dates spécifiques
            predictions = self.predire(dates_disponibles, etablissement, article)
            
            # Création des arrays pour les métriques
            y_true = df_reel['y'].values
            y_pred = np.array([pred['prediction'] for pred in predictions])
            
            # Vérification de la cohérence des données
            if len(y_true) != len(y_pred):
                raise ValueError(f"Nombre différent de valeurs réelles ({len(y_true)}) et prédites ({len(y_pred)})")
            
            # Calcul des métriques
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                'r2': r2_score(y_true, y_pred),
                'me': np.mean(y_pred - y_true)
            }
            
            # Calcul de la précision directionnelle
            if len(y_true) > 1:  # On ne peut calculer la direction que s'il y a au moins 2 points
                direction_reelle = np.diff(y_true) > 0
                direction_pred = np.diff(y_pred) > 0
                metrics['precision_directionnelle'] = np.mean(direction_reelle == direction_pred) * 100
            
            # Calcul de l'intervalle de confiance à 95%
            residuals = y_pred - y_true
            metrics['ic_95'] = {
                'lower': np.percentile(residuals, 2.5),
                'upper': np.percentile(residuals, 97.5)
            }
            
            # Création des visualisations
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Prédictions vs Réel
            plt.subplot(2, 1, 1)
            plt.plot(dates_disponibles, y_true, 'b-', label='Réel', linewidth=2)
            plt.plot(dates_disponibles, y_pred, 'r--', label='Prédictions', linewidth=2)
            plt.fill_between(dates_disponibles, 
                            [pred['intervalle_confiance']['min'] for pred in predictions],
                            [pred['intervalle_confiance']['max'] for pred in predictions],
                            color='r', alpha=0.1, label='Intervalle de confiance')
            plt.title('Prédictions vs Réel')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Plot 2: Distribution des erreurs
            plt.subplot(2, 1, 2)
            plt.hist(residuals, bins=20, edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--', label='Erreur nulle')
            plt.axvline(x=metrics['me'], color='g', linestyle='-', label='Erreur moyenne')
            plt.title('Distribution des erreurs de prédiction')
            plt.xlabel('Erreur')
            plt.ylabel('Fréquence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Affichage du rapport
            print("\n📊 Rapport de performances du modèle")
            print("=" * 50)
            print(f"Période: du {date_debut.strftime('%Y-%m-%d')} au {date_fin.strftime('%Y-%m-%d')}")
            if etablissement:
                print(f"Établissement: {etablissement}")
            if article:
                print(f"Article: {article}")
            print(f"\nNombre de points de données: {len(y_true)}")
            print("\n📈 Métriques principales:")
            print(f"RMSE: {metrics['rmse']:.2f}")
            print(f"MAE: {metrics['mae']:.2f}")
            print(f"MAPE: {metrics['mape']:.2f}%")
            print(f"R²: {metrics['r2']:.4f}")
            print(f"Erreur moyenne: {metrics['me']:.2f}")
            if 'precision_directionnelle' in metrics:
                print(f"Précision directionnelle: {metrics['precision_directionnelle']:.2f}%")
            print(f"Intervalle de confiance à 95%: [{metrics['ic_95']['lower']:.2f}, {metrics['ic_95']['upper']:.2f}]")
            
            return metrics, plt.gcf()
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'évaluation des performances: {str(e)}")
            raise e

# Code de test
if __name__ == "__main__":
    try:
        # Initialisation du prédicteur
        predicteur = PredicteurTemporel()
        
        # Paramètres pour l'évaluation
        etablissement_test = "CHU ROUEN CH.NICOLLE"
        article_test = "SERVIETTE DE TOILETTE"
        
        # Évaluation sur une période passée pour validation
        date_debut = '2024-01-01'
        date_fin = '2024-01-15'  # Période plus courte pour test
        
        # Évaluation des performances
        metrics, fig = predicteur.evaluer_performances(
            date_debut, 
            date_fin,
            etablissement_test,
            article_test
        )
        
        # Affichage du graphique
        plt.show()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {str(e)}")