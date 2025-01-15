import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import logging

class PredicteurCommandes:
    def __init__(self, model_path='model_v2.joblib', scaler_path='scaler_v2.joblib'):
        """Initialise le prédicteur avec le modèle et le scaler sauvegardés"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Charger les données historiques pour les références
            self.df_historique = pd.read_csv("Final_data/donnees_completes_logistique_formatted.csv", low_memory=False)
            self.df_historique['DATE'] = pd.to_datetime(self.df_historique['DATE'])
            
            # Calculer jours_depuis_debut pour les features
            min_date = self.df_historique['DATE'].min()
            self.df_historique['jours_depuis_debut'] = (self.df_historique['DATE'] - min_date).dt.days
            
            # Créer les mappings pour les encodages
            self.etb_mapping = {v: k for k, v in enumerate(self.df_historique['ETBDES'].unique())}
            self.art_mapping = {v: k for k, v in enumerate(self.df_historique['ARTDES'].unique())}
            if 'PTLDES' in self.df_historique.columns:
                self.ptl_mapping = {v: k for k, v in enumerate(self.df_historique['PTLDES'].unique())}
            
            logging.info("🔄 Prédicteur initialisé avec succès")
        except Exception as e:
            logging.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
            raise e

    def _prepare_features(self, date_prediction, etablissement=None, article=None):
        """Prépare les features pour la prédiction avec correction des moyennes mobiles"""
        try:
            # Créer un DataFrame avec une seule ligne
            prediction_data = pd.DataFrame({
                'DATE': [date_prediction]
            })
            
            # 1. Features temporelles de base
            prediction_data['année'] = prediction_data['DATE'].dt.year
            prediction_data['mois'] = prediction_data['DATE'].dt.month
            prediction_data['jour'] = prediction_data['DATE'].dt.day
            prediction_data['jour_semaine'] = prediction_data['DATE'].dt.dayofweek
            prediction_data['trimestre'] = prediction_data['DATE'].dt.quarter
            
            # 2. Features cycliques
            prediction_data['mois_sin'] = np.sin(2 * np.pi * prediction_data['mois']/12)
            prediction_data['mois_cos'] = np.cos(2 * np.pi * prediction_data['mois']/12)
            prediction_data['jour_sin'] = np.sin(2 * np.pi * prediction_data['jour']/31)
            prediction_data['jour_cos'] = np.cos(2 * np.pi * prediction_data['jour']/31)
            
            # 3. Features de tendance
            min_date = self.df_historique['DATE'].min()
            prediction_data['jours_depuis_debut'] = (prediction_data['DATE'] - min_date).dt.days
            prediction_data['tendance_normalisee'] = (
                (prediction_data['jours_depuis_debut'] - self.df_historique['jours_depuis_debut'].mean()) / 
                self.df_historique['jours_depuis_debut'].std()
            )
            
            # 4. Encodage des établissements et articles
            if etablissement:
                etb_encoded = self.etb_mapping.get(etablissement, -1)
                filtered_hist = self.df_historique[self.df_historique['ETBDES'] == etablissement]
            else:
                etb_encoded = -1
                filtered_hist = self.df_historique.copy()
            prediction_data['ETBDES_encoded'] = etb_encoded
                
            if article:
                art_encoded = self.art_mapping.get(article, -1)
                filtered_hist = filtered_hist[filtered_hist['ARTDES'] == article]
            else:
                art_encoded = -1
            prediction_data['ARTDES_encoded'] = art_encoded
            
            # 5. PTLDES encoded
            prediction_data['PTLDES_encoded'] = -1
            
            # 6. Calcul des moyennes mobiles et écarts-types (CORRIGÉ)
            for lag in [1, 7, 14, 30]:
                # Calculer la date limite pour la fenêtre mobile
                date_limite = date_prediction - pd.Timedelta(days=lag)
                
                # Données globales pour la période
                donnees_periode = self.df_historique[
                    (self.df_historique['DATE'] <= date_prediction) & 
                    (self.df_historique['DATE'] > date_limite)
                ]
                
                # Pour l'établissement
                if etablissement:
                    recent_data_etb = donnees_periode[donnees_periode['ETBDES'] == etablissement]['QUANTITE']
                    prediction_data[f'moving_avg_{lag}_ETBDES'] = recent_data_etb.mean() if not recent_data_etb.empty else 0
                    prediction_data[f'moving_std_{lag}_ETBDES'] = recent_data_etb.std() if not recent_data_etb.empty else 0
                else:
                    prediction_data[f'moving_avg_{lag}_ETBDES'] = donnees_periode['QUANTITE'].mean()
                    prediction_data[f'moving_std_{lag}_ETBDES'] = donnees_periode['QUANTITE'].std()
                
                # Pour l'article
                if article:
                    recent_data_art = donnees_periode[donnees_periode['ARTDES'] == article]['QUANTITE']
                    prediction_data[f'moving_avg_{lag}_ARTDES'] = recent_data_art.mean() if not recent_data_art.empty else 0
                    prediction_data[f'moving_std_{lag}_ARTDES'] = recent_data_art.std() if not recent_data_art.empty else 0
                else:
                    prediction_data[f'moving_avg_{lag}_ARTDES'] = donnees_periode['QUANTITE'].mean()
                    prediction_data[f'moving_std_{lag}_ARTDES'] = donnees_periode['QUANTITE'].std()
            
            # 7. Calcul des tendances (AMÉLIORÉ)
            date_tendance = date_prediction - pd.Timedelta(days=30)
            donnees_tendance = self.df_historique[
                (self.df_historique['DATE'] <= date_prediction) & 
                (self.df_historique['DATE'] > date_tendance)
            ]
            
            if etablissement:
                trend_data_etb = donnees_tendance[donnees_tendance['ETBDES'] == etablissement]['QUANTITE']
                prediction_data['trend_ETBDES'] = trend_data_etb.diff().mean() if len(trend_data_etb) > 1 else 0
            else:
                prediction_data['trend_ETBDES'] = donnees_tendance['QUANTITE'].diff().mean()
                
            if article:
                trend_data_art = donnees_tendance[donnees_tendance['ARTDES'] == article]['QUANTITE']
                prediction_data['trend_ARTDES'] = trend_data_art.diff().mean() if len(trend_data_art) > 1 else 0
            else:
                prediction_data['trend_ARTDES'] = donnees_tendance['QUANTITE'].diff().mean()
            
            # 8. Vérification des features
            expected_features = [
                'année', 'mois', 'jour', 'jour_semaine', 'trimestre',
                'mois_sin', 'mois_cos', 'jour_sin', 'jour_cos',
                'jours_depuis_debut', 'tendance_normalisee',
                'moving_avg_1_ETBDES', 'moving_std_1_ETBDES',
                'moving_avg_1_ARTDES', 'moving_std_1_ARTDES',
                'moving_avg_7_ETBDES', 'moving_std_7_ETBDES',
                'moving_avg_7_ARTDES', 'moving_std_7_ARTDES',
                'moving_avg_14_ETBDES', 'moving_std_14_ETBDES',
                'moving_avg_14_ARTDES', 'moving_std_14_ARTDES',
                'moving_avg_30_ETBDES', 'moving_std_30_ETBDES',
                'moving_avg_30_ARTDES', 'moving_std_30_ARTDES',
                'trend_ETBDES', 'trend_ARTDES',
                'ETBDES_encoded', 'ARTDES_encoded', 'PTLDES_encoded'
            ]
            
            # 9. Réorganiser les colonnes
            prediction_data = prediction_data[expected_features]
            
            return prediction_data
                
        except Exception as e:
            logging.error(f"❌ Erreur lors de la préparation des features: {str(e)}")
            raise e

    def predire(self, date_prediction, etablissement=None, article=None):
        """
        Prédit la quantité de commande pour une date donnée avec options d'établissement et d'article
        
        Args:
            date_prediction (str ou datetime): Date pour laquelle faire la prédiction
            etablissement (str, optional): Nom de l'établissement
            article (str, optional): Nom de l'article
            
        Returns:
            dict: Résultats de la prédiction avec intervalles de confiance
        """
        try:
            # Convertir la date si nécessaire
            if isinstance(date_prediction, str):
                date_prediction = pd.to_datetime(date_prediction, format='%d/%m/%Y', dayfirst=True)
            
            # Préparer les features
            X_pred = self._prepare_features(date_prediction, etablissement, article)
            
            # Faire la prédiction
            pred_scaled = self.model.predict(X_pred)[0]
            
            # Dénormaliser la prédiction
            prediction = float(self.scaler.inverse_transform([[pred_scaled]])[0][0])
            
            # Calculer l'intervalle de confiance (simplifié)
            confidence_interval = {
                'lower': max(0, prediction * 0.9),  # -10%
                'upper': prediction * 1.1  # +10%
            }
            
            # Préparer le résultat
            resultat = {
                'date': date_prediction.strftime('%Y-%m-%d'),
                'prediction': round(prediction, 2),
                'intervalle_confiance': {
                    'min': round(confidence_interval['lower'], 2),
                    'max': round(confidence_interval['upper'], 2)
                }
            }
            
            if etablissement:
                resultat['etablissement'] = etablissement
            if article:
                resultat['article'] = article
                
            return resultat
            
        except Exception as e:
            logging.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            raise e