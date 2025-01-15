import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

def setup_logging(log_file="prediction_model_v2.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.info("🚀 Début du processus v2")

def split_temporel(df, train_ratio=0.8, val_ratio=0.1):
    """Split temporel avec une fenêtre plus courte"""
    total_rows = len(df)
    train_idx = int(total_rows * train_ratio)
    val_idx = int(total_rows * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    logging.info("📊 Split des données effectué (nouveau ratio):")
    logging.info(f"Train: {len(train_df)} lignes ({train_ratio*100}%)")
    logging.info(f"Validation: {len(val_df)} lignes ({val_ratio*100}%)")
    logging.info(f"Test: {len(test_df)} lignes ({(1-train_ratio-val_ratio)*100}%)")
    
    return train_df, val_df, test_df

def create_temporal_features(df):
    df = df.copy()
    
    # Features temporelles basiques
    df['année'] = df['DATE'].dt.year
    df['mois'] = df['DATE'].dt.month
    df['jour'] = df['DATE'].dt.day
    df['jour_semaine'] = df['DATE'].dt.dayofweek
    df['trimestre'] = df['DATE'].dt.quarter
    
    # Features cycliques améliorées
    df['mois_sin'] = np.sin(2 * np.pi * df['mois']/12)
    df['mois_cos'] = np.cos(2 * np.pi * df['mois']/12)
    df['jour_sin'] = np.sin(2 * np.pi * df['jour']/31)
    df['jour_cos'] = np.cos(2 * np.pi * df['jour']/31)
    
    # Features de tendance
    df['jours_depuis_debut'] = (df['DATE'] - df['DATE'].min()).dt.days
    df['tendance_normalisee'] = (df['jours_depuis_debut'] - df['jours_depuis_debut'].mean()) / df['jours_depuis_debut'].std()
    
    return df

def create_lag_features(df, group_cols, target_col, lags=[1, 7, 14, 30]):
    df = df.copy()
    
    # Features de lag standards
    for lag in lags:
        for col in group_cols:
            df[f'moving_avg_{lag}_{col}'] = df.groupby(col)[target_col].transform(
                lambda x: x.rolling(window=lag, min_periods=1).mean()
            )
            # Ajout d'écart-type mobile
            df[f'moving_std_{lag}_{col}'] = df.groupby(col)[target_col].transform(
                lambda x: x.rolling(window=lag, min_periods=1).std()
            )
    
    # Features de tendance par groupe
    for col in group_cols:
        df[f'trend_{col}'] = df.groupby(col)[target_col].transform(
            lambda x: (x - x.rolling(window=7, min_periods=1).mean()) / x.rolling(window=7, min_periods=1).std()
        )
    
    return df

def prepare_features(df, scaler=None, is_training=True):
    logging.info("📊 Préparation des features améliorée...")
    
    df = create_temporal_features(df)
    group_cols = ['ETBDES', 'ARTDES']
    df = create_lag_features(df, group_cols, 'QUANTITE')
    
    # Encodage des variables catégorielles
    for col in group_cols:
        df[f'{col}_encoded'] = pd.factorize(df[col])[0]
    
    if 'PTLDES' in df.columns:
        df['PTLDES_encoded'] = pd.factorize(df['PTLDES'])[0]
    
    # Normalisation de la cible
    if is_training:
        scaler = RobustScaler()
        df['QUANTITE_SCALED'] = scaler.fit_transform(df[['QUANTITE']])
    else:
        df['QUANTITE_SCALED'] = scaler.transform(df[['QUANTITE']])
    
    return df, scaler

def train_predict_model(train_df, val_df, test_df, feature_cols, target_col='QUANTITE_SCALED'):
    logging.info("🎯 Entraînement du modèle amélioré...")
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Configuration améliorée de LightGBM
    model = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.01,  # Learning rate réduit
        n_estimators=2000,   # Plus d'estimateurs
        subsample=0.8,       # Sous-échantillonnage pour réduire l'overfitting
        colsample_bytree=0.8,
        reg_alpha=0.1,       # Régularisation L1
        reg_lambda=0.1,      # Régularisation L2
    )

    # Entraînement avec early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100)],
        eval_metric='rmse'
    )

    # Prédictions et dénormalisation
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calcul des métriques sur données normalisées
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    logging.info(f"📈 Validation - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
    logging.info(f"📈 Test - RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")

    # Analyse des features importantes
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info("\n🔍 Features les plus importantes:")
    logging.info(feature_importance.head(10))

    return model, feature_importance

def main():
    setup_logging()
    try:
        # Chargement des données
        logging.info("📖 Chargement des données...")
        df = pd.read_csv("Final_data/donnees_completes_logistique_2022-2024.csv")
        
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        if df['DATE'].isnull().any():
            df = df[df['DATE'].notnull()]
        
        df = df.sort_values('DATE')
        
        # Split avec nouveau ratio
        train_df, val_df, test_df = split_temporel(df, train_ratio=0.8, val_ratio=0.1)
        
        # Préparation des features avec normalisation
        train_df, scaler = prepare_features(train_df, is_training=True)
        val_df, _ = prepare_features(val_df, scaler=scaler, is_training=False)
        test_df, _ = prepare_features(test_df, scaler=scaler, is_training=False)
        
        feature_cols = [col for col in train_df.columns if col not in 
                       ['DATE', 'QUANTITE', 'ETBDES', 'ARTDES', 'PTLDES', 'QUANTITE_SCALED']]
        
        # Entraînement et évaluation
        model, feature_importance = train_predict_model(train_df, val_df, test_df, feature_cols)
        
        # Sauvegarde du modèle et du scaler
        import joblib
        joblib.dump(model, 'model_v2.joblib')
        joblib.dump(scaler, 'scaler_v2.joblib')
        
        logging.info("✨ Processus terminé avec succès!")
        
    except Exception as e:
        logging.error(f"❌ Erreur: {str(e)}")
        raise e

if __name__ == "__main__":
    main()