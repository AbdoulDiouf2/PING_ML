#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from predicteur_commandes import PredicteurCommandes

def executer_tests_prediction_future():
    """Exécute les tests de prédiction pour une période future avec comparaison historique"""
    
    print("Démarrage des tests de prédiction future...")
    print("="*50)

    # 1. Chargement du prédicteur
    try:
        predicteur = PredicteurCommandes()
        print("✅ Prédicteur chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du prédicteur: {str(e)}")
        return

    # 2. Chargement des données historiques
    try:
        df_hist = pd.read_csv("Final_data/donnees_completes_logistique_formatted.csv", low_memory=False)
        df_hist['DATE'] = pd.to_datetime(df_hist['DATE'])
        print("✅ Données historiques chargées")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données historiques: {str(e)}")
        return

    # 3. Définition des périodes de test
    dates_prediction = pd.date_range(start='2025-01-20', end='2025-01-24', freq='D')
    dates_2024 = pd.date_range(start='2024-01-20', end='2024-01-24', freq='D')
    dates_2023 = pd.date_range(start='2023-01-20', end='2023-01-24', freq='D')

    # 4. Sélection des données de test
    article_test = df_hist['ARTDES'].value_counts().index[0]
    etb_test = df_hist['ETBDES'].value_counts().index[0]

    print(f"\nPrédiction pour la période du {dates_prediction[0].strftime('%d/%m/%Y')} au {dates_prediction[-1].strftime('%d/%m/%Y')}")
    print(f"Article testé: {article_test}")
    print(f"Établissement testé: {etb_test}")
    print("-"*50)

    # 5. Prédictions futures
    predictions = []
    for date in dates_prediction:
        pred = predicteur.predire(
            date_prediction=date,
            etablissement=etb_test,
            article=article_test
        )
        predictions.append(pred)
    
    df_predictions = pd.DataFrame(predictions)
    df_predictions['DATE'] = pd.to_datetime(df_predictions['date'])
    
    # 6. Récupération des données historiques pour comparaison
    donnees_2024 = df_hist[
        (df_hist['DATE'].isin(dates_2024)) &
        (df_hist['ETBDES'] == etb_test) &
        (df_hist['ARTDES'] == article_test)
    ].groupby('DATE')['QUANTITE'].sum().reset_index()

    donnees_2023 = df_hist[
        (df_hist['DATE'].isin(dates_2023)) &
        (df_hist['ETBDES'] == etb_test) &
        (df_hist['ARTDES'] == article_test)
    ].groupby('DATE')['QUANTITE'].sum().reset_index()

    # 7. Affichage des résultats
    print("\nPrédictions 2025:")
    print(df_predictions[['date', 'prediction']].to_string(index=False))
    
    print("\nDonnées 2024:")
    print(donnees_2024.to_string(index=False))
    
    print("\nDonnées 2023:")
    print(donnees_2023.to_string(index=False))

    # 8. Visualisation
    plt.figure(figsize=(12, 6))
    
    # Prédictions 2025
    plt.plot(range(5), df_predictions['prediction'], 
             marker='o', label='Prédictions 2025', 
             color='#2ecc71', linewidth=2)
    
    # Données 2024
    if not donnees_2024.empty:
        plt.plot(range(5), donnees_2024['QUANTITE'], 
                marker='s', label='Réel 2024', 
                color='#e74c3c', linewidth=2)
    
    # Données 2023
    if not donnees_2023.empty:
        plt.plot(range(5), donnees_2023['QUANTITE'], 
                marker='^', label='Réel 2023', 
                color='#3498db', linewidth=2)

    plt.title(f'Comparaison des prédictions futures avec l\'historique\n{article_test} - {etb_test}')
    plt.xlabel('Jours')
    plt.ylabel('Quantité')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(5), ['J1', 'J2', 'J3', 'J4', 'J5'])
    plt.tight_layout()
    plt.show()

    # 9. Analyse des variations
    if not donnees_2024.empty:
        variation_2024 = ((df_predictions['prediction'].mean() - donnees_2024['QUANTITE'].mean()) 
                         / donnees_2024['QUANTITE'].mean() * 100)
        print(f"\nVariation moyenne par rapport à 2024: {variation_2024:.2f}%")
    
    if not donnees_2023.empty:
        variation_2023 = ((df_predictions['prediction'].mean() - donnees_2023['QUANTITE'].mean()) 
                         / donnees_2023['QUANTITE'].mean() * 100)
        print(f"Variation moyenne par rapport à 2023: {variation_2023:.2f}%")

    print("\nTests terminés avec succès! ✨")

if __name__ == "__main__":
    executer_tests_prediction_future()