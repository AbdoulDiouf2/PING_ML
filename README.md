# 🚀 Prédicteur de Commandes Logistiques

## 📋 Description
Ce projet implémente un système de prédiction des commandes logistiques basé sur un modèle LightGBM. Il permet de prédire les quantités de commandes futures en fonction de différents paramètres (établissement, article, date) en utilisant l'historique des données.

## 🔧 Installation

### Prérequis
- Python 3.8+
- pip ou conda

### Dépendances
```bash
pip install -r requirements.txt
```

Les principales dépendances incluent :
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn
- joblib

### Structure des fichiers
```
├── Final_data/
│   ├── donnees_completes_logistique_2022-2024.csv
│   └── donnees_completes_logistique_formatted.csv
├── model_v2.joblib
├── scaler_v2.joblib
├── split_and_train_model.py
├── predicteur_commandes.py
└── test_predicteur_commandes.py
```

## 🎯 Utilisation

### 1. Préparation des données
Les données doivent être au format CSV avec les colonnes suivantes :
- DATE : Date de la commande (format DD/MM/YYYY)
- ETBDES : Nom de l'établissement
- ARTDES : Nom de l'article
- QUANTITE : Quantité commandée

### 2. Entraînement du modèle
Pour entraîner un nouveau modèle :
```bash
python split_and_train_model.py
```
Cela générera :
- `model_v2.joblib` : Le modèle entraîné
- `scaler_v2.joblib` : Le scaler pour la normalisation des données

### 3. Utilisation du prédicteur
Pour utiliser le prédicteur dans votre code :
```python
from predicteur_commandes import PredicteurCommandes

# Initialiser le prédicteur
predicteur = PredicteurCommandes()

# Faire une prédiction
prediction = predicteur.predire(
    date_prediction="01/02/2024",
    etablissement="ETB1",
    article="ART1"
)

print(prediction)
```

### 4. Tests et validation
Pour exécuter les tests et visualiser les résultats :
```bash
python test_predicteur_commandes.py
```

## 📊 Fonctionnalités

### Prédictions disponibles
1. **Prédiction globale**
   - Prévision de la quantité totale de commandes
   
2. **Prédiction par établissement**
   - Prévision spécifique pour un établissement donné
   
3. **Prédiction par article**
   - Prévision spécifique pour un article donné
   
4. **Prédiction combinée**
   - Prévision pour un couple établissement-article spécifique

### Caractéristiques du modèle
- Features temporelles (année, mois, jour, jour de la semaine)
- Caractéristiques cycliques (saisonnalité)
- Moyennes mobiles et tendances
- Intervalle de confiance pour chaque prédiction

## 📈 Performances

Le modèle est évalué sur plusieurs métriques :
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

Les résultats détaillés sont disponibles dans les logs d'entraînement.

## 🛠 Maintenance et mise à jour

### Mise à jour du modèle
Pour mettre à jour le modèle avec de nouvelles données :
1. Mettre à jour le fichier CSV dans `Final_data/`
2. Réexécuter `split_and_train_model.py`

### Format des dates
Le système accepte deux formats de dates :
- Format chaîne : "DD/MM/YYYY"
- Format datetime Python

## ⚠️ Notes importantes

1. **Données historiques**
   - Les données doivent être triées chronologiquement
   - Pas de valeurs manquantes dans les colonnes clés

2. **Performance**
   - Le modèle est optimisé pour les prédictions à court terme (7-30 jours)
   - Les prédictions long terme peuvent être moins précises

3. **Mémoire**
   - Le modèle charge l'ensemble des données historiques en mémoire
   - Prévoir suffisamment de RAM selon la taille des données

## 🤝 Contribution
Pour contribuer au projet :
1. Fork le repository
2. Créer une branche pour votre feature
3. Commiter vos changements
4. Pousser vers la branche
5. Créer une Pull Request