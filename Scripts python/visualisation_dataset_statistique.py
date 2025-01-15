import pandas as pd
import matplotlib.pyplot as plt

def creer_visualisations():
    try:
        print("📊 Début de la création des visualisations...")
        
        # 1. Lecture des fichiers CSV
        print("📖 Lecture des fichiers de données...")
        stats_temp = pd.read_csv('Statistiques_Dataset/statistiques_temporelles.csv')
        stats_etab = pd.read_csv('Statistiques_Dataset/statistiques_etablissements.csv')
        stats_art = pd.read_csv('Statistiques_Dataset/statistiques_articles.csv')
        
        # Configuration générale des graphiques
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = [15, 8]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
        
        # 2. Évolution temporelle
        print("📈 Création du graphique d'évolution temporelle...")
        fig, ax = plt.subplots()
        ax.plot(range(len(stats_temp['mois'])), stats_temp['sum'], 
                marker='o', linewidth=2, color='#1f77b4',
                markersize=6, markerfacecolor='white')
        ax.set_title('Évolution des commandes dans le temps', pad=20, fontsize=12, fontweight='bold')
        ax.set_xlabel('Mois', fontsize=10)
        ax.set_ylabel('Quantité totale', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Alignement des étiquettes
        ax.set_xticks(range(len(stats_temp['mois'])))
        ax.set_xticklabels(stats_temp['mois'], rotation=45, ha='center', va='top')
        
        # Ajout des valeurs sur les points
        """
        for i, val in enumerate(stats_temp['sum']):
            ax.text(i, val, f'{val:,.0f}', ha='center', va='bottom', fontsize=8)
        """    

        plt.tight_layout()
        plt.savefig('evolution_temporelle.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 3. Top 10 établissements
        print("📊 Création du graphique des top établissements...")
        fig, ax = plt.subplots()
        top_10_etab = stats_etab.nlargest(10, 'sum')
        bars = ax.bar(range(len(top_10_etab)), top_10_etab['sum'], 
                     color='skyblue', align='center', alpha=0.8)
        ax.set_title('Top 10 des établissements par volume', pad=20, fontsize=12, fontweight='bold')
        ax.set_xlabel('Établissements', fontsize=10)
        ax.set_ylabel('Quantité totale', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Alignement des étiquettes
        ax.set_xticks(range(len(top_10_etab)))
        ax.set_xticklabels(top_10_etab['ETBDES'], rotation=45, ha='right')
        
        # Ajout des valeurs sur les barres
        """
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom', fontsize=8)
        """
        plt.tight_layout()
        plt.savefig('top_10_etablissements.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 4. Top 10 articles
        print("📊 Création du graphique des top articles...")
        fig, ax = plt.subplots()
        top_10_art = stats_art.nlargest(10, 'sum')
        bars = ax.bar(range(len(top_10_art)), top_10_art['sum'], 
                     color='lightgreen', align='center', alpha=0.8)
        ax.set_title('Top 10 des articles commandés', pad=20, fontsize=12, fontweight='bold')
        ax.set_xlabel('Articles', fontsize=10)
        ax.set_ylabel('Quantité totale', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Alignement des étiquettes
        ax.set_xticks(range(len(top_10_art)))
        ax.set_xticklabels(top_10_art['ARTDES'], rotation=45, ha='right')
        
        # Ajout des valeurs sur les barres
        """
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom', fontsize=8)
        """
        plt.tight_layout()
        plt.savefig('top_10_articles.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print("\n✅ Visualisations générées avec succès!")
        print("📂 Fichiers créés:")
        print("- evolution_temporelle.png")
        print("- top_10_etablissements.png")
        print("- top_10_articles.png")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des visualisations: {str(e)}")
        print(f"💡 Détails de l'erreur pour le débogage: {type(e).__name__}")

if __name__ == "__main__":
    print("🚀 Démarrage du programme de visualisation...")
    creer_visualisations()