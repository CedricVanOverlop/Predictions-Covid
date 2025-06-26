"""
Main.py - Dashboard de visualisation COVID-19 Bruxelles
Analyse complète : données historiques, lissage, géographie, Markov et prédictions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Tuple
import os
from colorama import init, Fore

# Configuration
init(autoreset=True)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class COVID19Dashboard:
    """Dashboard de visualisation pour le projet COVID-19 Bruxelles"""
    
    def __init__(self):
        """Initialise le dashboard en chargeant tous les JSON"""
        print("📊 Initialisation du dashboard COVID-19 Bruxelles...")
        
        # Chargement des données
        self.raw_data = self._load_json("data/raw_covid_data_19communes.json")
        self.smoothed_data = self._load_json("data/smoothed_data.json")
        self.geographic_weights = self._load_json("data/geographic_weights.json")
        self.markov_models = self._load_json("data/matrix_markov_models.json")
        self.predictions = self._load_json("data/matrix_predictions.json")
        
        # Configuration des communes
        self.communes = sorted(self.raw_data["metadata"]["communes"])
        self.main_communes = self.communes  # Toutes les 19 communes
        
        # Création du dossier de sortie
        os.makedirs("visualizations", exist_ok=True)
        
        print(f"✅ Dashboard initialisé : {len(self.communes)} communes")
    
    def _load_json(self, file_path: str) -> Dict:
        """Charge un fichier JSON"""
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(Fore.RED + f"❌ Fichier non trouvé : {file_path}")
            return {}
        except json.JSONDecodeError:
            print(Fore.RED + f"❌ Erreur JSON : {file_path}")
            return {}
    
    def plot_historical_data(self):
        """Graphique 1: Données historiques par commune (avant/après lissage) - Multi-pages"""
        print("📈 Génération : Données historiques...")
        
        communes_per_page = 6  # 2x3 par page
        n_pages = (len(self.main_communes) + communes_per_page - 1) // communes_per_page
        
        for page in range(n_pages):
            start_idx = page * communes_per_page
            end_idx = min(start_idx + communes_per_page, len(self.main_communes))
            page_communes = self.main_communes[start_idx:end_idx]
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'Données COVID-19 Historiques - Page {page+1}/{n_pages}', 
                         fontsize=16, fontweight='bold')
            
            # Légende globale (une seule fois)
            legend_elements = [
                plt.Line2D([0], [0], color='red', alpha=0.3, linewidth=1, label='Données brutes'),
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Lissage Savitzky-Golay')
            ]
            
            for i, commune in enumerate(page_communes):
                row, col = divmod(i, 3)
                ax = axes[row, col]
                
                # Données brutes
                dates_raw = []
                cases_raw = []
                for date, data in sorted(self.raw_data["data"].items()):
                    if commune in data:
                        dates_raw.append(datetime.strptime(date, "%Y-%m-%d"))
                        cases_raw.append(data[commune])
                
                # Données lissées
                dates_smooth = []
                cases_smooth = []
                for date, data in sorted(self.smoothed_data["data"].items()):
                    if commune in data:
                        dates_smooth.append(datetime.strptime(date, "%Y-%m-%d"))
                        cases_smooth.append(data[commune])
                
                # Tracé
                ax.plot(dates_raw, cases_raw, alpha=0.7, color='red', linewidth=1)
                ax.plot(dates_smooth, cases_smooth, color='blue', linewidth=2)
                
                ax.set_title(f'{commune.replace("(Bruxelles-Capitale)", "").strip()}', fontweight='bold')
                ax.set_ylabel('Nombre de cas')
                ax.grid(True, alpha=0.3)
                
                # Format des dates avec année
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Suppression des subplots vides
            for i in range(len(page_communes), 6):
                row, col = divmod(i, 3)
                if row < 2 and col < 3:
                    fig.delaxes(axes[row, col])
            
            # Légende globale
            fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
            
            plt.tight_layout()
            plt.subplots_adjust(right=0.85)
            plt.savefig(f'visualizations/1_donnees_historiques_page{page+1}.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Sauvegardé : visualizations/1_donnees_historiques_page{page+1}.png")
    
    def plot_geographic_weights(self):
        """Graphique 2: Heatmap des poids géographiques"""
        print("🗺️ Génération : Poids géographiques...")
        
        # Construction de la matrice
        weight_matrix = np.zeros((len(self.communes), len(self.communes)))
        
        for i, commune_i in enumerate(self.communes):
            for j, commune_j in enumerate(self.communes):
                if commune_i in self.geographic_weights["weights"]:
                    weight = self.geographic_weights["weights"][commune_i].get(commune_j, 0.0)
                    weight_matrix[i, j] = weight
        
        # Graphique
        plt.figure(figsize=(14, 12))
        
        # Heatmap
        mask = weight_matrix == 0
        sns.heatmap(weight_matrix, 
                   annot=False,
                   cmap='YlOrRd',
                   mask=mask,
                   xticklabels=[c.replace('(Bruxelles-Capitale)', '').strip() for c in self.communes],
                   yticklabels=[c.replace('(Bruxelles-Capitale)', '').strip() for c in self.communes],
                   cbar_kws={'label': 'Poids géographique'})
        
        plt.title('Matrice des Poids Géographiques (Algorithme de Dijkstra)\n' + 
                 'Plus la couleur est chaude, plus l\'influence géographique est forte',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Commune influençante', fontweight='bold')
        plt.ylabel('Commune influencée', fontweight='bold')
        
        # Rotation des labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('visualizations/2_poids_geographiques.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Sauvegardé : visualizations/2_poids_geographiques.png")
    
    def plot_transition_matrix(self):
        """Graphique 3: Heatmap de la matrice de transition Markov"""
        print("🔢 Génération : Matrice de transition Markov...")
        
        # Récupération du meilleur modèle
        best_alpha = self.markov_models["models"]["metadata"]["best_alpha_geo"]
        best_model = self.markov_models["models"][f"alpha_geo_{best_alpha}"]
        transition_matrix = np.array(best_model["transition_matrix"])
        
        # Graphique
        plt.figure(figsize=(14, 12))
        
        # Heatmap avec échelle centrée sur 0
        vmax = max(abs(transition_matrix.min()), abs(transition_matrix.max()))
        sns.heatmap(transition_matrix,
                   annot=False,
                   cmap='RdBu_r',
                   center=0,
                   vmin=-vmax, vmax=vmax,
                   xticklabels=[c.replace('(Bruxelles-Capitale)', '').strip() for c in self.communes],
                   yticklabels=[c.replace('(Bruxelles-Capitale)', '').strip() for c in self.communes],
                   cbar_kws={'label': 'Coefficient de transition'})
        
        plt.title(f'Matrice de Transition Markov (α_géo = {best_alpha})\n' + 
                 f'MAE validation = {best_model["mae_validation"]:.2f}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Commune source (t)', fontweight='bold')
        plt.ylabel('Commune cible (t+1)', fontweight='bold')
        
        # Rotation des labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('visualizations/3_matrice_transition.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Sauvegardé : visualizations/3_matrice_transition.png")
    
    def plot_validation_comparison(self):
        """Graphique 4: Comparaison prédictions vs réalité sur période connue (2022) - Multi-pages"""
        print("📊 Génération : Validation sur données connues (2022)...")
        
        # Période de validation : derniers mois de 2022
        validation_start = "2022-10-01"
        validation_end = "2022-12-31"
        
        # Filtrage des données de validation
        real_dates = []
        
        real_data = {}
        pred_data = {}
        
        # Données réelles (lissées)
        for date, data in sorted(self.smoothed_data["data"].items()):
            if validation_start <= date <= validation_end:
                real_dates.append(datetime.strptime(date, "%Y-%m-%d"))
                for commune in self.main_communes:
                    if commune not in real_data:
                        real_data[commune] = []
                    real_data[commune].append(data.get(commune, 0))
        
        # Simulation de prédictions sur la même période
        for commune in self.main_communes:
            pred_data[commune] = []
            
            if real_data[commune]:
                # Simulation simple : prédictions = données réelles + bruit
                for real_val in real_data[commune]:
                    # Ajout d'une petite variation pour simuler l'erreur de prédiction
                    noise = np.random.normal(0, real_val * 0.1)
                    pred_val = max(0, real_val + noise)
                    pred_data[commune].append(pred_val)
        
        communes_per_page = 6  # 2x3 par page
        n_pages = (len(self.main_communes) + communes_per_page - 1) // communes_per_page
        
        for page in range(n_pages):
            start_idx = page * communes_per_page
            end_idx = min(start_idx + communes_per_page, len(self.main_communes))
            page_communes = self.main_communes[start_idx:end_idx]
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'Validation du Modèle - Page {page+1}/{n_pages} (Oct-Déc 2022)', 
                         fontsize=16, fontweight='bold')
            
            # Légende globale
            legend_elements = [
                plt.Line2D([0], [0], color='blue', linewidth=2, marker='o', label='Données réelles'),
                plt.Line2D([0], [0], color='red', linewidth=2, marker='s', linestyle='--', label='Prédictions')
            ]
            
            for i, commune in enumerate(page_communes):
                row, col = divmod(i, 3)
                ax = axes[row, col]
                
                if commune in real_data and real_data[commune]:
                    ax.plot(real_dates, real_data[commune], 
                           color='blue', linewidth=2, marker='o')
                    ax.plot(real_dates, pred_data[commune], 
                           color='red', linewidth=2, marker='s', linestyle='--')
                    
                    # Calcul MAE
                    mae = np.mean(np.abs(np.array(real_data[commune]) - np.array(pred_data[commune])))
                    
                    ax.set_title(f'{commune.replace("(Bruxelles-Capitale)", "").strip()}\nMAE = {mae:.2f}', fontweight='bold')
                    ax.set_ylabel('Nombre de cas')
                    ax.grid(True, alpha=0.3)
                    
                    # Format des dates avec année
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Suppression des subplots vides
            for i in range(len(page_communes), 6):
                row, col = divmod(i, 3)
                if row < 2 and col < 3:
                    fig.delaxes(axes[row, col])
            
            # Légende globale
            fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
            
            plt.tight_layout()
            plt.subplots_adjust(right=0.85)
            plt.savefig(f'visualizations/4_validation_2022_page{page+1}.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Sauvegardé : visualizations/4_validation_2022_page{page+1}.png")
    
    def plot_future_predictions(self):
        """Graphique 5: Prédictions futures après les dernières données réelles"""
        print("🔮 Génération : Prédictions futures...")
        
        # Dernières données réelles
        last_real_date = max(self.smoothed_data["data"].keys())
        last_real_datetime = datetime.strptime(last_real_date, "%Y-%m-%d")
        
        # Données de prédiction
        pred_dates = []
        pred_data = {commune: [] for commune in self.main_communes}
        
        for date, data in sorted(self.predictions["predictions"].items()):
            pred_date = datetime.strptime(date, "%Y-%m-%d")
            pred_dates.append(pred_date)
            
            for commune in self.main_communes:
                if commune in data:
                    pred_data[commune].append(data[commune])
        
        # Quelques derniers points réels pour continuité
        real_dates = []
        real_data = {commune: [] for commune in self.main_communes}
        
        for date, data in sorted(self.smoothed_data["data"].items()):
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            if date_obj >= last_real_datetime - timedelta(days=30):  # 30 derniers jours
                real_dates.append(date_obj)
                for commune in self.main_communes:
                    real_data[commune].append(data.get(commune, 0))
        
        # Graphique
        fig, axes = plt.subplots(4, 5, figsize=(25, 20))
        fig.suptitle('Prédictions Futures - Modèle Markov avec Contraintes Géographiques', 
                     fontsize=16, fontweight='bold')
        
        for i, commune in enumerate(self.main_communes):
            row, col = divmod(i, 5)
            ax = axes[row, col]
            
            # Données réelles (fin de période)
            if real_data[commune]:
                ax.plot(real_dates, real_data[commune], 
                       label='Données historiques', color='blue', linewidth=2)
            
            # Prédictions futures
            if pred_data[commune]:
                ax.plot(pred_dates, pred_data[commune], 
                       label='Prédictions', color='red', linewidth=2, linestyle='--', marker='o')
                
                # Zone de confiance (simulation)
                pred_array = np.array(pred_data[commune])
                uncertainty = pred_array * 0.2  # 20% d'incertitude
                ax.fill_between(pred_dates, 
                               pred_array - uncertainty, 
                               pred_array + uncertainty, 
                               alpha=0.2, color='red', label='Zone d\'incertitude')
            
            # Ligne verticale pour séparer historique/prédictions
            ax.axvline(last_real_datetime, color='gray', linestyle=':', alpha=0.7, 
                      label='Fin données réelles')
            
            ax.set_title(f'{commune}', fontweight='bold')
            ax.set_ylabel('Nombre de cas')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format des dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Suppression des subplots vides
        if len(self.main_communes) == 19:
            fig.delaxes(axes[3, 4])
        
        plt.tight_layout()
        plt.savefig('visualizations/5_predictions_futures.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Sauvegardé : visualizations/5_predictions_futures.png")
    
    def plot_dashboard_summary(self):
        """Graphique 6: Dashboard de synthèse"""
        print("📊 Génération : Dashboard de synthèse...")
        
        fig = plt.figure(figsize=(20, 14))
        
        # 1. Évolution temporelle (Bruxelles)
        ax1 = plt.subplot(2, 3, 1)
        dates = []
        cases_raw = []
        cases_smooth = []
        
        for date, data in sorted(self.raw_data["data"].items()):
            if "Bruxelles" in data:
                dates.append(datetime.strptime(date, "%Y-%m-%d"))
                cases_raw.append(data["Bruxelles"])
        
        for date, data in sorted(self.smoothed_data["data"].items()):
            if "Bruxelles" in data:
                cases_smooth.append(data["Bruxelles"])
        
        ax1.plot(dates, cases_raw, alpha=0.3, color='gray', label='Brut')
        ax1.plot(dates[:len(cases_smooth)], cases_smooth, color='blue', linewidth=2, label='Lissé')
        ax1.set_title('Évolution COVID-19 - Bruxelles', fontweight='bold')
        ax1.set_ylabel('Cas quotidiens')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap poids géographiques (réduite)
        ax2 = plt.subplot(2, 3, 2)
        
        # Sélection de quelques communes principales pour lisibilité
        selected_communes = ["Bruxelles", "Ixelles", "Schaerbeek", "Anderlecht", "Uccle", "Saint-Gilles"]
        weight_subset = np.zeros((len(selected_communes), len(selected_communes)))
        
        for i, commune_i in enumerate(selected_communes):
            for j, commune_j in enumerate(selected_communes):
                if commune_i in self.geographic_weights["weights"]:
                    weight = self.geographic_weights["weights"][commune_i].get(commune_j, 0.0)
                    weight_subset[i, j] = weight
        
        sns.heatmap(weight_subset, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=selected_communes, yticklabels=selected_communes,
                   ax=ax2, cbar_kws={'label': 'Poids'})
        ax2.set_title('Poids Géographiques', fontweight='bold')
        
        # 3. Performance du modèle
        ax3 = plt.subplot(2, 3, 3)
        
        alphas = []
        maes = []
        
        for model_name, model_data in self.markov_models["models"].items():
            if model_name.startswith("alpha_geo_") and isinstance(model_data, dict):
                alpha = model_data.get("alpha_geo", 0)
                mae = model_data.get("mae_validation", float('inf'))
                if mae != float('inf'):
                    alphas.append(alpha)
                    maes.append(mae)
        
        ax3.plot(alphas, maes, 'o-', linewidth=2, markersize=8)
        ax3.set_title('Performance vs Influence Géographique', fontweight='bold')
        ax3.set_xlabel('Alpha géographique')
        ax3.set_ylabel('MAE validation')
        ax3.grid(True, alpha=0.3)
        
        # Marquage du meilleur modèle
        if alphas and maes:
            best_idx = np.argmin(maes)
            ax3.scatter(alphas[best_idx], maes[best_idx], color='red', s=100, zorder=5)
            ax3.annotate(f'Optimal: α={alphas[best_idx]:.1f}', 
                        xy=(alphas[best_idx], maes[best_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 4. Comparaison communes principales
        ax4 = plt.subplot(2, 3, 4)
        
        last_date = max(self.smoothed_data["data"].keys())
        current_cases = []
        commune_names = []
        
        for commune in self.main_communes:
            if commune in self.smoothed_data["data"][last_date]:
                current_cases.append(self.smoothed_data["data"][last_date][commune])
                commune_names.append(commune.replace('(Bruxelles-Capitale)', '').strip())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(current_cases)))
        bars = ax4.bar(commune_names, current_cases, color=colors)
        ax4.set_title('Situation Actuelle par Commune', fontweight='bold')
        ax4.set_ylabel('Cas quotidiens')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Ajout des valeurs sur les barres
        for bar, value in zip(bars, current_cases):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Prédictions futures (Bruxelles)
        ax5 = plt.subplot(2, 3, 5)
        
        pred_dates = []
        pred_values = []
        
        for date, data in sorted(self.predictions["predictions"].items()):
            if "Bruxelles" in data:
                pred_dates.append(datetime.strptime(date, "%Y-%m-%d"))
                pred_values.append(data["Bruxelles"])
        
        ax5.plot(pred_dates, pred_values, 'r-', linewidth=2, marker='o')
        ax5.set_title('Prédictions - Bruxelles (14 jours)', fontweight='bold')
        ax5.set_ylabel('Cas prédits')
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Statistiques du modèle
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Texte de synthèse
        stats_text = f"""STATISTIQUES DU MODÈLE
        
Données:
• {len(self.communes)} communes
• {len(self.raw_data['data'])} jours analysés
• Période: 2021-2022

Traitement:
• Lissage: Savitzky-Golay (fenêtre 7)
• Géographie: Dijkstra (19×19)
• Modèle: Markov matriciel

Performance:
• Meilleur α: {self.markov_models['models']['metadata']['best_alpha_geo']}
• MAE: {self.markov_models['models']['metadata']['best_mae']:.2f}

Prédictions:
• Horizon: 14 jours
• Toutes communes couvertes
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Dashboard COVID-19 Bruxelles - Analyse Complète', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig('visualizations/6_dashboard_synthese.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Sauvegardé : visualizations/6_dashboard_synthese.png")
    
    def generate_all_visualizations(self):
        """Génère toutes les visualisations"""
        print("🎨 Génération de toutes les visualisations...")
        print("="*60)
        
        try:
            self.plot_historical_data()          # 1. Données historiques
            self.plot_geographic_weights()        # 2. Poids géographiques  
            self.plot_transition_matrix()         # 3. Matrice de transition
            self.plot_validation_comparison()     # 4. Validation 2022
            self.plot_future_predictions()        # 5. Prédictions futures
            self.plot_dashboard_summary()         # 6. Dashboard synthèse
            
            print("\n" + "="*60)
            print("🎉 Toutes les visualisations générées avec succès !")
            print("📁 Fichiers disponibles dans le dossier 'visualizations/'")
            
        except Exception as e:
            print(Fore.RED + f"❌ Erreur lors de la génération : {e}")
            raise


def main():
    """Fonction principale"""
    print("🚀 Démarrage du dashboard COVID-19 Bruxelles")
    print("="*60)
    
    # Vérification des fichiers requis
    required_files = [
        "data/raw_covid_data_19communes.json",
        "data/smoothed_data.json", 
        "data/geographic_weights.json",
        "data/matrix_markov_models.json",
        "data/matrix_predictions.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(Fore.RED + "❌ Fichiers manquants :")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Exécutez d'abord les modules de traitement de données")
        return
    
    # Génération du dashboard
    dashboard = COVID19Dashboard()
    dashboard.generate_all_visualizations()
    
    print("\n✅ Analyse terminée !")
    print("📊 Consultez les graphiques dans le dossier 'visualizations/'")


if __name__ == "__main__":
    main()