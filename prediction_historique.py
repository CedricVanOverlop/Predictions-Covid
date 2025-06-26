"""
Modification du modèle de prédiction pour validation historique
Utilise la période Mars-Juin 2021 pour comparer prédictions vs réalité
"""

import json
import os
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional
from colorama import init, Fore

# Initialisation de colorama
init(autoreset=True)


class HistoricalValidationModel:
    """
    Modèle de validation sur données historiques Mars-Juin 2021
    """
    
    def __init__(self, smoothed_data_file: str = "data/smoothed_data.json", 
                 geographic_weights_file: str = "data/geographic_weights.json"):
        """
        Initialise le modèle pour validation historique
        """
        print("🔢 Initialisation du modèle de validation historique...")
        
        # Chargement des données
        self.smoothed_data = self._load_smoothed_data(smoothed_data_file)
        self.geographic_weights = self._load_geographic_weights(geographic_weights_file)
        
        # Communes (ordre fixe pour la matrice)
        self.communes = sorted(list(self.geographic_weights.keys()))
        self.n_communes = len(self.communes)
        self.commune_to_index = {commune: i for i, commune in enumerate(self.communes)}
        
        # Périodes définies
        self.training_start = "2021-01-01"
        self.training_end = "2021-02-28"
        self.validation_start = "2021-03-01"
        self.validation_end = "2021-06-30"
        
        print(f"✅ Modèle initialisé pour validation historique")
        print(f"📅 Entraînement : {self.training_start} → {self.training_end}")
        print(f"📅 Validation : {self.validation_start} → {self.validation_end}")
    
    def _load_smoothed_data(self, file_path: str) -> Dict:
        """Charge les données lissées"""
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                data = json.load(f)
            
            if "data" in data:
                return data["data"]
            else:
                return data
                
        except FileNotFoundError:
            print(Fore.RED + f"❌ Fichier non trouvé : {file_path}")
            raise
    
    def _load_geographic_weights(self, file_path: str) -> Dict[str, Dict[str, float]]:
        """Charge les poids géographiques"""
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                data = json.load(f)
            
            if "weights" in data:
                return data["weights"]
            else:
                return data
                
        except FileNotFoundError:
            print(Fore.RED + f"❌ Fichier non trouvé : {file_path}")
            raise
    
    def extract_period_data(self, start_date: str, end_date: str) -> Tuple[List[str], np.ndarray]:
        """
        Extrait les données pour une période donnée
        
        Returns:
            dates: Liste des dates
            data_matrix: Matrice [communes × jours]
        """
        period_dates = []
        period_data = []
        
        for date in sorted(self.smoothed_data.keys()):
            if start_date <= date <= end_date:
                period_dates.append(date)
                
                # Extraction des cas pour toutes les communes à cette date
                day_data = []
                for commune in self.communes:
                    cases = self.smoothed_data[date].get(commune, 0.0)
                    day_data.append(float(cases))
                
                period_data.append(day_data)
        
        if not period_data:
            raise ValueError(f"Aucune donnée trouvée pour la période {start_date} - {end_date}")
        
        # Conversion en matrice numpy [communes × jours]
        data_matrix = np.array(period_data).T
        
        print(f"📊 Période {start_date} - {end_date} : {len(period_dates)} jours, {data_matrix.shape[0]} communes")
        
        return period_dates, data_matrix
    
    def train_model_on_period(self, alpha_geo: float = 0.5) -> np.ndarray:
        """
        Entraîne le modèle de Markov sur la période d'entraînement
        
        Args:
            alpha_geo: Paramètre de pondération géographique
            
        Returns:
            Matrice de transition [19×19]
        """
        print(f"🎯 Entraînement du modèle sur {self.training_start} - {self.training_end}...")
        
        # Extraction des données d'entraînement
        train_dates, train_matrix = self.extract_period_data(self.training_start, self.training_end)
        
        if train_matrix.shape[1] < 2:
            raise ValueError("Pas assez de données pour l'entraînement")
        
        # Préparation des matrices X(t) et X(t+1)
        X_t = train_matrix[:, :-1]   # [19 × (T-1)]
        X_t1 = train_matrix[:, 1:]   # [19 × (T-1)]
        
        print(f"📈 Matrices d'entraînement : {X_t.shape}")
        
        # Estimation de la matrice de base par moindres carrés
        try:
            XtXt_T = X_t @ X_t.T
            regularization = 1e-6 * np.eye(self.n_communes)
            XtXt_T_reg = XtXt_T + regularization
            XtXt_T_inv = np.linalg.inv(XtXt_T_reg)
            A_base = X_t1 @ X_t.T @ XtXt_T_inv
        except np.linalg.LinAlgError:
            print(Fore.YELLOW + "⚠️ Utilisation de la pseudo-inverse")
            A_base = X_t1 @ np.linalg.pinv(X_t)
        
        # Construction de la matrice géographique
        geo_matrix = np.zeros((self.n_communes, self.n_communes))
        for i, commune_i in enumerate(self.communes):
            for j, commune_j in enumerate(self.communes):
                if commune_i in self.geographic_weights:
                    weight = self.geographic_weights[commune_i].get(commune_j, 0.0)
                    geo_matrix[i, j] = weight
        
        # Application des contraintes géographiques
        A_geographic = A_base * geo_matrix
        A_final = (1 - alpha_geo) * A_base + alpha_geo * A_geographic
        
        print(f"✅ Modèle entraîné avec α_geo = {alpha_geo}")
        
        return A_final
    
    def predict_validation_period(self, transition_matrix: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Effectue des prédictions sur la période de validation
        
        Args:
            transition_matrix: Matrice de transition entraînée
            
        Returns:
            dates: Dates de validation
            real_data: Données réelles [communes × jours]
            predictions: Prédictions [communes × jours]
        """
        print(f"🔮 Prédictions sur {self.validation_start} - {self.validation_end}...")
        
        # Données réelles de validation
        val_dates, real_data = self.extract_period_data(self.validation_start, self.validation_end)
        
        # État initial : dernière observation de la période d'entraînement
        _, train_data = self.extract_period_data(self.training_start, self.training_end)
        initial_state = train_data[:, -1].reshape(-1, 1)  # Dernière colonne comme état initial
        
        print(f"🚀 État initial : {np.sum(initial_state):.1f} cas totaux")
        
        # Génération des prédictions jour par jour
        predictions = np.zeros((self.n_communes, len(val_dates)))
        current_state = initial_state.copy()
        
        for day in range(len(val_dates)):
            predictions[:, day] = current_state.flatten()
            
            # Prédiction du jour suivant avec la matrice de Markov
            if day < len(val_dates) - 1:
                next_state = transition_matrix @ current_state
                current_state = np.maximum(next_state, 0)  # Éviter les valeurs négatives
        
        print(f"✅ {len(val_dates)} jours de prédictions générées")
        
        return val_dates, real_data, predictions
    
    def evaluate_predictions(self, real_data: np.ndarray, predictions: np.ndarray) -> Dict:
        """
        Évalue la qualité des prédictions
        
        Args:
            real_data: Données réelles [communes × jours]
            predictions: Prédictions [communes × jours]
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        print("📊 Évaluation des prédictions...")
        
        # Métriques globales
        mae_global = np.mean(np.abs(real_data - predictions))
        rmse_global = np.sqrt(np.mean((real_data - predictions) ** 2))
        
        # Métriques par commune
        mae_by_commune = {}
        correlation_by_commune = {}
        
        for i, commune in enumerate(self.communes):
            real_series = real_data[i, :]
            pred_series = predictions[i, :]
            
            mae_commune = np.mean(np.abs(real_series - pred_series))
            
            # Corrélation (si variance non nulle)
            if np.var(real_series) > 1e-6 and np.var(pred_series) > 1e-6:
                correlation = np.corrcoef(real_series, pred_series)[0, 1]
            else:
                correlation = 0.0
            
            mae_by_commune[commune] = mae_commune
            correlation_by_commune[commune] = correlation
        
        # Métriques temporelles (évolution jour par jour)
        daily_errors = np.mean(np.abs(real_data - predictions), axis=0)
        
        results = {
            "mae_global": mae_global,
            "rmse_global": rmse_global,
            "mae_by_commune": mae_by_commune,
            "correlation_by_commune": correlation_by_commune,
            "daily_errors": daily_errors.tolist(),
            "best_commune": min(mae_by_commune.items(), key=lambda x: x[1]),
            "worst_commune": max(mae_by_commune.items(), key=lambda x: x[1])
        }
        
        print(f"📈 MAE globale : {mae_global:.2f}")
        print(f"📈 RMSE globale : {rmse_global:.2f}")
        print(f"🏆 Meilleure commune : {results['best_commune'][0]} (MAE = {results['best_commune'][1]:.2f})")
        print(f"🔻 Moins bonne commune : {results['worst_commune'][0]} (MAE = {results['worst_commune'][1]:.2f})")
        
        return results
    
    def optimize_alpha_on_validation(self, alpha_values: List[float] = None) -> Tuple[float, Dict]:
        """
        Optimise le paramètre alpha sur les données de validation
        
        Args:
            alpha_values: Liste des valeurs d'alpha à tester
            
        Returns:
            Meilleur alpha et résultats détaillés
        """
        if alpha_values is None:
            alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        
        print(f"🎯 Optimisation d'alpha sur {len(alpha_values)} valeurs...")
        
        results_by_alpha = {}
        best_alpha = None
        best_mae = float('inf')
        
        for alpha in alpha_values:
            print(f"\n🔧 Test α = {alpha}...")
            
            # Entraînement avec cet alpha
            transition_matrix = self.train_model_on_period(alpha_geo=alpha)
            
            # Prédictions
            val_dates, real_data, predictions = self.predict_validation_period(transition_matrix)
            
            # Évaluation
            evaluation = self.evaluate_predictions(real_data, predictions)
            
            results_by_alpha[alpha] = {
                "transition_matrix": transition_matrix.tolist(),
                "evaluation": evaluation,
                "val_dates": val_dates,
                "real_data": real_data.tolist(),
                "predictions": predictions.tolist()
            }
            
            # Mise à jour du meilleur modèle
            if evaluation["mae_global"] < best_mae:
                best_mae = evaluation["mae_global"]
                best_alpha = alpha
        
        print(f"\n🏆 Meilleur modèle : α = {best_alpha} (MAE = {best_mae:.2f})")
        
        return best_alpha, results_by_alpha
    
    def save_historical_validation_results(self, best_alpha: float, results_by_alpha: Dict, 
                                         output_file: str = "data/historical_validation_results.json"):
        """
        Sauvegarde les résultats de validation historique
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        output_data = {
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "model_type": "historical_validation_markov",
                "training_period": f"{self.training_start} to {self.training_end}",
                "validation_period": f"{self.validation_start} to {self.validation_end}",
                "best_alpha": best_alpha,
                "communes": self.communes
            },
            "results": results_by_alpha
        }
        
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Résultats sauvegardés : {output_file}")
        
        return output_file
    
    def generate_comparison_data_for_visualization(self, best_alpha: float, results_by_alpha: Dict) -> Dict:
        """
        Génère les données formatées pour la visualisation dans main.py
        """
        print("📊 Génération des données pour visualisation...")
        
        best_results = results_by_alpha[best_alpha]
        
        # Conversion des données pour compatibilité avec main.py
        val_dates = best_results["val_dates"]
        real_data = np.array(best_results["real_data"])
        predictions = np.array(best_results["predictions"])
        
        # Format compatible avec main.py
        comparison_data = {
            "metadata": {
                "validation_period": f"{self.validation_start} to {self.validation_end}",
                "best_alpha": best_alpha,
                "mae_global": best_results["evaluation"]["mae_global"],
                "communes": self.communes
            },
            "dates": val_dates,
            "real_data": {},
            "predictions": {},
            "metrics": best_results["evaluation"]
        }
        
        # Organisation par commune
        for i, commune in enumerate(self.communes):
            comparison_data["real_data"][commune] = real_data[i, :].tolist()
            comparison_data["predictions"][commune] = predictions[i, :].tolist()
        
        # Sauvegarde pour main.py
        with open("data/historical_comparison_data.json", 'w', encoding='utf8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        print("💾 Données de comparaison sauvegardées : data/historical_comparison_data.json")
        
        return comparison_data


def run_historical_validation():
    """
    Fonction principale pour exécuter la validation historique
    """
    print("🚀 Démarrage de la validation historique Mars-Juin 2021")
    print("="*60)
    
    try:
        # Initialisation du modèle
        model = HistoricalValidationModel()
        
        # Optimisation d'alpha
        best_alpha, results_by_alpha = model.optimize_alpha_on_validation()
        
        # Sauvegarde des résultats
        model.save_historical_validation_results(best_alpha, results_by_alpha)
        
        # Génération des données pour visualisation
        comparison_data = model.generate_comparison_data_for_visualization(best_alpha, results_by_alpha)
        
        print("\n" + "="*60)
        print("✅ Validation historique terminée avec succès !")
        print(f"🏆 Meilleur modèle : α = {best_alpha}")
        print(f"📊 MAE globale : {comparison_data['metadata']['mae_global']:.2f}")
        print("📁 Résultats disponibles dans 'data/'")
        
        return comparison_data
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur lors de la validation : {e}")
        raise


def test_historical_validation():
    """Test de la validation historique"""
    print("🧪 Test de la validation historique...")
    
    try:
        comparison_data = run_historical_validation()
        
        # Affichage d'exemples de résultats
        print(f"\n📊 Exemple de résultats pour Bruxelles :")
        if "Bruxelles" in comparison_data["real_data"]:
            real_brussels = comparison_data["real_data"]["Bruxelles"][:5]
            pred_brussels = comparison_data["predictions"]["Bruxelles"][:5]
            dates_sample = comparison_data["dates"][:5]
            
            for i, date in enumerate(dates_sample):
                print(f"   {date}: Réel = {real_brussels[i]:.1f}, Prédit = {pred_brussels[i]:.1f}")
        
        print("✅ Test terminé avec succès !")
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur dans le test : {e}")
        raise


if __name__ == "__main__":
    test_historical_validation()