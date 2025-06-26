"""
IMPLÉMENTATION CORRECTE SELON VOTRE RAISONNEMENT MATHÉMATIQUE
Suit exactement : X⃗(t+1) = A · X⃗(t) + ε⃗(t)
Avec A_finale = (1 - α) · A_brute + α · (A_brute ⊙ G)
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from colorama import init, Fore

init(autoreset=True)


class MatrixMarkovModel:
    """
    Implémentation exacte de votre modèle mathématique :
    X⃗(t+1) = A · X⃗(t) avec contraintes géographiques
    """
    
    def __init__(self, smoothed_data_file: str = "data/smoothed_data.json", 
                 geographic_weights_file: str = "data/geographic_weights.json"):
        """
        Initialise selon votre modèle mathématique
        """
        print("🔬 Initialisation du modèle de Markov selon le raisonnement mathématique...")
        
        # Chargement des données
        self.smoothed_data = self._load_smoothed_data(smoothed_data_file)
        self.geographic_weights = self._load_geographic_weights(geographic_weights_file)
        
        # Communes (19 communes de Bruxelles)
        self.communes = sorted(list(self.geographic_weights.keys()))
        self.n_communes = len(self.communes)
        self.commune_to_index = {commune: i for i, commune in enumerate(self.communes)}
        
        # Modèle selon votre approche
        self.transition_matrix = None  # A_finale
        self.geographic_matrix = self._build_geographic_matrix_G()  # Matrice G
        
        print(f"✅ Modèle mathématique initialisé : {self.n_communes} communes")
        print("📋 Suivant : X⃗(t+1) = A · X⃗(t) + ε⃗(t)")
    
    def _load_smoothed_data(self, file_path: str) -> Dict:
        """Charge les données lissées Savitzky-Golay"""
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                data = json.load(f)
            return data["data"] if "data" in data else data
        except FileNotFoundError:
            print(Fore.RED + f"❌ Fichier non trouvé : {file_path}")
            raise
    
    def _load_geographic_weights(self, file_path: str) -> Dict:
        """Charge les poids géographiques (résultats Dijkstra)"""
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                data = json.load(f)
            return data["weights"] if "weights" in data else data
        except FileNotFoundError:
            print(Fore.RED + f"❌ Fichier non trouvé : {file_path}")
            raise
    
    def _build_geographic_matrix_G(self) -> np.ndarray:
        """
        Construit la matrice G selon votre spécification :
        G[i,j] = influence géographique de commune i sur commune j
        Normalisation : Σⱼ G[i,j] = 1 (stochastique)
        """
        print("🗺️ Construction de la matrice géographique G...")
        
        G = np.zeros((self.n_communes, self.n_communes))
        
        # Remplissage selon les poids Dijkstra
        for i, commune_i in enumerate(self.communes):
            for j, commune_j in enumerate(self.communes):
                if commune_i in self.geographic_weights:
                    weight = self.geographic_weights[commune_i].get(commune_j, 0.0)
                    G[i, j] = weight
        
        # Normalisation stochastique (selon votre spécification)
        for i in range(self.n_communes):
            row_sum = np.sum(G[i, :])
            if row_sum > 0:
                G[i, :] = G[i, :] / row_sum  # Σⱼ G[i,j] = 1
            else:
                G[i, i] = 1.0  # Auto-boucle si isolée
        
        print(f"✅ Matrice G construite : stochastique avec Σⱼ G[i,j] = 1")
        return G
    
    def prepare_data_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare X(t) et X(t+1) selon votre formulation :
        X⃗(t) ∈ ℝ¹⁹ : vecteur d'état (cas par commune au jour t)
        """
        print("📊 Préparation des matrices X(t) et X(t+1)...")
        
        sorted_dates = sorted(self.smoothed_data.keys())
        n_dates = len(sorted_dates)
        
        # Matrice des données [communes × temps]
        data_matrix = np.zeros((self.n_communes, n_dates))
        
        for t, date in enumerate(sorted_dates):
            for i, commune in enumerate(self.communes):
                if commune in self.smoothed_data[date]:
                    value = float(self.smoothed_data[date][commune])
                    data_matrix[i, t] = max(0, value)  # ỹᵢ(t) ∈ ℝ⁺
                else:
                    data_matrix[i, t] = 0.0
        
        # X(t) et X(t+1) selon votre formulation
        X_t = data_matrix[:, :-1]   # [19 × (T-1)]
        X_t1 = data_matrix[:, 1:]   # [19 × (T-1)]
        
        print(f"✅ Matrices préparées : X(t) et X(t+1) forme {X_t.shape}")
        print(f"📅 Période d'entraînement : {sorted_dates[0]} → {sorted_dates[-1]}")
        
        return X_t, X_t1
    
    def estimate_A_brute_by_least_squares(self, X_t: np.ndarray, X_t1: np.ndarray) -> np.ndarray:
        """
        Estimation de A_brute par moindres carrés selon votre formulation :
        
        min ||X⃗(t+1) - A · X⃗(t)||²
        
        Solution analytique :
        A = X(t+1) · X(t)ᵀ · (X(t) · X(t)ᵀ)⁻¹
        """
        print("🔢 Estimation de A_brute par moindres carrés...")
        print("📐 Formule : A = X(t+1) · X(t)ᵀ · (X(t) · X(t)ᵀ)⁻¹")
        
        try:
            # Calcul selon votre formule exacte
            X_t_transpose = X_t.T
            XtXt_T = X_t @ X_t_transpose  # X(t) · X(t)ᵀ
            
            # Régularisation pour stabilité numérique
            epsilon = 1e-6
            XtXt_T_reg = XtXt_T + epsilon * np.eye(self.n_communes)
            
            # Inversion
            XtXt_T_inv = np.linalg.inv(XtXt_T_reg)
            
            # Solution analytique
            A_brute = X_t1 @ X_t_transpose @ XtXt_T_inv
            
            print(f"✅ A_brute estimée : forme {A_brute.shape}")
            print(f"📊 Valeurs : min={np.min(A_brute):.4f}, max={np.max(A_brute):.4f}")
            
            return A_brute
            
        except np.linalg.LinAlgError:
            print(Fore.YELLOW + "⚠️ Problème d'inversion, utilisation pseudo-inverse")
            X_t_pinv = np.linalg.pinv(X_t)
            A_brute = X_t1 @ X_t_pinv
            return A_brute
    
    def apply_geographic_constraints(self, A_brute: np.ndarray, alpha: float) -> np.ndarray:
        """
        Application des contraintes géographiques selon votre formulation EXACTE :
        
        A_finale = (1 - α) · A_brute + α · (A_brute ⊙ G)
        
        où ⊙ est le produit de Hadamard (élément par élément)
        """
        print(f"🗺️ Application des contraintes géographiques (α={alpha})...")
        print("📐 Formule : A_finale = (1 - α) · A_brute + α · (A_brute ⊙ G)")
        
        # Produit de Hadamard : A_brute ⊙ G
        hadamard_product = A_brute * self.geographic_matrix  # ⊙
        
        # Combinaison linéaire selon votre formule
        A_finale = (1 - alpha) * A_brute + alpha * hadamard_product
        
        print(f"✅ A_finale calculée avec α={alpha}")
        
        # Vérification des propriétés
        eigenvalues = np.linalg.eigvals(A_finale)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        
        print(f"📊 Rayon spectral : {max_eigenvalue:.4f}")
        
        if max_eigenvalue > 1.1:
            print(Fore.YELLOW + f"⚠️ Système potentiellement instable (λ_max > 1)")
        
        return A_finale
    
    def evaluate_model_performance(self, A: np.ndarray, X_t: np.ndarray, X_t1: np.ndarray, 
                                 validation_split: float = 0.8) -> float:
        """
        Évaluation selon votre critère : MAE_validation(α)
        """
        n_train = int(X_t.shape[1] * validation_split)
        
        if n_train < 10:  # Besoin d'assez de données
            return float('inf')
        
        # Division entraînement/validation
        X_val_t = X_t[:, n_train:]
        X_val_t1 = X_t1[:, n_train:]
        
        if X_val_t.shape[1] == 0:
            return float('inf')
        
        # Prédictions : X⃗(t+1) = A · X⃗(t)
        X_pred = A @ X_val_t
        
        # Mean Absolute Error
        mae = np.mean(np.abs(X_pred - X_val_t1))
        
        return mae
    
    def train_model(self, alpha_geo_values: List[float] = None) -> Dict:
        """
        Entraînement selon votre optimisation :
        α* = argmin(α) MAE_validation(α)
        """
        if alpha_geo_values is None:
            alpha_geo_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        
        print("🎯 Entraînement selon le modèle mathématique...")
        print("🔍 Optimisation : α* = argmin(α) MAE_validation(α)")
        
        # Préparation des données
        X_t, X_t1 = self.prepare_data_matrices()
        
        # Estimation de A_brute (étape 1)
        A_brute = self.estimate_A_brute_by_least_squares(X_t, X_t1)
        
        # Test de différents α (étape 2)
        models = {}
        best_alpha = None
        best_mae = float('inf')
        
        for alpha in alpha_geo_values:
            print(f"\n🔧 Test α = {alpha}...")
            
            # Application des contraintes géographiques
            A_finale = self.apply_geographic_constraints(A_brute, alpha)
            
            # Évaluation
            mae = self.evaluate_model_performance(A_finale, X_t, X_t1)
            
            models[f"alpha_geo_{alpha}"] = {
                "alpha_geo": alpha,
                "transition_matrix": A_finale.tolist(),
                "mae_validation": mae
            }
            
            print(f"📊 MAE_validation({alpha}) = {mae:.3f}")
            
            # Optimisation
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
                self.transition_matrix = A_finale
        
        if best_alpha is None:
            print(Fore.RED + "❌ Aucun α valide trouvé")
            return {}
        
        print(f"\n🏆 α* optimal = {best_alpha} (MAE = {best_mae:.3f})")
        
        # Métadonnées
        models["metadata"] = {
            "best_alpha_geo": best_alpha,
            "best_mae": best_mae,
            "communes": self.communes,
            "n_observations": X_t.shape[1],
            "model_formula": "X(t+1) = A * X(t)",
            "constraint_formula": "A = (1-α)*A_brute + α*(A_brute⊙G)"
        }
        
        return models
    
    def predict(self, initial_state: np.ndarray, horizon_days: int = 14) -> np.ndarray:
        """
        Prédictions selon votre modèle : X⃗(t+1) = A · X⃗(t)
        PAS de zone d'incertitude, PAS de valeurs fixes, courbe qui monte/descend
        """
        if self.transition_matrix is None:
            raise ValueError("Modèle non entraîné")
        
        print(f"🔮 Prédictions selon X⃗(t+1) = A · X⃗(t) ({horizon_days} jours)...")
        
        predictions = np.zeros((self.n_communes, horizon_days + 1))
        predictions[:, 0] = initial_state.flatten()
        
        current_state = initial_state.copy()
        
        # Application itérative : X⃗(t+1) = A · X⃗(t)
        for day in range(1, horizon_days + 1):
            next_state = self.transition_matrix @ current_state
            
            # Pas de contraintes artificielles - juste le modèle mathématique
            predictions[:, day] = next_state.flatten()
            current_state = next_state
        
        print("✅ Prédictions générées selon le modèle de Markov")
        
        return predictions
    
    def predict_by_commune(self, initial_cases: Dict[str, float], 
                          horizon_days: int = 14) -> Dict[str, List[float]]:
        """Interface pour prédictions par commune"""
        initial_vector = np.zeros((self.n_communes, 1))
        for commune, cases in initial_cases.items():
            if commune in self.commune_to_index:
                idx = self.commune_to_index[commune]
                initial_vector[idx, 0] = cases
        
        predictions_matrix = self.predict(initial_vector, horizon_days)
        
        predictions_dict = {}
        for i, commune in enumerate(self.communes):
            predictions_dict[commune] = predictions_matrix[i, :].tolist()
        
        return predictions_dict
    
    def save_model(self, models: Dict, output_file: str = "data/matrix_markov_models.json"):
        """Sauvegarde selon votre modèle mathématique"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        output_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "model_type": "markov_mathematical_formulation",
                "communes": self.communes,
                "mathematical_foundation": {
                    "state_equation": "X(t+1) = A * X(t) + ε(t)",
                    "estimation": "A = X(t+1) * X(t)^T * (X(t) * X(t)^T)^(-1)",
                    "constraints": "A_finale = (1-α)*A_brute + α*(A_brute⊙G)",
                    "optimization": "α* = argmin(α) MAE_validation(α)"
                }
            },
            "models": models
        }
        
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Modèle mathématique sauvegardé : {output_file}")
    
    def generate_final_predictions(self, horizon_days: int = 14) -> Dict:
        """Génère les prédictions finales selon votre modèle"""
        if self.transition_matrix is None:
            raise ValueError("Modèle non entraîné")
        
        print(f"🔮 Prédictions finales selon le modèle de Markov ({horizon_days} jours)...")
        
        # État initial = dernières données lissées
        last_date = max(self.smoothed_data.keys())
        initial_cases = {}
        
        for commune in self.communes:
            if commune in self.smoothed_data[last_date]:
                initial_cases[commune] = self.smoothed_data[last_date][commune]
            else:
                initial_cases[commune] = 0.0
        
        # Prédictions selon X⃗(t+1) = A · X⃗(t)
        predictions = self.predict_by_commune(initial_cases, horizon_days)
        
        # Formatage avec dates
        last_date_obj = datetime.strptime(last_date, "%Y-%m-%d")
        
        predictions_formatted = {}
        for day in range(horizon_days + 1):
            prediction_date = (last_date_obj + timedelta(days=day)).strftime("%Y-%m-%d")
            predictions_formatted[prediction_date] = {}
            
            for commune in self.communes:
                predictions_formatted[prediction_date][commune] = predictions[commune][day]
        
        print("✅ Prédictions finales générées selon le modèle mathématique")
        
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "base_date": last_date,
                "horizon_days": horizon_days,
                "communes": self.communes,
                "mathematical_model": "X(t+1) = A * X(t)"
            },
            "predictions": predictions_formatted
        }


def test_matrix_markov():
    """Test du modèle selon votre raisonnement mathématique"""
    print("🧪 Test du modèle selon le raisonnement mathématique...")
    print("📋 Implémentation : X⃗(t+1) = A · X⃗(t) + ε⃗(t)")
    
    try:
        # Initialisation
        model = MatrixMarkovModel()
        
        # Entraînement selon votre optimisation
        models = model.train_model()
        
        if not models:
            print(Fore.RED + "❌ Entraînement échoué")
            return
        
        # Sauvegarde
        model.save_model(models)
        
        # Prédictions finales
        predictions = model.generate_final_predictions(horizon_days=14)
        
        with open("data/matrix_predictions.json", 'w', encoding='utf8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print("💾 Prédictions sauvegardées : data/matrix_predictions.json")
        
        # Vérification du modèle
        best_alpha = models["metadata"]["best_alpha_geo"]
        best_model = models[f"alpha_geo_{best_alpha}"]
        
        A_finale = np.array(best_model["transition_matrix"])
        print(f"\n📊 Matrice A_finale :")
        print(f"   - Forme : {A_finale.shape}")
        print(f"   - Valeurs : [{np.min(A_finale):.3f}, {np.max(A_finale):.3f}]")
        
        # Test de prédiction
        print(f"\n📋 Exemple de prédictions pour Bruxelles :")
        if "Bruxelles" in predictions["predictions"][list(predictions["predictions"].keys())[0]]:
            for i, (date, data) in enumerate(list(predictions["predictions"].items())[:5]):
                cases = data["Bruxelles"]
                print(f"   {date}: {cases:.2f} cas")
        
        print("✅ Test selon le raisonnement mathématique terminé !")
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur : {e}")
        raise


if __name__ == "__main__":
    test_matrix_markov()