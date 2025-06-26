"""
Modèle de Markov matriciel pour prédiction COVID-19 Bruxelles
Prédiction(t+1) = MatriceTransition × Données(t)
Avec pondération géographique via poids Dijkstra
"""

import json
import os
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional
from colorama import init, Fore

# Initialisation de colorama
init(autoreset=True)


class MatrixMarkovModel:
    """
    Modèle de Markov matriciel avec pondération géographique
    X(t+1) = A × X(t) où A est la matrice de transition [19×19]
    """
    
    def __init__(self, smoothed_data_file: str = "data/smoothed_data.json", 
                 geographic_weights_file: str = "data/geographic_weights.json"):
        """
        Initialise le modèle matriciel
        """
        print("🔢 Initialisation du modèle de Markov matriciel...")
        
        # Chargement des données
        self.smoothed_data = self._load_smoothed_data(smoothed_data_file)
        self.geographic_weights = self._load_geographic_weights(geographic_weights_file)
        
        # Communes (ordre fixe pour la matrice)
        self.communes = sorted(list(self.geographic_weights.keys()))
        self.n_communes = len(self.communes)
        self.commune_to_index = {commune: i for i, commune in enumerate(self.communes)}
        
        # Matrice de transition
        self.transition_matrix = None
        self.geographic_weight_matrix = self._build_geographic_matrix()
        
        print(f"✅ Modèle initialisé : {self.n_communes} communes")
        print(f"📍 Ordre des communes : {self.communes[:3]}...{self.communes[-1]}")
    
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
    
    def _build_geographic_matrix(self) -> np.ndarray:
        """
        Construit la matrice de poids géographiques [19×19]
        """
        print("🗺️ Construction de la matrice de poids géographiques...")
        
        geo_matrix = np.zeros((self.n_communes, self.n_communes))
        
        for i, commune_i in enumerate(self.communes):
            for j, commune_j in enumerate(self.communes):
                if commune_i in self.geographic_weights:
                    weight = self.geographic_weights[commune_i].get(commune_j, 0.0)
                    geo_matrix[i, j] = weight
        
        print(f"✅ Matrice géographique {geo_matrix.shape} construite")
        print(f"📊 Poids moyens : diagonale={np.mean(np.diag(geo_matrix)):.3f}, "
              f"hors-diagonale={np.mean(geo_matrix[geo_matrix != np.diag(geo_matrix)]):.3f}")
        
        return geo_matrix
    
    def prepare_data_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les matrices X(t) et X(t+1) pour l'estimation
        
        Returns:
            X_t: matrice [19×T] des observations au temps t
            X_t1: matrice [19×T] des observations au temps t+1
        """
        print("📊 Préparation des matrices de données...")
        
        # Tri des dates
        sorted_dates = sorted(self.smoothed_data.keys())
        n_dates = len(sorted_dates)
        
        # Matrice des données [communes × dates]
        data_matrix = np.zeros((self.n_communes, n_dates))
        
        for t, date in enumerate(sorted_dates):
            for i, commune in enumerate(self.communes):
                if commune in self.smoothed_data[date]:
                    data_matrix[i, t] = float(self.smoothed_data[date][commune])
                else:
                    data_matrix[i, t] = 0.0
        
        # X(t) et X(t+1) pour l'estimation
        X_t = data_matrix[:, :-1]   # [19 × (T-1)]
        X_t1 = data_matrix[:, 1:]   # [19 × (T-1)]
        
        print(f"✅ Matrices préparées : X(t) et X(t+1) de forme {X_t.shape}")
        print(f"📅 Période : {sorted_dates[0]} → {sorted_dates[-1]}")
        
        return X_t, X_t1
    
    def estimate_base_transition_matrix(self, X_t: np.ndarray, X_t1: np.ndarray) -> np.ndarray:
        """
        Estime la matrice de transition de base par moindres carrés
        X(t+1) = A × X(t) → A = X(t+1) × X(t)^T × (X(t) × X(t)^T)^(-1)
        """
        print("🔢 Estimation de la matrice de transition de base...")
        
        try:
            # Méthode des moindres carrés : A = X(t+1) × X(t)^T × (X(t) × X(t)^T)^(-1)
            XtXt_T = X_t @ X_t.T  # [19×19]
            
            # Régularisation pour éviter la singularité
            regularization = 1e-6 * np.eye(self.n_communes)
            XtXt_T_reg = XtXt_T + regularization
            
            # Inversion
            XtXt_T_inv = np.linalg.inv(XtXt_T_reg)
            
            # Estimation finale
            A_base = X_t1 @ X_t.T @ XtXt_T_inv
            
            print(f"✅ Matrice de base estimée : {A_base.shape}")
            print(f"📊 Valeurs : min={np.min(A_base):.4f}, max={np.max(A_base):.4f}")
            
            return A_base
            
        except np.linalg.LinAlgError:
            print(Fore.YELLOW + "⚠️ Problème d'inversion, utilisation de la pseudo-inverse")
            A_base = X_t1 @ np.linalg.pinv(X_t)
            return A_base
    
    def apply_geographic_constraints(self, A_base: np.ndarray, alpha_geo: float = 0.5) -> np.ndarray:
        """
        Applique les contraintes géographiques à la matrice de transition
        
        """
        print(f"🗺️ Application des contraintes géographiques (α={alpha_geo})...")
        
        # Pondération géographique
        A_geographic = A_base * self.geographic_weight_matrix
        
        # Combinaison linéaire
        A_final = (1 - alpha_geo) * A_base + alpha_geo * A_geographic
        
        # Normalisation pour stabilité (optionnel)
        # Chaque ligne représente comment une commune influence les autres
        for i in range(self.n_communes):
            row_sum = np.sum(np.abs(A_final[i, :]))
            if row_sum > 2.0:  # Éviter une divergence
                A_final[i, :] = A_final[i, :] / row_sum * 1.5
        
        print(f"✅ Contraintes appliquées")
        print(f"📊 Impact géographique moyen : {np.mean(A_final * self.geographic_weight_matrix):.4f}")
        
        return A_final
    
    def evaluate_model(self, A: np.ndarray, X_t: np.ndarray, X_t1: np.ndarray, 
                      validation_split: float = 0.8) -> float:
        """
        Évalue la performance du modèle sur données de validation
        
        Args:
            A: matrice de transition [19×19]
            X_t, X_t1: données d'entraînement
            validation_split: proportion d'entraînement
            
        Returns:
            Erreur moyenne absolue sur validation
        """
        n_train = int(X_t.shape[1] * validation_split)
        
        # Validation sur la fin
        X_val_t = X_t[:, n_train:]
        X_val_t1 = X_t1[:, n_train:]
        
        if X_val_t.shape[1] == 0:
            return float('inf')
        
        # Prédictions
        X_pred = A @ X_val_t
        
        # Erreur moyenne absolue
        mae = np.mean(np.abs(X_pred - X_val_t1))
        
        return mae
    
    def train_model(self, alpha_geo_values: List[float] = None) -> Dict:
        """
        Entraîne le modèle avec différentes valeurs d'alpha_geo
        """
        if alpha_geo_values is None:
            alpha_geo_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        
        print("🎯 Entraînement du modèle de Markov matriciel...")
        
        # Préparation des données
        X_t, X_t1 = self.prepare_data_matrices()
        
        # Estimation de la matrice de base
        A_base = self.estimate_base_transition_matrix(X_t, X_t1)
        
        # Test de différents alpha_geo
        models = {}
        best_alpha = None
        best_error = float('inf')
        
        for alpha_geo in alpha_geo_values:
            print(f"\n🔧 Test alpha_geo = {alpha_geo}...")
            
            # Application des contraintes géographiques
            A_constrained = self.apply_geographic_constraints(A_base, alpha_geo)
            
            # Évaluation
            mae = self.evaluate_model(A_constrained, X_t, X_t1)
            
            models[f"alpha_geo_{alpha_geo}"] = {
                "alpha_geo": alpha_geo,
                "transition_matrix": A_constrained.tolist(),
                "mae_validation": mae
            }
            
            print(f"📊 MAE validation : {mae:.2f}")
            
            if mae < best_error:
                best_error = mae
                best_alpha = alpha_geo
                self.transition_matrix = A_constrained
        
        print(f"\n🏆 Meilleur modèle : alpha_geo = {best_alpha} (MAE = {best_error:.2f})")
        
        # Métadonnées
        models["metadata"] = {
            "best_alpha_geo": best_alpha,
            "best_mae": best_error,
            "communes": self.communes,
            "n_observations": X_t.shape[1]
        }
        
        return models
    
    def predict(self, initial_state: np.ndarray, horizon_days: int = 14) -> np.ndarray:
        """
        Effectue des prédictions sur plusieurs jours
        
        Args:
            initial_state: état initial [19×1]
            horizon_days: nombre de jours à prédire
            
        Returns:
            Prédictions [19×(horizon_days+1)] incluant l'état initial
        """
        if self.transition_matrix is None:
            raise ValueError("Modèle non entraîné. Appelez train_model() d'abord.")
        
        predictions = np.zeros((self.n_communes, horizon_days + 1))
        predictions[:, 0] = initial_state.flatten()
        
        current_state = initial_state.copy()
        
        for day in range(1, horizon_days + 1):
            next_state = self.transition_matrix @ current_state
            # Éviter les valeurs négatives
            next_state = np.maximum(next_state, 0)
            
            predictions[:, day] = next_state.flatten()
            current_state = next_state
        
        return predictions
    
    def predict_by_commune(self, initial_cases: Dict[str, float], 
                          horizon_days: int = 14) -> Dict[str, List[float]]:
        """
        Interface conviviale pour prédictions par commune
        
        Args:
            initial_cases: {commune: cas_actuels}
            horizon_days: horizon de prédiction
            
        Returns:
            {commune: [cas_jour0, cas_jour1, ..., cas_jourN]}
        """
        # Conversion en vecteur
        initial_vector = np.zeros((self.n_communes, 1))
        for commune, cases in initial_cases.items():
            if commune in self.commune_to_index:
                idx = self.commune_to_index[commune]
                initial_vector[idx, 0] = cases
        
        # Prédictions matricielles
        predictions_matrix = self.predict(initial_vector, horizon_days)
        
        # Conversion en dictionnaire
        predictions_dict = {}
        for i, commune in enumerate(self.communes):
            predictions_dict[commune] = predictions_matrix[i, :].tolist()
        
        return predictions_dict
    
    def save_model(self, models: Dict, output_file: str = "data/matrix_markov_models.json"):
        """Sauvegarde les modèles"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        output_data = {
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "model_type": "matrix_markov_geographic",
                "communes": self.communes,
                "matrix_shape": [self.n_communes, self.n_communes]
            },
            "models": models
        }
        
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Modèles sauvegardés : {output_file}")
    
    def generate_final_predictions(self, horizon_days: int = 14) -> Dict:
        """
        Génère les prédictions finales avec le meilleur modèle
        
        Args:
            horizon_days: horizon de prédiction
            
        Returns:
            Prédictions formatées pour sauvegarde
        """
        if self.transition_matrix is None:
            raise ValueError("Modèle non entraîné")
        
        print(f"🔮 Génération des prédictions finales ({horizon_days} jours)...")
        
        # État initial = dernières données disponibles
        last_date = max(self.smoothed_data.keys())
        initial_cases = {}
        
        for commune in self.communes:
            if commune in self.smoothed_data[last_date]:
                initial_cases[commune] = self.smoothed_data[last_date][commune]
            else:
                initial_cases[commune] = 0.0
        
        # Prédictions
        predictions = self.predict_by_commune(initial_cases, horizon_days)
        
        # Formatage avec dates
        from datetime import datetime, timedelta
        last_date_obj = datetime.strptime(last_date, "%Y-%m-%d")
        
        predictions_formatted = {}
        for day in range(horizon_days + 1):
            prediction_date = (last_date_obj + timedelta(days=day)).strftime("%Y-%m-%d")
            predictions_formatted[prediction_date] = {}
            
            for commune in self.communes:
                predictions_formatted[prediction_date][commune] = predictions[commune][day]
        
        print("✅ Prédictions générées pour toutes les communes")
        
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "base_date": last_date,
                "horizon_days": horizon_days,
                "communes": self.communes
            },
            "predictions": predictions_formatted
        }


def test_matrix_markov():
    """Test du modèle de Markov matriciel"""
    print("🧪 Test du modèle de Markov matriciel...")
    
    try:
        # Initialisation
        model = MatrixMarkovModel()
        
        # Entraînement
        models = model.train_model()
        
        # Sauvegarde
        model.save_model(models)
        
        # Prédictions finales
        predictions = model.generate_final_predictions(horizon_days=14)
        
        with open("data/matrix_predictions.json", 'w', encoding='utf8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print("💾 Prédictions sauvegardées : data/matrix_predictions.json")
        
        # Affichage d'exemple
        print(f"\n📊 Exemple de prédictions pour Bruxelles :")
        if "Bruxelles" in predictions["predictions"][list(predictions["predictions"].keys())[0]]:
            for i, (date, data) in enumerate(list(predictions["predictions"].items())[:5]):
                cases = data["Bruxelles"]
                print(f"   {date}: {cases:.1f} cas")
        
        print("✅ Test terminé avec succès !")
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur dans le test : {e}")
        raise


if __name__ == "__main__":
    test_matrix_markov()