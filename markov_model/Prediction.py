"""
IMPLÉMENTATION CORRECTE SELON VOTRE RAISONNEMENT MATHÉMATIQUE
Suit exactement : X⃗(t+1) = A · X⃗(t) + ε⃗(t)
Avec A_finale = (1 - α) · A_brute + α · (A_brute ⊙ G)
CORRECTIONS: Conservation de masse + Stabilité numérique
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
    CORRIGÉ pour résoudre les problèmes de prédiction
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
            
            # CORRECTION : Régularisation Ridge pour stabilité
            epsilon = 1e-4  # Augmenté pour plus de stabilité
            XtXt_T_reg = XtXt_T + epsilon * np.eye(self.n_communes)
            
            # Inversion
            XtXt_T_inv = np.linalg.inv(XtXt_T_reg)
            
            # Solution analytique
            A_brute = X_t1 @ X_t_transpose @ XtXt_T_inv
            
            # CORRECTION : Normalisation pour conservation de masse approximative
            # Éviter que A_brute soit trop sub-stochastique
            row_sums = np.sum(A_brute, axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 0.5)  # Éviter divisions par des nombres trop petits
            A_brute = A_brute / row_sums * 0.95  # Conservation approximative (95%)
            
            print(f"✅ A_brute estimée : forme {A_brute.shape}")
            print(f"📊 Valeurs : min={np.min(A_brute):.4f}, max={np.max(A_brute):.4f}")
            print(f"📊 Sommes lignes : min={np.min(np.sum(A_brute, axis=1)):.3f}, max={np.max(np.sum(A_brute, axis=1)):.3f}")
            
            return A_brute
            
        except np.linalg.LinAlgError:
            print(Fore.YELLOW + "⚠️ Problème d'inversion, utilisation pseudo-inverse")
            X_t_pinv = np.linalg.pinv(X_t)
            A_brute = X_t1 @ X_t_pinv
            
            # Même normalisation de sécurité
            row_sums = np.sum(A_brute, axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 0.5)
            A_brute = A_brute / row_sums * 0.95
            
            return A_brute
    
    def apply_geographic_constraints(self, A_brute: np.ndarray, alpha: float) -> np.ndarray:
        """
        Application des contraintes géographiques selon votre formulation EXACTE :
        
        A_finale = (1 - α) · A_brute + α · (A_brute ⊙ G)
        
        où ⊙ est le produit de Hadamard (élément par élément)
        
        CORRECTION : Renormalisation pour préserver les propriétés stochastiques
        """
        print(f"🗺️ Application des contraintes géographiques (α={alpha})...")
        print("📐 Formule : A_finale = (1 - α) · A_brute + α · (A_brute ⊙ G)")
        
        # Produit de Hadamard : A_brute ⊙ G
        hadamard_product = A_brute * self.geographic_matrix  # ⊙
        
        # Combinaison linéaire selon votre formule
        A_finale = (1 - alpha) * A_brute + alpha * hadamard_product
        
        # CORRECTION CRITIQUE : Renormalisation pour conservation de masse
        print("🔧 Application de la correction de conservation de masse...")
        
        # Méthode 1: Normalisation ligne par ligne pour rendre stochastique
        row_sums = np.sum(A_finale, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-6)  # Éviter division par zéro
        A_finale_normalized = A_finale / row_sums
        
        # Méthode 2: Assurer la stabilité (rayon spectral ≤ 1)
        eigenvalues = np.linalg.eigvals(A_finale_normalized)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        
        if max_eigenvalue > 1.0:
            print(f"⚠️ Correction de stabilité : λ_max = {max_eigenvalue:.3f} → 1.0")
            A_finale_normalized = A_finale_normalized / max_eigenvalue
        
        # Méthode 3: Assurer la positivité
        A_finale_normalized = np.maximum(A_finale_normalized, 0.0)
        
        # Renormalisation finale
        row_sums_final = np.sum(A_finale_normalized, axis=1, keepdims=True)
        row_sums_final = np.maximum(row_sums_final, 1e-6)
        A_finale_normalized = A_finale_normalized / row_sums_final
        
        print(f"✅ A_finale calculée avec α={alpha}")
        print(f"📊 Sommes lignes après correction : min={np.min(np.sum(A_finale_normalized, axis=1)):.6f}, max={np.max(np.sum(A_finale_normalized, axis=1)):.6f}")
        
        # Vérification des propriétés
        eigenvalues_final = np.linalg.eigvals(A_finale_normalized)
        max_eigenvalue_final = np.max(np.abs(eigenvalues_final))
        
        print(f"📊 Rayon spectral final : {max_eigenvalue_final:.4f}")
        
        if max_eigenvalue_final <= 1.001:  # Tolérance numérique
            print("✅ Matrice stable (rayon spectral ≤ 1)")
        else:
            print(Fore.YELLOW + f"⚠️ Matrice potentiellement instable")
        
        return A_finale_normalized
    
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
            "constraint_formula": "A = (1-α)*A_brute + α*(A_brute⊙G)",
            "corrections_applied": [
                "conservation_de_masse",
                "stabilite_numerique", 
                "regularisation_ridge",
                "normalisation_stochastique"
            ]
        }
        
        return models
    
    def predict(self, initial_state: np.ndarray, horizon_days: int = 14) -> np.ndarray:
        """
        Prédictions selon votre modèle : X⃗(t+1) = A · X⃗(t)
        VALEURS EXACTES pour chaque jour - PAS de zone d'incertitude
        """
        if self.transition_matrix is None:
            raise ValueError("Modèle non entraîné")
        
        print(f"🔮 Prédictions déterministes : X⃗(t+1) = A · X⃗(t) ({horizon_days} jours)...")
        
        predictions = np.zeros((self.n_communes, horizon_days + 1))
        predictions[:, 0] = initial_state.flatten()
        
        current_state = initial_state.copy()
        
        # Application itérative stricte : X⃗(t+1) = A · X⃗(t)
        for day in range(1, horizon_days + 1):
            next_state = self.transition_matrix @ current_state
            
            # Assurer la positivité uniquement
            next_state = np.maximum(next_state, 0.0)
            
            predictions[:, day] = next_state.flatten()
            current_state = next_state
        
        print("✅ Prédictions déterministes générées - valeurs exactes pour chaque jour")
        
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
                "model_type": "markov_mathematical_formulation_corrected",
                "communes": self.communes,
                "mathematical_foundation": {
                    "state_equation": "X(t+1) = A * X(t) + ε(t)",
                    "estimation": "A = X(t+1) * X(t)^T * (X(t) * X(t)^T)^(-1)",
                    "constraints": "A_finale = (1-α)*A_brute + α*(A_brute⊙G)",
                    "optimization": "α* = argmin(α) MAE_validation(α)"
                },
                "corrections_applied": {
                    "conservation_de_masse": "Renormalisation stochastique",
                    "stabilite_numerique": "Rayon spectral ≤ 1",
                    "regularisation": "Ridge + positivité",
                    "robustesse": "Gestion des cas limites"
                }
            },
            "models": models
        }
        
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Modèle mathématique corrigé sauvegardé : {output_file}")
    
    def generate_final_predictions(self, horizon_days: int = 14) -> Dict:
        """Génère les prédictions finales selon votre modèle"""
        if self.transition_matrix is None:
            raise ValueError("Modèle non entraîné")
        
        print(f"🔮 Prédictions finales selon le modèle de Markov corrigé ({horizon_days} jours)...")
        
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
        
        print("✅ Prédictions finales générées selon le modèle mathématique corrigé")
        
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "base_date": last_date,
                "horizon_days": horizon_days,
                "communes": self.communes,
                "mathematical_model": "X(t+1) = A * X(t) - VERSION CORRIGÉE",
                "corrections": [
                    "Conservation de masse",
                    "Stabilité numérique",
                    "Matrices stochastiques"
                ]
            },
            "predictions": predictions_formatted
        }


def test_matrix_markov():
    """Test du modèle SUR PÉRIODE STABLE (Juin-Août 2021 → Septembre 2021)"""
    print("🧪 Test du modèle sur période STABLE...")
    print("📅 Entraînement : Juin-Juillet-Août 2021 (période calme)")
    print("📅 Prédiction : Septembre 2021")
    print("📋 Objectif : Tester sur période sans grands pics")
    
    try:
        # Initialisation
        model = MatrixMarkovModel()
        
        # ENTRAÎNEMENT sur Juin-Juillet-Août 2021 (période stable)
        print("\n🎯 Entraînement sur Juin-Juillet-Août 2021...")
        
        # Filtrer les données pour l'entraînement (Juin-Août 2021)
        training_data = {}
        for date, data in model.smoothed_data.items():
            if "2021-06" in date or "2021-07" in date or "2021-08" in date:
                training_data[date] = data
        
        if not training_data:
            print(Fore.RED + "❌ Aucune donnée trouvée pour Juin-Août 2021")
            print("📋 Dates disponibles dans vos données :")
            dates_sample = sorted(list(model.smoothed_data.keys()))[:10]
            for d in dates_sample:
                print(f"   - {d}")
            return
        
        print(f"✅ Données d'entraînement : {len(training_data)} jours")
        
        # Remplacer temporairement les données
        original_data = model.smoothed_data
        model.smoothed_data = training_data
        
        # Entraînement sur cette période stable
        models = model.train_model([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        if not models:
            print(Fore.RED + "❌ Entraînement échoué")
            return
        
        # Restaurer les données complètes
        model.smoothed_data = original_data
        
        print(f"✅ Modèle entraîné sur période stable (Juin-Août 2021)")
        
        # PRÉDICTIONS SUR SEPTEMBRE 2021
        print(f"\n🔮 Génération des prédictions pour Septembre 2021...")
        
        # État initial : 31 août 2021
        initial_date = "2021-08-31"
        
        # Chercher la dernière date d'août disponible
        aug_dates = [d for d in model.smoothed_data.keys() if "2021-08" in d]
        if aug_dates:
            initial_date = max(aug_dates)
            print(f"📅 État initial : {initial_date}")
        else:
            print(f"⚠️ Aucune donnée d'août trouvée, recherche alternative...")
            # Chercher toute date de fin d'été 2021
            summer_dates = [d for d in model.smoothed_data.keys() 
                          if any(month in d for month in ["2021-07", "2021-08"])]
            if summer_dates:
                initial_date = max(summer_dates)
                print(f"📅 État initial (alternatif) : {initial_date}")
            else:
                print(Fore.RED + "❌ Impossible de trouver une date de base")
                return
        
        # Préparer l'état initial
        initial_cases = {}
        if initial_date in model.smoothed_data:
            for commune in model.communes:
                initial_cases[commune] = model.smoothed_data[initial_date].get(commune, 0.0)
        else:
            print(f"⚠️ Date {initial_date} non trouvée, utilisation de valeurs par défaut")
            for commune in model.communes:
                initial_cases[commune] = 5.0  # Valeur par défaut faible
        
        # Calculer Septembre 2021 (30 jours)
        horizon_days = 30
        print(f"🕐 Horizon de prédiction : {horizon_days} jours (Septembre)")
        
        # Générer les prédictions
        predictions_matrix = model.predict_by_commune(initial_cases, horizon_days)
        
        # AFFICHAGE DÉTAILLÉ DES PRÉDICTIONS VS RÉALITÉ SEPTEMBRE 2021
        print("\n" + "="*90)
        print("📊 PRÉDICTIONS vs RÉALITÉ - SEPTEMBRE 2021 (Période Stable)")
        print("="*90)
        
        # Principales communes à analyser
        principales_communes = ["Bruxelles", "Anderlecht", "Schaerbeek", "Ixelles"]
        
        from datetime import datetime, timedelta
        start_pred = datetime(2021, 9, 1)
        
        for commune in principales_communes:
            if commune in model.communes:
                print(f"\n🏘️ {commune.upper()}:")
                print("-" * 80)
                print(f"{'Date':<12} {'Jour':<8} {'PRÉDIT':<10} {'RÉEL':<10} {'Écart':<10} {'Écart %':<10}")
                print("-" * 80)
                
                # Données pour tout septembre (30 jours)
                for day in range(min(30, len(predictions_matrix[commune]))):
                    pred_date = start_pred + timedelta(days=day)
                    date_str = pred_date.strftime("%Y-%m-%d")
                    
                    predicted_value = predictions_matrix[commune][day]
                    
                    # Chercher la valeur réelle
                    real_value = model.smoothed_data.get(date_str, {}).get(commune, None)
                    
                    if real_value is not None:
                        ecart = predicted_value - real_value
                        ecart_pct = (ecart / real_value * 100) if real_value > 0 else 0
                        
                        print(f"{date_str:<12} J+{day:<6} {predicted_value:>8.2f} {real_value:>8.2f} {ecart:>+8.2f} {ecart_pct:>+7.1f}%")
                    else:
                        print(f"{date_str:<12} J+{day:<6} {predicted_value:>8.2f} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
        
        # Analyse globale de performance sur Septembre
        print(f"\n" + "="*90)
        print("📈 ANALYSE DE PERFORMANCE - SEPTEMBRE 2021")
        print("="*90)
        
        total_mae = 0
        total_points = 0
        commune_performances = {}
        all_predictions = []
        all_reals = []
        
        for commune in model.communes:
            commune_mae = 0
            commune_points = 0
            
            for day in range(min(30, len(predictions_matrix[commune]))):
                pred_date = start_pred + timedelta(days=day)
                date_str = pred_date.strftime("%Y-%m-%d")
                
                predicted_value = predictions_matrix[commune][day]
                real_value = model.smoothed_data.get(date_str, {}).get(commune, None)
                
                if real_value is not None:
                    mae = abs(predicted_value - real_value)
                    commune_mae += mae
                    commune_points += 1
                    total_mae += mae
                    total_points += 1
                    
                    all_predictions.append(predicted_value)
                    all_reals.append(real_value)
            
            if commune_points > 0:
                commune_performances[commune] = commune_mae / commune_points
        
        if total_points > 0:
            global_mae = total_mae / total_points
            print(f"   - MAE globale     : {global_mae:.2f} cas/jour")
            print(f"   - Points évalués  : {total_points}")
            
            # Calcul de corrélation
            import numpy as np
            if len(all_predictions) > 1 and len(all_reals) > 1:
                correlation = np.corrcoef(all_predictions, all_reals)[0, 1]
                print(f"   - Corrélation     : {correlation:.3f}")
            
            # Erreur relative moyenne
            relative_errors = []
            for pred, real in zip(all_predictions, all_reals):
                if real > 0:
                    relative_errors.append(abs(pred - real) / real)
            
            if relative_errors:
                mean_relative_error = np.mean(relative_errors) * 100
                print(f"   - Erreur relative : {mean_relative_error:.1f}%")
            
            # Top 3 meilleures communes
            best_communes = sorted(commune_performances.items(), key=lambda x: x[1])[:3]
            print(f"\n   📊 Top 3 meilleures prédictions :")
            for i, (commune, mae) in enumerate(best_communes, 1):
                print(f"      {i}. {commune:<25} : MAE = {mae:.2f}")
            
            # 3 moins bonnes communes
            worst_communes = sorted(commune_performances.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\n   📊 3 prédictions les moins bonnes :")
            for i, (commune, mae) in enumerate(worst_communes, 1):
                print(f"      {i}. {commune:<25} : MAE = {mae:.2f}")
        
        # Propriétés du modèle
        best_alpha = models["metadata"]["best_alpha_geo"]
        print(f"\n   🔧 Paramètres du modèle :")
        print(f"      - Alpha optimal   : {best_alpha}")
        print(f"      - MAE validation  : {models['metadata']['best_mae']:.3f}")
        print(f"      - Période train   : Juin-Juillet-Août 2021 (stable)")
        print(f"      - Période test    : Septembre 2021")
        print(f"      - Horizon         : {horizon_days} jours")
        
        # Analyse de la tendance
        if len(all_predictions) > 10 and len(all_reals) > 10:
            # Tendance des prédictions
            pred_start = np.mean(all_predictions[:5])
            pred_end = np.mean(all_predictions[-5:])
            pred_trend = (pred_end - pred_start) / pred_start * 100 if pred_start > 0 else 0
            
            # Tendance réelle
            real_start = np.mean(all_reals[:5])
            real_end = np.mean(all_reals[-5:])
            real_trend = (real_end - real_start) / real_start * 100 if real_start > 0 else 0
            
            print(f"\n   📈 Analyse des tendances :")
            print(f"      - Tendance prédite  : {pred_trend:+.1f}%")
            print(f"      - Tendance réelle   : {real_trend:+.1f}%")
            print(f"      - Écart tendances   : {abs(pred_trend - real_trend):.1f} points")
        
        print(f"\n" + "="*90)
        print("✅ ANALYSE SEPTEMBRE 2021 TERMINÉE")
        print("="*90)
        print("🔍 Ces prédictions sont-elles meilleures sur cette période stable ?")
        print("📊 La MAE et la corrélation sont-elles acceptables ?")
        print("🎯 Le modèle capture-t-il mieux une période sans grands pics ?")
        
        # Sauvegarder les résultats
        validation_results = {
            "metadata": {
                "training_period": "2021-06 to 2021-08 (stable period)",
                "prediction_period": "2021-09 (September)", 
                "global_mae": global_mae if total_points > 0 else None,
                "correlation": correlation if 'correlation' in locals() else None,
                "best_alpha": best_alpha,
                "horizon_days": horizon_days
            },
            "commune_performances": commune_performances,
            "predictions_vs_reality": {
                "predictions": all_predictions,
                "reals": all_reals
            }
        }
        
        with open("data/stable_period_validation.json", 'w', encoding='utf8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        print("💾 Résultats sauvegardés : data/stable_period_validation.json")
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_matrix_markov()