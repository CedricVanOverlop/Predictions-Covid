"""
Script principal pour exécuter l'analyse complète COVID-19 Bruxelles
Inclut la validation historique Mars-Juin 2021 et les nouvelles visualisations
"""

import os
import sys
from colorama import init, Fore
import subprocess

# Initialisation
init(autoreset=True)

def run_module(module_name, description):
    """Exécute un module et gère les erreurs"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        # Import dynamique du module
        if module_name == "data_downloader":
            from Data_Processing.data_downloader import test_export_download
            test_export_download()
            
        elif module_name == "data_loader":
            from Data_Processing.data_loader import load_covid_data_pipeline
            load_covid_data_pipeline()
            
        elif module_name == "savitzky_golay":
            from Data_Processing.savitzky_golay import smooth_covid_data
            smooth_covid_data()
            
        elif module_name == "dijkstra":
            from geography.dijkstra import test_dijkstra
            test_dijkstra()
            
        elif module_name == "markov_prediction":
            from markov_model.Prediction import test_matrix_markov
            test_matrix_markov()
            
        elif module_name == "historical_validation":
            # Import du nouveau module de validation historique
            sys.path.append('.')
            from prediction_historique import run_historical_validation
            run_historical_validation()
            
        elif module_name == "main_dashboard":
            from main import main as main_dashboard
            main_dashboard()
        
        print(f"✅ {description} - Terminé avec succès")
        return True
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur dans {description}: {e}")
        return False

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    print("🔍 Vérification des dépendances...")
    
    required_packages = [
        'numpy', 'matplotlib', 'seaborn', 'networkx', 
        'colorama', 'requests', 'json'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(Fore.RED + f"❌ Packages manquants : {missing_packages}")
        print("💡 Installez-les avec : pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Toutes les dépendances sont installées")
    return True

def create_directory_structure():
    """Crée la structure de dossiers nécessaire"""
    print("📁 Création de la structure de dossiers...")
    
    directories = [
        "data",
        "visualizations", 
        "config",
        "Data_Processing",
        "geography",
        "markov_model"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Structure de dossiers créée")

def main():
    """Fonction principale d'exécution complète"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           ANALYSE COVID-19 BRUXELLES - VERSION AMÉLIORÉE     ║
    ║                   Graphes Probabilistes et                   ║
    ║                    Chaînes de Markov                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Vérifications préliminaires
    if not check_dependencies():
        return
    
    create_directory_structure()
    
    # Menu interactif
    print("\n🎯 Choisissez le mode d'exécution :")
    print("1. Analyse complète (recommandé)")
    print("2. Étapes individuelles")
    print("3. Validation historique uniquement")
    print("4. Visualisations uniquement")
    
    choice = input("\nVotre choix (1-4): ").strip()
    
    if choice == "1":
        run_complete_analysis()
    elif choice == "2":
        run_individual_steps()
    elif choice == "3":
        run_historical_validation_only()
    elif choice == "4":
        run_visualizations_only()
    else:
        print(Fore.RED + "❌ Choix invalide")

def run_complete_analysis():
    """Exécute l'analyse complète"""
    print("\n🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE")
    
    steps = [
        ("data_downloader", "1. Téléchargement des données COVID-19"),
        ("data_loader", "2. Organisation et nettoyage des données"),
        ("savitzky_golay", "3. Lissage temporel Savitzky-Golay"),
        ("dijkstra", "4. Calcul des poids géographiques (Dijkstra)"),
        ("markov_prediction", "5. Modélisation Markov matricielle"),
        ("historical_validation", "6. Validation historique (Mars-Juin 2021)"),
        ("main_dashboard", "7. Génération des visualisations")
    ]
    
    success_count = 0
    
    for module, description in steps:
        if run_module(module, description):
            success_count += 1
        else:
            print(Fore.YELLOW + f"⚠️ Erreur à l'étape : {description}")
            user_choice = input("Continuer malgré l'erreur ? (o/n): ").lower()
            if user_choice != 'o':
                break
    
    print(f"\n{'='*60}")
    print(f"📊 RÉSUMÉ DE L'EXÉCUTION")
    print(f"{'='*60}")
    print(f"✅ Étapes réussies : {success_count}/{len(steps)}")
    
    if success_count == len(steps):
        print("🎉 ANALYSE COMPLÈTE TERMINÉE AVEC SUCCÈS !")
        print("\n📁 Résultats disponibles dans :")
        print("   - data/ : Fichiers de données et modèles")
        print("   - visualizations/ : Graphiques et analyses")
        
        # Affichage des nouveaux fichiers créés
        print(f"\n📊 NOUVELLES VISUALISATIONS CRÉÉES :")
        print("   - 7_graphes_probabilistes.png : Réseaux géographique et Markov")
        print("   - 8_matrices_markov_analyse.png : Analyse des matrices de transition")
        print("   - 9_validation_historique_pageX.png : Validation Mars-Juin 2021")
        print("   - 10_proprietes_stochastiques.png : Propriétés mathématiques")
    else:
        print("⚠️ Analyse incomplète - Vérifiez les erreurs ci-dessus")

def run_individual_steps():
    """Permet d'exécuter des étapes individuelles"""
    steps = {
        "1": ("data_downloader", "Téléchargement des données"),
        "2": ("data_loader", "Organisation des données"),
        "3": ("savitzky_golay", "Lissage Savitzky-Golay"),
        "4": ("dijkstra", "Calcul géographique Dijkstra"),
        "5": ("markov_prediction", "Modèle de Markov"),
        "6": ("historical_validation", "Validation historique"),
        "7": ("main_dashboard", "Visualisations")
    }
    
    print("\n🔧 ÉTAPES INDIVIDUELLES")
    print("📋 Étapes disponibles :")
    for key, (_, description) in steps.items():
        print(f"   {key}. {description}")
    
    choice = input("\nChoisissez une étape (1-7): ").strip()
    
    if choice in steps:
        module, description = steps[choice]
        run_module(module, description)
    else:
        print(Fore.RED + "❌ Choix invalide")

def run_historical_validation_only():
    """Exécute uniquement la validation historique"""
    print("\n🔍 VALIDATION HISTORIQUE MARS-JUIN 2021")
    
    # Vérifier que les prérequis sont présents
    required_files = [
        "data/smoothed_data.json",
        "data/geographic_weights.json"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(Fore.RED + f"❌ Fichiers manquants : {missing}")
        print("💡 Exécutez d'abord les étapes 1-4 de l'analyse complète")
        return
    
    run_module("historical_validation", "Validation historique Mars-Juin 2021")

def run_visualizations_only():
    """Exécute uniquement les visualisations"""
    print("\n🎨 GÉNÉRATION DES VISUALISATIONS")
    
    # Vérifier que les données sont présentes
    required_files = [
        "data/raw_covid_data_19communes.json",
        "data/smoothed_data.json",
        "data/geographic_weights.json",
        "data/matrix_markov_models.json"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(Fore.RED + f"❌ Fichiers manquants : {missing}")
        print("💡 Exécutez d'abord l'analyse complète ou les étapes 1-5")
        return
    
    run_module("main_dashboard", "Génération des visualisations améliorées")

if __name__ == "__main__":
    main()