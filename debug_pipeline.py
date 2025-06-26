"""
Script de débogage pour identifier pourquoi le pipeline ne fonctionne pas
"""

import os
import sys
from pathlib import Path

def debug_step_by_step():
    """Débogue chaque étape du pipeline individuellement"""
    print("🔍 DÉBOGAGE ÉTAPE PAR ÉTAPE")
    print("="*50)
    
    # Étape 1: Vérifier la structure des fichiers
    print("\n1️⃣ VÉRIFICATION DE LA STRUCTURE")
    print("-"*30)
    
    required_modules = [
        "Data_Processing/data_downloader.py",
        "Data_Processing/data_loader.py", 
        "Data_Processing/savitzky_golay.py",
        "geography/dijkstra.py",
        "markov_model/Prediction.py",
        "config/settings.json",
        "main.py"
    ]
    
    missing_modules = []
    for module in required_modules:
        if os.path.exists(module):
            print(f"✅ {module}")
        else:
            print(f"❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️ {len(missing_modules)} fichiers manquants !")
        return False
    
    # Étape 2: Tester l'import des modules
    print("\n2️⃣ TEST D'IMPORT DES MODULES")
    print("-"*30)
    
    try:
        sys.path.append('.')
        from Data_Processing.data_downloader import test_export_download
        print("✅ data_downloader importé")
    except Exception as e:
        print(f"❌ data_downloader: {e}")
        return False
    
    try:
        from Data_Processing.data_loader import load_covid_data_pipeline
        print("✅ data_loader importé")
    except Exception as e:
        print(f"❌ data_loader: {e}")
        return False
    
    try:
        from Data_Processing.savitzky_golay import smooth_covid_data
        print("✅ savitzky_golay importé")
    except Exception as e:
        print(f"❌ savitzky_golay: {e}")
        return False
    
    # Étape 3: Tester le téléchargement des données
    print("\n3️⃣ TEST DE TÉLÉCHARGEMENT")
    print("-"*30)
    
    try:
        print("🌐 Tentative de téléchargement...")
        result = test_export_download()
        if result and os.path.exists(result):
            print(f"✅ Téléchargement réussi: {result}")
        else:
            print(f"❌ Téléchargement échoué")
            return False
    except Exception as e:
        print(f"❌ Erreur téléchargement: {e}")
        return False
    
    # Étape 4: Vérifier le fichier téléchargé
    print("\n4️⃣ VÉRIFICATION DU FICHIER TÉLÉCHARGÉ")
    print("-"*30)
    
    possible_files = [
        "Data/COVID19BE_CASES_MUNI.json",
        "COVID19BE_CASES_MUNI.json",
        "data/COVID19BE_CASES_MUNI.json"
    ]
    
    data_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            data_file = file_path
            print(f"✅ Fichier trouvé: {file_path}")
            break
    
    if not data_file:
        print("❌ Aucun fichier de données trouvé")
        return False
    
    # Vérifier le contenu du fichier
    try:
        import json
        with open(data_file, 'r', encoding='utf8') as f:
            data = json.load(f)
        
        print(f"📊 Fichier contient {len(data)} enregistrements")
        
        # Vérifier la structure
        if isinstance(data, list) and len(data) > 0:
            first_record = data[0]
            required_fields = ['DATE', 'TX_DESCR_FR', 'CASES']
            
            for field in required_fields:
                if field in first_record:
                    print(f"✅ Champ {field} présent")
                else:
                    print(f"❌ Champ {field} manquant")
                    return False
        else:
            print("❌ Structure de données invalide")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lecture fichier: {e}")
        return False
    
    print("\n✅ Toutes les vérifications passées !")
    return True

def test_data_loader_individually():
    """Teste spécifiquement le data_loader"""
    print("\n🔄 TEST SPÉCIFIQUE DU DATA_LOADER")
    print("-"*40)
    
    try:
        from Data_Processing.data_loader import load_covid_data_pipeline
        
        # Chercher le fichier de données
        data_files = [
            "Data/COVID19BE_CASES_MUNI.json",
            "COVID19BE_CASES_MUNI.json",
            "data/COVID19BE_CASES_MUNI.json"
        ]
        
        data_file = None
        for file_path in data_files:
            if os.path.exists(file_path):
                data_file = file_path
                break
        
        if not data_file:
            print("❌ Aucun fichier de données trouvé pour le data_loader")
            return False
        
        print(f"📄 Utilisation du fichier: {data_file}")
        
        # Exécuter le pipeline
        result = load_covid_data_pipeline(data_file)
        
        if result:
            print("✅ Data loader terminé avec succès")
            return True
        else:
            print("❌ Data loader a échoué")
            return False
            
    except Exception as e:
        print(f"❌ Erreur dans data_loader: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_each_step_individually():
    """Teste chaque étape individuellement avec gestion d'erreur"""
    print("\n🧪 TEST INDIVIDUEL DE CHAQUE ÉTAPE")
    print("="*50)
    
    steps = [
        ("téléchargement", test_download_step),
        ("data_loader", test_data_loader_step),
        ("lissage", test_smoothing_step),
        ("géographie", test_geography_step),
        ("markov", test_markov_step)
    ]
    
    results = {}
    
    for step_name, step_function in steps:
        print(f"\n🔧 Test: {step_name}")
        print("-" * 20)
        
        try:
            success = step_function()
            results[step_name] = success
            if success:
                print(f"✅ {step_name} réussi")
            else:
                print(f"❌ {step_name} échoué")
        except Exception as e:
            print(f"❌ {step_name} erreur: {e}")
            results[step_name] = False
    
    # Résumé
    print(f"\n📊 RÉSUMÉ DES TESTS")
    print("-" * 20)
    for step_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
    
    return results

def test_download_step():
    """Teste uniquement l'étape de téléchargement"""
    try:
        from Data_Processing.data_downloader import test_export_download
        
        # Créer le dossier Data s'il n'existe pas
        os.makedirs("Data", exist_ok=True)
        
        result = test_export_download()
        return result is not None and os.path.exists(result if result else "")
    except Exception as e:
        print(f"Erreur téléchargement: {e}")
        return False

def test_data_loader_step():
    """Teste uniquement l'étape de data_loader"""
    try:
        # Vérifier qu'on a un fichier de données
        data_files = ["Data/COVID19BE_CASES_MUNI.json", "COVID19BE_CASES_MUNI.json"]
        data_file = None
        
        for file_path in data_files:
            if os.path.exists(file_path):
                data_file = file_path
                break
        
        if not data_file:
            print("Pas de fichier de données pour data_loader")
            return False
        
        from Data_Processing.data_loader import load_covid_data_pipeline
        result = load_covid_data_pipeline(data_file)
        
        # Vérifier que le fichier de sortie a été créé
        expected_output = "data/raw_covid_data_19communes.json"
        return os.path.exists(expected_output)
        
    except Exception as e:
        print(f"Erreur data_loader: {e}")
        return False

def test_smoothing_step():
    """Teste uniquement l'étape de lissage"""
    try:
        # Vérifier qu'on a les données d'entrée
        input_file = "data/raw_covid_data_19communes.json"
        if not os.path.exists(input_file):
            print("Pas de fichier d'entrée pour le lissage")
            return False
        
        from Data_Processing.savitzky_golay import smooth_covid_data
        result = smooth_covid_data()
        
        return result is not None and os.path.exists("data/smoothed_data.json")
        
    except Exception as e:
        print(f"Erreur lissage: {e}")
        return False

def test_geography_step():
    """Teste uniquement l'étape géographique"""
    try:
        from geography.dijkstra import test_dijkstra
        test_dijkstra()
        
        return os.path.exists("data/geographic_weights.json")
        
    except Exception as e:
        print(f"Erreur géographie: {e}")
        return False

def test_markov_step():
    """Teste uniquement l'étape Markov"""
    try:
        # Vérifier les prérequis
        required_files = [
            "data/smoothed_data.json",
            "data/geographic_weights.json"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Fichier requis manquant: {file_path}")
                return False
        
        from markov_model.Prediction import test_matrix_markov
        test_matrix_markov()
        
        return os.path.exists("data/matrix_markov_models.json")
        
    except Exception as e:
        print(f"Erreur Markov: {e}")
        return False

def quick_fix_attempt():
    """Tentative de correction rapide"""
    print("\n🚀 TENTATIVE DE CORRECTION RAPIDE")
    print("="*40)
    
    # 1. Créer tous les dossiers nécessaires
    print("📁 Création des dossiers...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("Data", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # 2. Tester étape par étape
    return test_each_step_individually()

if __name__ == "__main__":
    print("🔍 OUTIL DE DÉBOGAGE PIPELINE COVID-19")
    print("="*50)
    
    print("\nQue voulez-vous faire ?")
    print("1. Débogage complet étape par étape")
    print("2. Test rapide de chaque module")
    print("3. Tentative de correction automatique")
    
    choice = input("\nVotre choix (1-3): ").strip()
    
    if choice == "1":
        debug_step_by_step()
    elif choice == "2":
        test_each_step_individually()
    elif choice == "3":
        quick_fix_attempt()
    else:
        print("Choix invalide, démarrage du débogage complet...")
        debug_step_by_step()