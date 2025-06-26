"""
Lissage Savitzky-Golay pour données COVID-19 Bruxelles
Implémentation du filtre avec fenêtre 7, ordre 3 selon votre spécification
"""

import json
import os
import datetime
from typing import Dict, List, Optional
from colorama import init, Fore

# Initialisation de colorama
init(autoreset=True)


def load_config() -> Dict:
    """Charge la configuration depuis settings.json"""
    config_paths = [
        'config/settings.json',
        '../config/settings.json'
    ]
    
    for config_path in config_paths:
        try:
            with open(config_path, 'r', encoding='utf8') as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError("Configuration settings.json non trouvée")


def savitzky_golay_filter(data: List[float], window_size: int = 7, polynomial_order: int = 3) -> List[float]:
    """
    Applique le filtre Savitzky-Golay selon votre spécification
    
    Coefficients pour fenêtre 7, ordre 3 : [-2, 3, 6, 7, 6, 3, -2] / 21
    
    """
    if len(data) < window_size:
        print(Fore.YELLOW + f"⚠️ Données trop courtes ({len(data)}) pour fenêtre {window_size}")
        return data.copy()
    
    # Coefficients Savitzky-Golay pour fenêtre 7, ordre 3
    coefficients = [-2, 3, 6, 7, 6, 3, -2]
    divisor = 21
    
    smoothed_data = data.copy()
    half_window = window_size // 2
    
    # Application du filtre (évite les bords comme dans votre spécification)
    for i in range(half_window, len(data) - half_window):
        smoothed_value = 0
        
        for j, coeff in enumerate(coefficients):
            data_index = i - half_window + j
            smoothed_value += coeff * data[data_index]
        
        smoothed_data[i] = smoothed_value / divisor
    
    return smoothed_data


def smooth_commune_data(commune_data: Dict[str, int], commune_name: str) -> Dict[str, float]:
    """
    Lisse les données d'une commune
    
    Args:
        commune_data: {date: cases} pour une commune
        commune_name: Nom de la commune
        
    Returns:
        {date: smoothed_cases} données lissées
    """
    if not commune_data:
        return {}
    
    # Tri des dates
    sorted_dates = sorted(commune_data.keys())
    values = [commune_data[date] for date in sorted_dates]
    
    print(f"   📈 {commune_name}: {len(values)} points → lissage...")
    
    # Application du filtre
    smoothed_values = savitzky_golay_filter(values)
    
    # Reconstruction du dictionnaire
    smoothed_data = {}
    for date, smoothed_value in zip(sorted_dates, smoothed_values):
        smoothed_data[date] = max(0, smoothed_value)  # Éviter les valeurs négatives
    
    return smoothed_data


def smooth_covid_data(input_file: str = None, output_file: str = None) -> Optional[str]:
    """
    Pipeline principal de lissage des données COVID-19
    
    Args:
        input_file: Fichier d'entrée (si None, utilise la config)
        output_file: Fichier de sortie (si None, utilise la config)
        
    Returns:
        Chemin du fichier de sortie ou None si erreur
    """
    print("=====================================================")
    print(Fore.YELLOW + "🔄 Lissage Savitzky-Golay des données COVID-19")
    print("=====================================================")
    
    temps_debut = datetime.datetime.now()
    
    # Chargement de la configuration
    config = load_config()
    
    # Détermination des fichiers
    if input_file is None:
        # Chercher le fichier de données dans différents emplacements
        possible_files = [
            'Conversion_Sciensano/COVID19BE_CASES_MUNI.json',
            'Data\COVID19BE_CASES_MUNI.json',  # Votre fichier
            'data/raw_covid_data.json',
            'Espérance_de_maximisation/COVID_19BXL.json'
        ]
        
        input_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                input_file = file_path
                break
        
        if input_file is None:
            print(Fore.RED + "❌ Aucun fichier de données trouvé")
            return None
    
    if output_file is None:
        os.makedirs('data', exist_ok=True)
        output_file = 'data/smoothed_data.json'
    
    print(f"📄 Fichier d'entrée : {Fore.CYAN}{input_file}")
    print(f"💾 Fichier de sortie : {Fore.CYAN}{output_file}")
    
    # Chargement des données
    try:
        with open(input_file, 'r', encoding='utf8') as f:
            raw_data = json.load(f)
        print(f"✅ Données chargées : {len(raw_data)} entrées")
    except Exception as e:
        print(Fore.RED + f"❌ Erreur chargement : {e}")
        return None
    
    # Détection du format des données
    if isinstance(raw_data, list):
        # Format liste d'enregistrements [{DATE, TX_DESCR_FR, CASES}, ...]
        print("📊 Format détecté : Liste d'enregistrements")
        organized_data = organize_data_from_list(raw_data)
    elif isinstance(raw_data, dict):
        # Format organisé {date: {commune: cases}}
        print("📊 Format détecté : Données organisées")
        organized_data = raw_data
    else:
        print(Fore.RED + "❌ Format de données non reconnu")
        return None
    
    if not organized_data:
        print(Fore.RED + "❌ Aucune donnée à traiter")
        return None
    
    print(f"📅 Dates à traiter : {len(organized_data)}")
    
    # Organisation par commune pour le lissage
    print("\n🔄 Réorganisation par commune...")
    commune_series = reorganize_by_commune(organized_data)
    
    if not commune_series:
        print(Fore.RED + "❌ Impossible de réorganiser les données")
        return None
    
    print(f"🏘️ Communes à lisser : {len(commune_series)}")
    for commune in sorted(commune_series.keys()):
        print(f"   - {commune}: {len(commune_series[commune])} points")
    
    # Application du lissage
    print(f"\n📈 Application du filtre Savitzky-Golay...")
    print(f"   - Fenêtre : {config['smoothing']['window_size']}")
    print(f"   - Ordre : {config['smoothing']['polynomial_order']}")
    
    smoothed_commune_series = {}
    for commune_name, commune_data in commune_series.items():
        smoothed_commune_series[commune_name] = smooth_commune_data(commune_data, commune_name)
    
    # Reconversion au format de sortie {date: {commune: cases}}
    print("\n🔄 Reconversion au format de sortie...")
    smoothed_organized_data = reorganize_by_date(smoothed_commune_series)
    
    # Sauvegarde
    try:
        # Ajout de métadonnées
        output_data = {
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "source_file": input_file,
                "smoothing_method": config['smoothing']['method'],
                "window_size": config['smoothing']['window_size'],
                "polynomial_order": config['smoothing']['polynomial_order'],
                "communes_count": len(smoothed_commune_series),
                "dates_count": len(smoothed_organized_data)
            },
            "data": smoothed_organized_data
        }
        
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Données lissées sauvegardées : {Fore.CYAN}{output_file}")
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur sauvegarde : {e}")
        return None
    
    # Statistiques finales
    temps_fin = datetime.datetime.now()
    duree = temps_fin - temps_debut
    
    print(f"\n📊 Statistiques finales :")
    print(f"   - Communes traitées : {len(smoothed_commune_series)}")
    print(f"   - Dates traitées : {len(smoothed_organized_data)}")
    print(f"   - Temps d'exécution : {Fore.GREEN}{duree}")
    print("✅ Lissage terminé avec succès !")
    
    return output_file


def organize_data_from_list(raw_data: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Convertit une liste d'enregistrements en format organisé par date
    
    Args:
        raw_data: [{DATE, TX_DESCR_FR, CASES}, ...]
        
    Returns:
        {date: {commune: cases}}
    """
    organized = {}
    config = load_config()
    communes_bruxelles = set([
        "Anderlecht", "Auderghem", "Berchem-Sainte-Agathe", "Bruxelles",
        "Etterbeek", "Evere", "Forest (Bruxelles-Capitale)", "Ganshoren", "Ixelles", "Jette",
        "Koekelberg", "Molenbeek-Saint-Jean", "Saint-Gilles", "Saint-Josse-ten-Noode",
        "Schaerbeek", "Uccle", "Watermael-Boitsfort", "Woluwe-Saint-Lambert", "Woluwe-Saint-Pierre"
    ])
    
    for record in raw_data:
        date = record.get('DATE')
        commune = record.get('TX_DESCR_FR')
        cases = record.get('CASES')
        
        # Filtrer uniquement les communes de Bruxelles
        if commune in communes_bruxelles and date and cases is not None:
            try:
                cases = int(cases) if cases != "<5" else 1
                organized.setdefault(date, {})[commune] = cases
            except (ValueError, TypeError):
                continue
    
    return organized


def reorganize_by_commune(organized_data: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """
    Réorganise les données par commune : {commune: {date: cases}}
    
    Args:
        organized_data: {date: {commune: cases}}
        
    Returns:
        {commune: {date: cases}}
    """
    by_commune = {}
    
    for date, communes_data in organized_data.items():
        for commune, cases in communes_data.items():
            by_commune.setdefault(commune, {})[date] = cases
    
    return by_commune


def reorganize_by_date(commune_series: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Reconvertit les données par date : {date: {commune: cases}}
    
    Args:
        commune_series: {commune: {date: cases}}
        
    Returns:
        {date: {commune: cases}}
    """
    by_date = {}
    
    for commune, dates_data in commune_series.items():
        for date, cases in dates_data.items():
            by_date.setdefault(date, {})[commune] = cases
    
    return by_date


def test_smoothing():
    """Test du module de lissage"""
    print("🧪 Test du module de lissage...")
    
    # Test avec données factices
    test_data = [1, 3, 2, 5, 4, 6, 8, 7, 9, 10, 12, 11, 13, 15, 14]
    print(f"Données test : {test_data}")
    
    smoothed = savitzky_golay_filter(test_data)
    print(f"Données lissées : {[round(x, 2) for x in smoothed]}")
    
    # Test avec vraies données si disponibles
    result = smooth_covid_data()
    
    if result:
        print(f"✅ Test réussi ! Fichier généré : {result}")
        
        # Vérification du fichier
        try:
            with open(result, 'r', encoding='utf8') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            print(f"📊 Métadonnées : {metadata.get('communes_count')} communes, {metadata.get('dates_count')} dates")
            
        except Exception as e:
            print(f"⚠️ Erreur vérification : {e}")
    else:
        print("❌ Test échoué")


if __name__ == "__main__":
    test_smoothing()