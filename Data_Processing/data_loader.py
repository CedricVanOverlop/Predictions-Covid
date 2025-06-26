"""
Data Loader pour COVID-19 Bruxelles
Adaptation du code original pour les 19 communes de Bruxelles
Chargement et conversion des données Sciensano
"""

import json
import os
import datetime
from typing import Dict, List, Optional
from colorama import init, Fore

# Initialisation de colorama pour les couleurs
init(autoreset=True)


def load_config() -> Dict:
    """Charge la configuration depuis settings.json"""
    try:
        with open('config/settings.json', 'r', encoding='utf8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(Fore.RED + "Erreur : Fichier config/settings.json introuvable")
        raise
    except json.JSONDecodeError:
        print(Fore.RED + "Erreur : Format JSON invalide dans settings.json")
        raise


def load_raw_covid_data(data_file_path: str = None) -> Optional[Dict]:
    """
    Charge les données brutes COVID depuis le fichier Sciensano
    
    Args:
        data_file_path: Chemin vers le fichier JSON Sciensano
        
    Returns:
        Dictionnaire avec les données brutes ou None si erreur
    """
    # Charger la config pour le nom du fichier par défaut
    config = load_config()
    
    if data_file_path is None:
        data_file_path = config['data_sources']['sciensano_file']
    
    print("=====================================================")
    print(Fore.YELLOW + "🔄 Chargement des données COVID-19")
    print("=====================================================")
    
    temps_debut = datetime.datetime.now()
    
    try:
        # Tentative de lecture du fichier
        with open(data_file_path, "r", encoding='utf8') as f:
            data = json.load(f)
            
        print(f"✅ Fichier trouvé : {Fore.CYAN}{data_file_path}")
        print(f"📊 Nombre d'entrées : {Fore.YELLOW}{len(data)}")
        
        return data
        
    except FileNotFoundError:
        print(Fore.RED + f"❌ Erreur : Fichier {data_file_path} introuvable")
        print(Fore.YELLOW + "💡 Assurez-vous d'avoir téléchargé les données depuis :")
        print("   https://epistat.wiv-isp.be/covid/")
        return None
        
    except json.JSONDecodeError as e:
        print(Fore.RED + f"❌ Erreur de format JSON : {e}")
        return None
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur inattendue : {e}")
        return None


def convert_raw_data_to_communes(raw_data: List[Dict]) -> Dict[str, Dict[str, int]]:
    
    config = load_config()
    communes_list = config['communes']
    
    print(f"🏘️  Traitement des {len(communes_list)} communes de Bruxelles...")
    
    # Dictionnaire résultat : {date: {commune: cases}}
    organized_data = {}
    
    # Compteurs pour le suivi
    total_entries = 0
    processed_entries = 0
    communes_found = set()
    
    for item in raw_data:
        total_entries += 1
        
        # Récupération des informations de l'entrée
        commune = item.get("TX_DESCR_FR")  # Nom de la commune en français
        date = item.get('DATE')
        cases = item.get('CASES')
        
        # Vérifier si c'est une commune de Bruxelles
        if commune in communes_list:
            processed_entries += 1
            communes_found.add(commune)
            
            # Conversion des cas "<5" en 1 (anonymisation Sciensano)
            if cases == "<5":
                cases = 1
            else:
                try:
                    cases = int(cases)
                except (ValueError, TypeError):
                    print(Fore.YELLOW + f"⚠️  Valeur invalide pour {commune} le {date}: {cases}")
                    cases = 0
            
            # Organisation par date puis par commune
            if date not in organized_data:
                organized_data[date] = {}
            
            organized_data[date][commune] = cases
    
    # Affichage des statistiques
    print(f"📈 Entrées traitées : {Fore.GREEN}{processed_entries}{Fore.RESET}/{total_entries}")
    print(f"🏘️  Communes trouvées : {Fore.GREEN}{len(communes_found)}{Fore.RESET}/{len(communes_list)}")
    print(f"📅 Dates uniques : {Fore.GREEN}{len(organized_data)}")
    
    # Vérification des communes manquantes
    communes_manquantes = set(communes_list) - communes_found
    if communes_manquantes:
        print(Fore.YELLOW + f"⚠️  Communes manquantes dans les données :")
        for commune in sorted(communes_manquantes):
            print(f"   - {commune}")
    
    return organized_data


def fill_missing_data(organized_data: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """
    Remplit les données manquantes pour certaines communes/dates
    
    Args:
        organized_data: Données organisées par date et commune
        
    Returns:
        Données complétées avec les valeurs manquantes
    """
    config = load_config()
    communes_list = config['communes']
    
    print("🔧 Completion des données manquantes...")
    
    dates = sorted(organized_data.keys())
    filled_count = 0
    
    for date in dates:
        for commune in communes_list:
            if commune not in organized_data[date]:
                # Valeur par défaut : 0 (pas de données = pas de cas rapportés)
                organized_data[date][commune] = 0
                filled_count += 1
    
    if filled_count > 0:
        print(f"✅ {filled_count} valeurs manquantes complétées avec 0")
    
    return organized_data


def save_processed_data(organized_data: Dict, output_file: str = None) -> bool:
    """
    Sauvegarde les données traitées dans un fichier JSON
    
    Args:
        organized_data: Données organisées à sauvegarder
        output_file: Nom du fichier de sortie
        
    Returns:
        True si succès, False sinon
    """
    config = load_config()
    
    if output_file is None:
        # Créer le nom du fichier avec toutes les communes
        output_file = f"{config['cache']['data_directory']}/raw_covid_data_19communes.json"
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Ajout de métadonnées
        save_data = {
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "communes_count": len(config['communes']),
                "dates_count": len(organized_data),
                "source": "Sciensano COVID-19 Belgium",
                "communes": config['communes']
            },
            "data": organized_data
        }
        
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Données sauvegardées : {Fore.CYAN}{output_file}")
        return True
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur lors de la sauvegarde : {e}")
        return False


def load_covid_data_pipeline(data_file_path: str = None) -> Optional[Dict]:
    """
    Pipeline complète de chargement des données COVID-19
    
    Args:
        data_file_path: Chemin vers le fichier Sciensano (optionnel)
        
    Returns:
        Données organisées par date et commune, ou None si erreur
    """
    print(Fore.CYAN + "🚀 Démarrage du pipeline de chargement des données")
    temps_debut = datetime.datetime.now()
    
    # Étape 1: Chargement des données brutes
    raw_data = load_raw_covid_data(data_file_path)
    if raw_data is None:
        return None
    
    # Étape 2: Conversion et organisation
    organized_data = convert_raw_data_to_communes(raw_data)
    if not organized_data:
        print(Fore.RED + "❌ Aucune donnée convertie")
        return None
    
    # Étape 3: Completion des données manquantes
    complete_data = fill_missing_data(organized_data)
    
    # Étape 4: Sauvegarde
    success = save_processed_data(complete_data)
    if not success:
        print(Fore.YELLOW + "⚠️  Données traitées mais pas sauvegardées")
    
    # Temps d'exécution
    temps_fin = datetime.datetime.now()
    duree = temps_fin - temps_debut
    print(f"⏱️  Pipeline terminé en {Fore.GREEN}{duree}")
    print("✅ Données prêtes pour le lissage !")
    
    return complete_data


# Fonction de test pour vérifier le bon fonctionnement
def test_data_loader():
    """Test du chargeur de données"""
    print("🧪 Test du data_loader...")
    
    # Test du chargement de config
    try:
        config = load_config()
        print(f"✅ Config chargée : {len(config['communes'])} communes")
    except Exception as e:
        print(f"❌ Erreur config : {e}")
        return
    
    # Pour tester avec des données factices si le vrai fichier n'existe pas
    print("💡 Pour tester avec de vraies données, placez COVID19BE_CASES_MUNI.json")
    print("   dans le dossier racine et appelez load_covid_data_pipeline()")


if __name__ == "__main__":
    # Exécution du test si le fichier est lancé directement
    test_data_loader()