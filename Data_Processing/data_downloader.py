"""
Version alternative avec l'API Export (pas de limite de 10k)
Télécharge TOUT le dataset COVID-19 depuis Opendatasoft
"""

import requests
import json
import os
import datetime
from colorama import init, Fore

init(autoreset=True)

def download_via_export_api():
    """
    Télécharge via l'API Export qui n'a pas de limite de 10k
    """
    print("=====================================================")
    print(Fore.YELLOW + "Téléchargement via API Export (dataset complet)")
    print("=====================================================")
    
    temps_debut = datetime.datetime.now()
    
    # API Export URL - télécharge TOUT
    export_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/covid-19-pandemic-belgium-cases-municipality/exports/json"
    
    try:
        print("🌐 Téléchargement du dataset complet...")
        print("⏳ Cela peut prendre 1-2 minutes...")
        
        response = requests.get(export_url, timeout=180)  # 3 minutes timeout
        response.raise_for_status()
        
        # L'API Export retourne directement un JSON
        data = response.json()
        print(f"✅ {len(data)} enregistrements téléchargés")
        
        # Créer le dossier
        os.makedirs("Data", exist_ok=True)
        
        # Conversion au format attendu par votre conversion.py
        print("🔄 Conversion au format attendu...")
        converted_data = []
        
        for item in data:
            converted_record = {
                "DATE": item.get("date"),
                "TX_DESCR_FR": item.get("tx_descr_fr"), 
                "CASES": item.get("cases")
            }
            
            # Validation des données essentielles
            if converted_record["DATE"] and converted_record["TX_DESCR_FR"]:
                converted_data.append(converted_record)
        
        # Sauvegarde
        file_path = "Data/COVID19BE_CASES_MUNI.json"
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        temps_fin = datetime.datetime.now()
        duree = temps_fin - temps_debut
        
        print(f"✅ {len(converted_data)} enregistrements sauvegardés dans {file_path}")
        print(f"⏱️ Téléchargement terminé en {Fore.GREEN}{duree}")
        
        return file_path
        
    except Exception as e:
        print(Fore.RED + f"❌ Erreur : {e}")
        return None

def test_export_download():
    """
    Test du téléchargement export avec analyse des communes
    """
    # Les 19 communes de Bruxelles (noms corrigés)
    COMMUNES_BXL = [
        "Anderlecht", "Auderghem", "Berchem-Sainte-Agathe", "Bruxelles", 
        "Etterbeek", "Evere", "Forest (Bruxelles-Capitale)", "Ganshoren", "Ixelles", "Jette",  
        "Koekelberg", "Molenbeek-Saint-Jean", "Saint-Gilles", "Saint-Josse-ten-Noode",
        "Schaerbeek", "Uccle", "Watermael-Boitsfort", "Woluwe-Saint-Lambert", "Woluwe-Saint-Pierre"
    ]
    
    # Téléchargement
    result = download_via_export_api()
    
    if not result:
        print("❌ Téléchargement échoué")
        return
    
    # Analyse rapide
    print("\n🔍 Analyse rapide des communes de Bruxelles...")
    
    try:
        with open(result, 'r', encoding='utf8') as f:
            data = json.load(f)
        
        # Compter les communes de Bruxelles
        communes_trouvees = set()
        entries_bxl = 0
        
        for item in data:
            commune = item.get("TX_DESCR_FR")
            if commune in COMMUNES_BXL:
                communes_trouvees.add(commune)
                entries_bxl += 1
        
        print(f"📊 Résultat:")
        print(f"   - Total enregistrements: {len(data)}")
        print(f"   - Enregistrements Bruxelles: {entries_bxl}")
        print(f"   - Communes trouvées: {len(communes_trouvees)}/19")
        
        if len(communes_trouvees) == 19:
            print("🎉 Toutes les 19 communes trouvées !")
        else:
            communes_manquantes = set(COMMUNES_BXL) - communes_trouvees
            print(f"❌ Communes manquantes: {sorted(communes_manquantes)}")
        
        print(f"✅ Communes trouvées: {sorted(communes_trouvees)}")
        
    except Exception as e:
        print(f"❌ Erreur analyse: {e}")

if __name__ == "__main__":
    test_export_download()