"""
Algorithme de Dijkstra pour calculer les poids géographiques
entre les 19 communes de Bruxelles
"""

import json
import os
from typing import Dict, List, Tuple
from colorama import init, Fore

# Initialisation de colorama
init(autoreset=True)


class BrusselsGeography:
    """
    Gestion de la géographie des communes de Bruxelles
    """
    
    def __init__(self):
        # Les 19 communes de Bruxelles
        self.communes = [
            "Anderlecht", "Auderghem", "Berchem-Sainte-Agathe", "Bruxelles",
            "Etterbeek", "Evere", "Forest (Bruxelles-Capitale)", "Ganshoren", 
            "Ixelles", "Jette", "Koekelberg", "Molenbeek-Saint-Jean", 
            "Saint-Gilles", "Saint-Josse-ten-Noode", "Schaerbeek", "Uccle", 
            "Watermael-Boitsfort", "Woluwe-Saint-Lambert", "Woluwe-Saint-Pierre"
        ]
        
        # Graphe des adjacences (communes qui partagent une frontière)
        self.adjacencies = self._build_adjacency_graph()
        
        # Longueurs des frontières (en km, estimées)
        self.frontier_lengths = self._build_frontier_lengths()
    
    def _build_adjacency_graph(self) -> Dict[str, List[str]]:
        """
        Construit le graphe d'adjacence des communes de Bruxelles
        Basé sur la géographie réelle des communes
        """
        adjacencies = {
            "Bruxelles": ["Saint-Josse-ten-Noode", "Schaerbeek", "Evere", "Saint-Gilles", "Ixelles", "Etterbeek"],
            
            "Anderlecht": ["Molenbeek-Saint-Jean", "Berchem-Sainte-Agathe", "Koekelberg", "Forest (Bruxelles-Capitale)", "Saint-Gilles"],
            
            "Auderghem": ["Ixelles", "Etterbeek", "Woluwe-Saint-Pierre", "Watermael-Boitsfort"],
            
            "Berchem-Sainte-Agathe": ["Ganshoren", "Jette", "Koekelberg", "Molenbeek-Saint-Jean", "Anderlecht"],
            
            "Etterbeek": ["Bruxelles", "Ixelles", "Woluwe-Saint-Lambert", "Woluwe-Saint-Pierre", "Auderghem"],
            
            "Evere": ["Bruxelles", "Schaerbeek", "Woluwe-Saint-Lambert"],
            
            "Forest (Bruxelles-Capitale)": ["Saint-Gilles", "Ixelles", "Uccle", "Anderlecht"],
            
            "Ganshoren": ["Jette", "Koekelberg", "Berchem-Sainte-Agathe"],
            
            "Ixelles": ["Bruxelles", "Saint-Gilles", "Forest (Bruxelles-Capitale)", "Uccle", "Etterbeek", "Auderghem"],
            
            "Jette": ["Ganshoren", "Koekelberg", "Berchem-Sainte-Agathe"],
            
            "Koekelberg": ["Molenbeek-Saint-Jean", "Berchem-Sainte-Agathe", "Ganshoren", "Jette", "Anderlecht"],
            
            "Molenbeek-Saint-Jean": ["Saint-Josse-ten-Noode", "Koekelberg", "Berchem-Sainte-Agathe", "Anderlecht"],
            
            "Saint-Gilles": ["Bruxelles", "Forest (Bruxelles-Capitale)", "Ixelles", "Anderlecht"],
            
            "Saint-Josse-ten-Noode": ["Bruxelles", "Schaerbeek", "Molenbeek-Saint-Jean"],
            
            "Schaerbeek": ["Bruxelles", "Saint-Josse-ten-Noode", "Evere", "Woluwe-Saint-Lambert"],
            
            "Uccle": ["Forest (Bruxelles-Capitale)", "Ixelles", "Watermael-Boitsfort"],
            
            "Watermael-Boitsfort": ["Auderghem", "Woluwe-Saint-Pierre", "Uccle"],
            
            "Woluwe-Saint-Lambert": ["Etterbeek", "Woluwe-Saint-Pierre", "Evere", "Schaerbeek"],
            
            "Woluwe-Saint-Pierre": ["Etterbeek", "Auderghem", "Watermael-Boitsfort", "Woluwe-Saint-Lambert"]
        }
        
        return adjacencies
    
    def _build_frontier_lengths(self) -> Dict[Tuple[str, str], float]:
        """
        Construit les longueurs des frontières entre communes adjacentes
        Valeurs estimées en kilomètres
        """
        frontier_lengths = {
            # Bruxelles (centre) - frontières importantes
            ("Bruxelles", "Saint-Josse-ten-Noode"): 2.1,
            ("Bruxelles", "Schaerbeek"): 3.5,
            ("Bruxelles", "Evere"): 2.8,
            ("Bruxelles", "Saint-Gilles"): 2.9,
            ("Bruxelles", "Ixelles"): 4.2,
            ("Bruxelles", "Etterbeek"): 3.1,
            
            # Anderlecht - grande commune
            ("Anderlecht", "Molenbeek-Saint-Jean"): 4.1,
            ("Anderlecht", "Berchem-Sainte-Agathe"): 2.8,
            ("Anderlecht", "Koekelberg"): 2.3,
            ("Anderlecht", "Forest (Bruxelles-Capitale)"): 3.7,
            ("Anderlecht", "Saint-Gilles"): 3.2,
            
            # Autres frontières
            ("Auderghem", "Ixelles"): 2.6,
            ("Auderghem", "Etterbeek"): 2.1,
            ("Auderghem", "Woluwe-Saint-Pierre"): 3.4,
            ("Auderghem", "Watermael-Boitsfort"): 4.2,
            
            ("Berchem-Sainte-Agathe", "Ganshoren"): 2.5,
            ("Berchem-Sainte-Agathe", "Jette"): 1.8,
            ("Berchem-Sainte-Agathe", "Koekelberg"): 2.2,
            ("Berchem-Sainte-Agathe", "Molenbeek-Saint-Jean"): 3.1,
            
            ("Etterbeek", "Ixelles"): 2.8,
            ("Etterbeek", "Woluwe-Saint-Lambert"): 2.4,
            ("Etterbeek", "Woluwe-Saint-Pierre"): 2.9,
            
            ("Evere", "Schaerbeek"): 3.6,
            ("Evere", "Woluwe-Saint-Lambert"): 2.7,
            
            ("Forest (Bruxelles-Capitale)", "Saint-Gilles"): 2.4,
            ("Forest (Bruxelles-Capitale)", "Ixelles"): 3.1,
            ("Forest (Bruxelles-Capitale)", "Uccle"): 4.8,
            
            ("Ganshoren", "Koekelberg"): 1.9,
            ("Ganshoren", "Jette"): 2.3,
            
            ("Ixelles", "Saint-Gilles"): 2.7,
            ("Ixelles", "Uccle"): 3.9,
            
            ("Jette", "Koekelberg"): 2.1,
            
            ("Koekelberg", "Molenbeek-Saint-Jean"): 2.4,
            
            ("Molenbeek-Saint-Jean", "Saint-Josse-ten-Noode"): 2.8,
            
            ("Saint-Josse-ten-Noode", "Schaerbeek"): 2.6,
            
            ("Schaerbeek", "Woluwe-Saint-Lambert"): 3.2,
            
            ("Uccle", "Watermael-Boitsfort"): 5.1,
            
            ("Watermael-Boitsfort", "Woluwe-Saint-Pierre"): 3.7,
            
            ("Woluwe-Saint-Lambert", "Woluwe-Saint-Pierre"): 2.8
        }
        
        # Ajouter les frontières symétriques
        symmetric_frontiers = {}
        for (a, b), length in frontier_lengths.items():
            symmetric_frontiers[(a, b)] = length
            symmetric_frontiers[(b, a)] = length
        
        return symmetric_frontiers
    
    def get_frontier_length(self, commune1: str, commune2: str) -> float:
        """
        Retourne la longueur de la frontière entre deux communes
        
        Args:
            commune1: Première commune
            commune2: Deuxième commune
            
        Returns:
            Longueur de la frontière (0 si pas adjacentes)
        """
        return self.frontier_lengths.get((commune1, commune2), 0.0)
    
    def calculate_geographic_weight(self, commune1: str, commune2: str, epsilon: float = 0.1) -> float:
        """
        Calcule le poids géographique selon votre formule : 1/(frontier_length + epsilon)
        
        """
        if commune1 == commune2:
            return 1.0  # Poids maximum pour la même commune
        
        frontier_length = self.get_frontier_length(commune1, commune2)
        
        if frontier_length == 0:
            return 0.0  # Pas de frontière = pas d'influence
        
        return 1.0 / (frontier_length + epsilon)
    
    def dijkstra_weights(self, source: str, epsilon: float = 0.1) -> Dict[str, float]:
        """
        Calcule les poids Dijkstra depuis une commune source
        """
        if source not in self.communes:
            raise ValueError(f"Commune '{source}' non reconnue")
        
        # Initialisation Dijkstra
        distances = {commune: float('inf') for commune in self.communes}
        distances[source] = 0.0
        
        visited = set()
        unvisited = set(self.communes)
        
        while unvisited:
            # Trouver la commune non visitée avec la plus petite distance
            current = min(unvisited, key=lambda x: distances[x])
            
            if distances[current] == float('inf'):
                break  # Plus de communes accessibles
            
            # Visiter les voisins
            for neighbor in self.adjacencies.get(current, []):
                if neighbor in visited:
                    continue
                
                # Distance = inverse du poids géographique
                weight = self.calculate_geographic_weight(current, neighbor, epsilon)
                
                if weight > 0:
                    distance = 1.0 / weight
                    new_distance = distances[current] + distance
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
            
            visited.add(current)
            unvisited.remove(current)
        
        # Convertir les distances en poids (inverse)
        weights = {}
        for commune, distance in distances.items():
            if distance == float('inf'):
                weights[commune] = 0.0
            elif distance == 0.0:
                weights[commune] = 1.0
            else:
                weights[commune] = 1.0 / distance
        
        return weights
    
    def calculate_all_geographic_weights(self, epsilon: float = 0.1) -> Dict[str, Dict[str, float]]:
        """
        Calcule tous les poids géographiques entre toutes les communes
        """
        print("🗺️ Calcul des poids géographiques avec Dijkstra...")
        
        all_weights = {}
        
        for source_commune in self.communes:
            print(f"   📍 Calcul depuis {source_commune}...")
            weights = self.dijkstra_weights(source_commune, epsilon)
            all_weights[source_commune] = weights
        
        print("✅ Poids géographiques calculés !")
        return all_weights
    
    def save_weights(self, weights: Dict[str, Dict[str, float]], output_file: str = "data/geographic_weights.json"):
        """
        Sauvegarde les poids géographiques
        
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        output_data = {
            "metadata": {
                "created_at": "2025-01-19",
                "method": "dijkstra",
                "epsilon": 0.1,
                "communes_count": len(self.communes)
            },
            "weights": weights
        }
        
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Poids sauvegardés : {output_file}")
    
    def print_adjacency_info(self):
        """Affiche les informations sur les adjacences"""
        print("\n🗺️ Informations géographiques :")
        print(f"   - Communes : {len(self.communes)}")
        print(f"   - Frontières : {len(self.frontier_lengths) // 2}")
        
        print("\n📍 Adjacences par commune :")
        for commune in sorted(self.communes):
            neighbors = self.adjacencies.get(commune, [])
            print(f"   - {commune}: {len(neighbors)} voisins")


def test_dijkstra():
    """Test de l'algorithme de Dijkstra géographique"""
    print("🧪 Test de l'algorithme de Dijkstra géographique...")
    
    geo = BrusselsGeography()
    
    # Afficher les informations
    geo.print_adjacency_info()
    
    # Test sur une commune
    print(f"\n🔍 Test des poids depuis Bruxelles :")
    weights_from_brussels = geo.dijkstra_weights("Bruxelles", epsilon=0.1)
    
    for commune, weight in sorted(weights_from_brussels.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {commune}: {weight:.4f}")
    
    # Calcul complet
    print(f"\n⚡ Calcul de tous les poids...")
    all_weights = geo.calculate_all_geographic_weights(epsilon=0.1)
    
    # Sauvegarde
    geo.save_weights(all_weights)
    
    print("✅ Test terminé !")


if __name__ == "__main__":
    test_dijkstra()