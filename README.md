# Prédictions COVID-19 Bruxelles
Modélisation de la propagation du COVID-19 dans les 19 communes de Bruxelles via chaînes de Markov avec contraintes géographiques.

## 🚀 Utilisation

```bash
# Installation des dépendances
pip install numpy matplotlib seaborn networkx colorama requests

# Lancement complet du projet
python run_complete_analysis.py
```

Choisir l'option `1` pour l'analyse complète automatique.

## 📊 Résultats

- **Données traitées :** 457k+ enregistrements COVID-19 (API Sciensano)
- **Lissage Savitzky-Golay :** Filtrage des artefacts administratifs
- **Géographie Dijkstra :** Poids d'influence entre communes  
- **Modèle Markov :** `X⃗(t+1) = A · X⃗(t)` avec corrections de stabilité
- **Visualisations :** 10+ graphiques générés dans `/visualizations/`

## ⚠️ Limitations

**Fonctionne uniquement sur périodes épidémiques stables** (sept. 2021 : MAE=0.69)  
**Échoue sur nouvelles vagues** (mars-juin 2021 : prédictions décroissantes)
