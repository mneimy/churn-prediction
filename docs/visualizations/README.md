# 📊 Visualisations - Storytelling Data Science

Ce dossier contient toutes les visualisations créées pour raconter l'histoire du projet et démontrer les compétences en Data Science.

## 🎯 Objectif

Créer des visualisations interactives qui :
- **Racontent une histoire** : Du problème business aux résultats
- **Démontrent les compétences** : Data Science, storytelling, visualisation
- **Sont prêtes pour le portfolio** : Professionnelles et impactantes

## 📁 Fichiers

### Scripts Python

- `visualizations/create_visualizations.py` : Visualisations principales
- `visualizations/create_all_insights.py` : Visualisations d’insights
- `notebooks/01_data_storytelling.ipynb` : Notebook Jupyter avec analyses

### Visualisations HTML (GitHub Pages)

Les pages HTML sont générées dans `docs/visualizations/` :

- `00_dashboard_complet.html` : Dashboard complet avec explications
- `01_probleme_churn.html` : Analyse du problème + interprétation
- `02_solution_ml.html` : Performance modèle + variables clés
- `03_impact_business.html` : Impact business chiffré
- `insight_days_since_last_order.html` : Insight délai d’inactivité
- `insight_purchase_frequency.html` : Insight fréquence d’achat
- `segmentation_rfm.html` : Segmentation RFM
- `feature_importance.html` : Features importantes

Les visualisations brutes Plotly sont conservées dans `docs/visualizations/raw/`
et intégrées via iframe dans les pages ci-dessus.

### Dashboard Interactif

- `dashboard/app.py` : Dashboard Streamlit interactif

## 🚀 Utilisation

### Générer toutes les visualisations

```bash
# Créer un environnement si besoin
python3 -m venv .venv
source .venv/bin/activate

# Générer les visualisations
python3 visualizations/create_visualizations.py
python3 visualizations/create_all_insights.py
```

Les fichiers HTML seront créés dans `docs/visualizations/` et les exports Plotly
dans `docs/visualizations/raw/`.

### Ouvrir les visualisations

Ouvrez simplement les fichiers HTML dans votre navigateur :

```bash
# macOS
open docs/visualizations/00_dashboard_complet.html

# Linux
xdg-open docs/visualizations/00_dashboard_complet.html

# Windows
start docs/visualizations/00_dashboard_complet.html
```

### Lancer le dashboard Streamlit

```bash
# Installer Streamlit si nécessaire
pip install streamlit

# Lancer le dashboard
streamlit run dashboard/app.py
```

Le dashboard sera accessible sur `http://localhost:8501`

## 📊 Types de Visualisations

### 1. Le Problème
- Distribution du churn
- Valeur client par segment
- Évolution temporelle
- Segments à risque

### 2. La Solution
- Performance du modèle (F1, Precision, Recall, ROC-AUC)
- Matrice de confusion
- Distribution des scores
- Features importantes

### 3. Impact Business
- Réduction du churn (avant/après)
- Revenus sauvés
- ROI des campagnes
- Détection proactive

### 4. Dashboard Complet
- Vue d'ensemble
- Toutes les métriques en un coup d'œil
- Timeline du projet

## 🎨 Technologies Utilisées

- **Plotly** : Graphiques interactifs
- **Streamlit** : Dashboard web interactif
- **Pandas** : Manipulation de données
- **Matplotlib/Seaborn** : Graphiques statiques (optionnel)

## 💡 Intégration dans le Portfolio

Ces visualisations peuvent être :

1. **Intégrées dans GitHub Pages** : Ajouter des iframes ou screenshots
2. **Partagées sur LinkedIn** : Screenshots des graphiques
3. **Incluses dans une présentation** : Export PDF ou images
4. **Démontrées en entretien** : Ouvrir le dashboard en direct

---

**Visualisations créées pour démontrer les compétences en Data Science et storytelling.** 🚀
