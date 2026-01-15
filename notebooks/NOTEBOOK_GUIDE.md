# 📓 Guide d'Utilisation du Notebook

## ✅ Problème Résolu

L'erreur `ValueError: Mime type rendering requires nbformat>=4.2.0` a été corrigée :

1. ✅ **nbformat installé** (version 5.10.4)
2. ✅ **ipywidgets installé** (pour widgets interactifs)
3. ✅ **Fonction helper `show_plot()`** créée pour affichage robuste
4. ✅ **Tous les `fig.show()` remplacés** par `show_plot(fig, html_path)`

## 🚀 Utilisation

### Lancer le notebook

```bash
# Activer l'environnement
source venv/bin/activate

# Lancer Jupyter
jupyter notebook notebooks/01_data_storytelling.ipynb
```

Ou avec JupyterLab :

```bash
jupyter lab notebooks/01_data_storytelling.ipynb
```

### Exécuter toutes les cellules

Dans Jupyter :
- **Menu** : `Cell → Run All`
- **Raccourci** : `Shift + Enter` (cellule par cellule)

## 📊 Visualisations Générées

Le notebook génère automatiquement **8 visualisations HTML** :

1. `insight_days_since_last_order.html` - Insight clé sur le délai
2. `insight_purchase_frequency.html` - Impact de la fréquence
3. `impact_financier.html` - Analyse financière
4. `segmentation_rfm.html` - Segmentation clients
5. `model_performance.html` - Performance du modèle
6. `feature_importance.html` - Features importantes
7. `impact_business_complet.html` - Impact business
8. `dashboard_storytelling.html` - Dashboard complet

## 🔧 Fonction Helper

La fonction `show_plot()` :
- ✅ Essaie d'afficher dans Jupyter
- ✅ Si échec, sauvegarde en HTML et affiche un message
- ✅ Toujours sauvegarde le HTML (même si affichage réussi)

## 💡 Si Problème Persiste

Si vous avez encore des erreurs :

1. **Vérifier l'installation** :
   ```bash
   pip list | grep -E "nbformat|ipywidgets|plotly"
   ```

2. **Réinstaller** :
   ```bash
   pip install --upgrade nbformat ipywidgets plotly
   ```

3. **Alternative** : Les graphiques sont toujours sauvegardés en HTML
   - Ouvrez les fichiers HTML dans votre navigateur
   - Tous les graphiques sont interactifs même en HTML

## 📁 Fichiers Créés

Toutes les visualisations sont sauvegardées dans :
```
visualizations/
├── insight_days_since_last_order.html
├── insight_purchase_frequency.html
├── impact_financier.html
├── segmentation_rfm.html
├── model_performance.html
├── feature_importance.html
├── impact_business_complet.html
└── dashboard_storytelling.html
```

## 🎯 Prochaines Étapes

1. **Exécuter le notebook** : `Cell → Run All`
2. **Vérifier les visualisations** : Ouvrir les fichiers HTML
3. **Personnaliser** : Modifier les analyses selon vos besoins
4. **Partager** : Intégrer dans GitHub Pages ou présentation

---

**Le notebook est maintenant fonctionnel et prêt à l'emploi !** 🚀
