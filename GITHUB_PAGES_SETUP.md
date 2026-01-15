# Guide de Configuration GitHub Pages

## 🚀 Activation Rapide

### Étape 0 : Créer le repository GitHub

Avant toute publication, créez le repository sur GitHub :

1. Aller sur https://github.com/new
2. **Repository name** : `churn-prediction` (ou le nom de votre choix)
3. **Visibility** : Public
4. Ne pas ajouter de README (le projet en a déjà un)
5. Cliquer sur **Create repository**

### Étape 1 : Personnaliser la page

Avant de publier, remplacez dans `docs/index.html` :

```html
<!-- Remplacer VOTRE_USERNAME par votre nom d'utilisateur GitHub -->
<a href="https://github.com/VOTRE_USERNAME/churn-prediction" ...>
```

**Rechercher et remplacer** :
- `VOTRE_USERNAME` → Votre nom d'utilisateur GitHub
- `VOTRE_PROFIL` → Votre profil LinkedIn (si applicable)

### Étape 2 : Publier sur GitHub

```bash
cd marketing/churn_prediction

# Initialiser git si nécessaire
git init

# Lier le repository distant
git remote add origin https://github.com/VOTRE_USERNAME/churn-prediction.git

# Ajouter tous les fichiers (y compris docs/)
git add .

# Commit
git commit -m "Add GitHub Pages"

# Push
git push origin main
```

### Étape 3 : Activer GitHub Pages

1. Aller sur votre repository GitHub
2. Cliquer sur **Settings**
3. Dans le menu de gauche, cliquer sur **Pages**
4. Sous **Source** :
   - Sélectionner **Deploy from a branch**
   - **Branch** : `main`
   - **Folder** : `/docs`
5. Cliquer sur **Save**

### Étape 4 : Accéder à votre page

Attendre 1-2 minutes, puis votre page sera accessible à :

```
https://VOTRE_USERNAME.github.io/churn-prediction/
```

## 📋 Checklist

- [ ] Repository GitHub créé
- [ ] Remplacé `VOTRE_USERNAME` dans `docs/index.html`
- [ ] Remplacé les liens LinkedIn/contact
- [ ] Fichiers commités et pushés sur GitHub
- [ ] GitHub Pages activé dans Settings
- [ ] Page accessible et fonctionnelle

## 🎨 Personnalisation

### Modifier les couleurs

Éditer `docs/assets/css/style.css` :

```css
:root {
    --primary-color: #2563eb;    /* Couleur principale */
    --secondary-color: #10b981;  /* Couleur secondaire */
    --danger-color: #ef4444;     /* Couleur d'alerte */
}
```

### Modifier le contenu

Éditer `docs/index.html` pour :
- Changer les statistiques
- Modifier les résultats
- Ajouter des sections
- Personnaliser les témoignages

## 🔍 Vérification

Une fois activé, vérifiez que :

1. ✅ La page se charge correctement
2. ✅ Les animations fonctionnent
3. ✅ Les liens GitHub sont corrects
4. ✅ Le design est responsive (test sur mobile)
5. ✅ Les sections s'affichent bien

## 🐛 Dépannage

### La page ne s'affiche pas

1. Vérifier que GitHub Pages est activé dans Settings
2. Vérifier que le dossier `/docs` contient `index.html`
3. Attendre quelques minutes (première publication peut prendre du temps)
4. Vérifier l'onglet "Actions" pour voir s'il y a des erreurs

### Les styles ne s'appliquent pas

1. Vérifier que les chemins dans `index.html` sont corrects :
   ```html
   <link rel="stylesheet" href="assets/css/style.css">
   ```
2. Vérifier que les fichiers CSS/JS sont bien dans `docs/assets/`

### Erreur 404

1. Vérifier l'URL : `https://USERNAME.github.io/REPO_NAME/`
2. Le nom du repository doit correspondre à l'URL

## 📱 Aperçu

La page GitHub Pages inclut :

- ✅ **Hero Section** : Titre accrocheur avec badges
- ✅ **Problème Client** : Statistiques et contexte
- ✅ **Solution** : 3 cartes avec les solutions
- ✅ **Impact Business** : Métriques chiffrées avec ROI
- ✅ **Approche Technique** : Stack et features
- ✅ **Résultats** : Métriques du modèle
- ✅ **Démonstration** : Exemples de code
- ✅ **Call-to-Action** : Liens vers GitHub et contact

## 🎯 Prochaines Étapes

1. **Partager le lien** sur LinkedIn/Twitter
2. **Ajouter au portfolio** principal
3. **Demander des feedbacks**
4. **Mettre à jour** régulièrement avec de nouveaux projets

---

**La page est prête ! Il suffit de l'activer dans les Settings GitHub.** 🚀
