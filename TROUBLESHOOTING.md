# Guide de Dépannage

## Problèmes d'Installation

### Erreur "metadata-generation-failed"

**Cause :** Incompatibilité entre les versions de packages et Python 3.13+

**Solutions :**

1. **Mettre à jour pip :**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Installer dans un environnement virtuel (recommandé) :**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Si xgboost pose problème :**
   ```bash
   # Installer xgboost séparément avec les binaires précompilés
   pip install xgboost --only-binary :all:
   ```

4. **Installer les packages un par un pour identifier le problème :**
   ```bash
   pip install pandas numpy scikit-learn
   pip install xgboost
   pip install fastapi uvicorn pydantic
   # etc.
   ```

### Erreur avec XGBoost

**Cause :** XGBoost nécessite parfois une compilation

**Solutions :**

1. **Utiliser les binaires précompilés :**
   ```bash
   pip install xgboost --only-binary :all:
   ```

2. **Installer via conda (alternative) :**
   ```bash
   conda install -c conda-forge xgboost
   ```

3. **Version alternative (si problème persiste) :**
   ```bash
   pip install xgboost==2.0.3 --no-build-isolation
   ```

### Erreur "externally-managed-environment"

**Cause :** Python système protégé (macOS avec Homebrew)

**Solution :** Toujours utiliser un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Problèmes de compatibilité Python 3.13

Si vous utilisez Python 3.13 et rencontrez des problèmes :

1. **Utiliser Python 3.11 ou 3.12 (recommandé pour stabilité) :**
   ```bash
   # Avec pyenv
   pyenv install 3.12.0
   pyenv local 3.12.0
   python -m venv venv
   source venv/bin/activate
   ```

2. **Ou utiliser des versions plus flexibles :**
   Le fichier `requirements.txt` utilise maintenant `>=` au lieu de `==` pour plus de flexibilité.

## Problèmes d'Exécution

### Erreur "ModuleNotFoundError"

**Solution :** Vérifier que vous êtes dans l'environnement virtuel et que les dépendances sont installées

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Erreur lors du chargement du modèle

**Cause :** Le modèle n'a pas été entraîné

**Solution :**
```bash
python scripts/train.py
```

### Erreur API "Modèle non chargé"

**Solution :** Entraîner le modèle d'abord, puis lancer l'API

```bash
# 1. Entraîner
python scripts/train.py

# 2. Lancer l'API
python scripts/run_api.py
```

## Vérification de l'Installation

Pour vérifier que tout est installé correctement :

```bash
python -c "import pandas, numpy, sklearn, xgboost, fastapi; print('✓ Tous les packages importés avec succès')"
```

## Support

Si les problèmes persistent :

1. Vérifier la version de Python : `python --version`
2. Vérifier que pip est à jour : `pip --version`
3. Créer un nouvel environnement virtuel propre
4. Installer les packages un par un pour identifier le problème
