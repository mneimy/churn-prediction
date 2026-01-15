#!/bin/bash
# Script de configuration de l'environnement pour macOS

echo "Configuration de l'environnement pour Churn Prediction..."

# Vérifier si libomp est installé
if ! brew list libomp &>/dev/null; then
    echo "Installation de libomp (requis pour XGBoost)..."
    brew install libomp
fi

# Exporter les variables d'environnement pour XGBoost
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

echo "✓ Variables d'environnement configurées"
echo ""
echo "Pour activer ces variables dans votre shell:"
echo "  source SETUP_ENV.sh"
echo ""
echo "Ou ajoutez ces lignes à votre ~/.zshrc ou ~/.bash_profile:"
echo "  export LDFLAGS=\"-L/opt/homebrew/opt/libomp/lib\""
echo "  export CPPFLAGS=\"-I/opt/homebrew/opt/libomp/include\""
