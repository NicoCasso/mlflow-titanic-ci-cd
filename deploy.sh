#!/bin/bash

set -e  # Arrêter le script si une commande échoue

echo "Début du déploiement..."

# Exemple : copier les fichiers vers un serveur distant
# Remplace les variables ci-dessous avec tes vraies valeurs
REMOTE_USER="ubuntu"
REMOTE_HOST="your.server.com"
REMOTE_DIR="/var/www/app"

echo "Copie des fichiers vers $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR ..."
scp -r * "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

echo "Connexion au serveur distant pour redémarrer l'application..."
ssh "$REMOTE_USER@$REMOTE_HOST" << EOF
  cd $REMOTE_DIR
  echo "Pulling latest changes (si besoin)..."
  git pull origin main
  echo "Redémarrage du service (ex : systemd ou docker)..."
  sudo systemctl restart my-app.service
EOF

echo "Déploiement terminé avec succès ✅"