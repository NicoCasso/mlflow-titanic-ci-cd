FROM python:3.11-slim

# Installer les dépendances système nécessaires, y compris libgomp1
# Installer uniquement ce qui est nécessaire, proprement
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . /app
WORKDIR /app

# Ajouter /app au PYTHONPATH
ENV PYTHONPATH="/app"

# Lancer le script predict.py automatiquement
CMD ["python", "src/predict.py", "&&", "sleep", "10"]