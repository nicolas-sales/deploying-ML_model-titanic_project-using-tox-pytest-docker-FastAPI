FROM python:3.11-slim

# Définir un répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements dans app
COPY requirements.txt /app

# Installer les dépendances
RUN pip install -r requirements.txt

# Copier le contenu du répertoire courant dans le conteneur
COPY . /app

# Commande pour exécuter l'application FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]