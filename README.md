# Generateur d'image a partir d'une description audio

Ce projet transforme une description vocale en image

## Fonctionnement

1. L'utilisateur enregistre un audio dans l'interface Streamlit.
2. L'API transcrit l'audio en texte.
3. Le texte est converti en prompt anglais.
4. Une image est generé et renvoyee a l'interface.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer le projet

Terminal 1 (API) :

```bash
uvicorn api:app --reload --port 8000
```

Terminal 2 (Interface) :

```bash
streamlit run app.py
```

## Endpoint API

- URL : `POST http://localhost:8000/generate_image`
- Type de requete : `multipart/form-data`
- Champ attendu : `description` (fichier audio WAV)
- Reponse en cas de succes : image PNG (`image/png`)

## Utilisation

- Ouvrir l'interface Streamlit dans le navigateur
- Enregistrer une description audio
- Cliquer sur **Générer**
- L'image generée s'affiche dans la page

## Structure minimale

- `app.py` : interface utilisateur Streamlit
- `api.py` : API FastAPI et pipeline IA
- `requirements.txt` : dependances Python
