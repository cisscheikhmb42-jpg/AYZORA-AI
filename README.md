# AYZORA AI — version complète corrigée

Cette version contient :
- Chatbot Gemini avec affichage des messages Gradio 5.
- Historique de conversation conservé pendant la session.
- Récupération optionnelle des connaissances depuis Supabase REST.
- Analyse de fichiers PDF, DOCX, TXT, MD, JSON, LOG, CSV, XLSX et XLSM.
- Analyse d'images PNG, JPG, JPEG et WEBP via Gemini.

## Variables d'environnement Render

Obligatoire :
- `GEMINI_API_KEY` : clé API Google AI Studio.

Facultatives :
- `GEMINI_MODEL` : modèle Gemini ; valeur par défaut `gemini-2.5-flash`.
- `SUPABASE_URL` : URL du projet Supabase.
- `SUPABASE_PUBLISHABLE_KEY` ou `SUPABASE_ANON_KEY` : clé publique/anon Supabase.

La table Supabase attendue est `ayzora_knowledge`, avec les colonnes `company`, `category`,
`title`, `content`, `verified` et `updated_at`. Si ton schéma est différent, adapte la requête
`load_knowledge()` avant de déployer. Utilise uniquement une clé publique/anon côté application,
jamais la clé `service_role`.

## Commandes Render
Build command : `pip install -r requirements.txt`
Start command : `python app.py`

Ne place jamais les clés API dans le code ou dans GitHub.

## Déploiement
1. Remplace entièrement `app.py` par le fichier fourni.
2. Remplace entièrement `requirements.txt` par le fichier fourni.
3. Fais un commit GitHub.
4. Dans Render, lance `Manual Deploy` → `Deploy latest commit`.
5. Consulte les Build Logs et Runtime Logs.

Remarque : l'historique est conservé dans la session Gradio, pas de façon permanente dans Supabase.
