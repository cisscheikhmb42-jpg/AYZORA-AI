# AYZORA AI — version corrigée

## Fichiers
- `app.py` : chatbot Gemini et lecture de documents.
- `requirements.txt` : dépendances compatibles avec Gradio 5 et `type="messages"`.

## Configuration Render
Dans Render → service AYZORA AI → Environment, configure :
- `GEMINI_API_KEY` : ta clé API Google AI Studio.
- `GEMINI_MODEL` (facultatif) : modèle Gemini à utiliser. Par défaut : `gemini-2.5-flash`.

Ne publie jamais ta clé API dans GitHub et ne l'écris pas directement dans `app.py`.

## Déploiement
1. Remplace le contenu de `app.py` par le fichier fourni.
2. Remplace le contenu de `requirements.txt` par le fichier fourni.
3. Enregistre les deux fichiers sur GitHub avec **Commit changes**.
4. Dans Render, lance **Manual Deploy → Deploy latest commit** si nécessaire.
5. Vérifie les logs de build et de démarrage.

## Attention
Cette version gère les documents joints à une conversation, mais ne réintègre pas automatiquement une éventuelle base de connaissances Supabase de l'ancienne application. Si AYZORA AI dépend de Supabase pour ses connaissances, cette intégration devra être rétablie selon le schéma exact de la base avant de considérer la migration comme complète.
