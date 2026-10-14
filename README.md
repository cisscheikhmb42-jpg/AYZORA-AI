# AYZORA AI

Assistant basé sur Gemini, avec récupération optionnelle des connaissances depuis Supabase et lecture de documents/images.

## Variables d'environnement Render
- `GEMINI_API_KEY` (obligatoire)
- `GEMINI_MODEL` (facultatif ; valeur par défaut `gemini-3.1-flash-lite-preview`)
- `SUPABASE_URL` (facultatif)
- `SUPABASE_PUBLISHABLE_KEY` ou `SUPABASE_ANON_KEY` (facultatif)

## Commandes Render
Build: `pip install -r requirements.txt`
Start: `python app.py`

L'historique est conservé dans l'état de la session Gradio ; il n'est pas encore stocké durablement dans Supabase.
