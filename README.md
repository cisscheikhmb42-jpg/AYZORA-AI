# AYZORA AI

Application web Python/Gradio basée sur Gemini, avec récupération facultative des connaissances depuis Supabase et analyse de fichiers.

## Variables d'environnement à configurer sur Render

- `GEMINI_API_KEY` : clé Gemini privée.
- `SUPABASE_URL` : URL du projet Supabase.
- `SUPABASE_PUBLISHABLE_KEY` : clé publishable/anon de Supabase (jamais une service-role key dans cette application client).
- `GEMINI_MODEL` (facultatif) : par défaut `gemini-3.1-flash-lite-preview`.
- `PORT` : Render la définit normalement automatiquement.

Ne mettez jamais de clés dans le code ni dans le dépôt GitHub.

## Déploiement Render

- Type : Web Service
- Runtime : Python
- Build command : `pip install -r requirements.txt`
- Start command : `python app.py`

Le service gratuit peut s'endormir après une période d'inactivité et son redémarrage peut prendre du temps. Vérifiez toujours le tarif sélectionné avant de confirmer.

## Limites de cette première version

- L'interface prend en charge le texte, les images PNG/JPG/WEBP, PDF, DOCX, TXT, CSV et XLSX.
- La voix et les workflows Google ADK présents dans le notebook Colab ne sont pas encore intégrés à cette version de déploiement. Ils devront être ajoutés et testés séparément.
- Les connaissances d'entreprise viennent de `public.ayzora_knowledge` dans Supabase. Si Supabase n'est pas configuré ou indisponible, l'application le signale et continue sans ces données.
