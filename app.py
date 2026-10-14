import os
import json
import csv
import tempfile
from pathlib import Path

import gradio as gr
import requests
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook

APP_TITLE = "AYZORA AI"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Clé Gemini absente. Configure GEMINI_API_KEY dans les variables d'environnement."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

INSTRUCTIONS = """
Tu es AYZORA AI, un assistant intelligent et professionnel.
Tu aides l'utilisateur dans ses demandes générales et professionnelles.
Réponds principalement en français, de façon claire, naturelle et utile.
Lorsque la question concerne Arrow Home Group ou Forza Luxury :
- distingue toujours les deux entreprises ;
- utilise uniquement les informations présentes dans la base de connaissances fournie ;
- n'invente jamais de services, prix, coordonnées, réalisations ou liens ;
- si une information manque ou n'est pas confirmée, dis-le clairement.
Pour toute autre demande, réponds normalement. Ne prétends pas avoir vérifié des informations récentes si tu ne l'as pas fait.
"""

def load_knowledge():
    """Récupère les connaissances depuis Supabase. Retourne un message explicite si indisponible."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return [], "Supabase n'est pas configuré : la réponse utilisera les connaissances générales seulement."
    url = f"{SUPABASE_URL}/rest/v1/ayzora_knowledge"
    params = {
        "select": "company,category,title,content,verified,updated_at",
        "order": "updated_at.desc",
    }
    headers = {"apikey": SUPABASE_KEY}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data, ""
    except Exception as exc:
        return [], f"Base de connaissances Supabase indisponible ({type(exc).__name__})."

def extract_text_file(path):
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        doc = Document(str(path))
        chunks = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                chunks.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(chunks)

    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    if ext == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            return "\n".join(" | ".join(row) for row in csv.reader(f))

    if ext == ".xlsx":
        wb = load_workbook(path, read_only=True, data_only=True)
        chunks = []
        for ws in wb.worksheets:
            chunks.append(f"Feuille : {ws.title}")
            for row in ws.iter_rows(values_only=True):
                values = ["" if value is None else str(value) for value in row]
                if any(values):
                    chunks.append(" | ".join(values))
        wb.close()
        return "\n".join(chunks)

    return ""

def prepare_attachments(files):
    """Prépare les pièces jointes : images envoyées à Gemini, documents convertis en texte."""
    parts = []
    notes = []
    max_chars_per_file = 25000

    for item in files or []:
        path = getattr(item, "name", None) or str(item)
        file_path = Path(path)
        if not file_path.exists():
            notes.append(f"Fichier ignoré (introuvable) : {file_path.name}")
            continue

        ext = file_path.suffix.lower()
        try:
            if ext in {".png", ".jpg", ".jpeg", ".webp"}:
                mime = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                }[ext]
                data = file_path.read_bytes()
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                notes.append(f"Image jointe : {file_path.name}")
            elif ext in {".pdf", ".docx", ".txt", ".csv", ".xlsx"}:
                extracted = extract_text_file(file_path)
                if not extracted.strip():
                    extracted = "(Aucun texte extractible. Le document peut être scanné ou vide.)"
                parts.append(types.Part(text=f"CONTENU DU FICHIER {file_path.name}:\n{extracted[:max_chars_per_file]}"))
                notes.append(f"Document joint : {file_path.name}")
            else:
                notes.append(f"Format non pris en charge : {file_path.name}")
        except Exception as exc:
            notes.append(f"Erreur de lecture pour {file_path.name} ({type(exc).__name__}).")

    return parts, notes

def format_history(history):
    """Convertit l'historique interne au format attendu par le prompt."""
    lines = []
    for item in (history or [])[-12:]:
        role = item.get("role", "")
        text = item.get("text", "")
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines) or "Début de conversation."

async def respond(message, files, history):
    history = list(history or [])
    if not message or not message.strip():
        return history, history, "", None

    message = message.strip()
    knowledge, knowledge_warning = load_knowledge()
    knowledge_json = json.dumps(knowledge, ensure_ascii=False, indent=2)
    attachment_parts, attachment_notes = prepare_attachments(files)

    prompt = f"""
{INSTRUCTIONS}

BASE DE CONNAISSANCES AYZORA (données récupérées de Supabase) :
{knowledge_json if knowledge else "Aucune donnée de connaissance n'a pu être récupérée."}

HISTORIQUE RÉCENT :
{format_history(history)}

MESSAGE ACTUEL :
{message}

RÈGLES :
- Réponds à la demande actuelle en tenant compte de l'historique.
- Si des fichiers sont joints, analyse-les et base-toi sur leur contenu.
- Ne prétends pas qu'un fichier ou une image contient des éléments que tu n'as pas pu lire.
- Ne révèle jamais de clés, variables d'environnement ou détails secrets.
"""
    parts = [types.Part(text=prompt)] + attachment_parts

    try:
        result = await __import__("asyncio").to_thread(
            client.models.generate_content,
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=parts)],
        )
        answer = (result.text or "").strip() or "Je n'ai pas reçu de réponse exploitable. Réessaie."
        if attachment_notes:
            answer += "\n\n---\n" + "\n".join(attachment_notes)
        if knowledge_warning:
            answer += "\n\nNote technique : " + knowledge_warning

        history.extend([
            {"role": "user", "text": message},
            {"role": "assistant", "text": answer},
        ])
        display = [{"role": item["role"], "content": item["text"]} for item in history]
        return display, history, "", None
    except Exception as exc:
        safe_error = f"AYZORA n'a pas pu traiter la demande ({type(exc).__name__}). Vérifie la configuration et les journaux du service."
        history.append({"role": "user", "text": message})
        history.append({"role": "assistant", "text": safe_error})
        display = [{"role": item["role"], "content": item["text"]} for item in history]
        return display, history, "", None

def clear_chat():
    return [], [], None, ""

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🟠 AYZORA AI\n"
        "Votre assistant intelligent — conversation, analyse de documents et d'images."
    )
    history_state = gr.State([])
    chat = gr.Chatbot(label="Conversation", height=520)
    attachments = gr.File(
        label="Joindre des images ou documents",
        file_count="multiple",
        type="filepath",
        file_types=[".png", ".jpg", ".jpeg", ".webp", ".pdf", ".docx", ".txt", ".csv", ".xlsx"],
    )
    with gr.Row():
        textbox = gr.Textbox(
            label="Message",
            placeholder="Écrivez votre message à AYZORA...",
            lines=2,
            scale=6,
        )
        send = gr.Button("Envoyer", variant="primary", scale=1)
        clear = gr.Button("Nouvelle conversation", scale=1)

    send.click(respond, inputs=[textbox, attachments, history_state],
               outputs=[chat, history_state, textbox, attachments])
    textbox.submit(respond, inputs=[textbox, attachments, history_state],
                   outputs=[chat, history_state, textbox, attachments])
    clear.click(clear_chat, outputs=[chat, history_state, attachments, textbox])

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "10000")),
    )
