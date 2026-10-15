import os
import csv
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook

APP_TITLE = "AYZORA AI"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = (
    os.getenv("SUPABASE_PUBLISHABLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
)
MAX_HISTORY_MESSAGES = 12
MAX_CHARS_PER_FILE = 25000
SUPPORTED_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def get_gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Clé Gemini absente. Configure GEMINI_API_KEY dans les variables d'environnement de Render."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def load_knowledge():
    """Récupère les connaissances Supabase si les variables sont configurées."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return [], "Supabase n'est pas configuré ; les réponses utiliseront les connaissances générales."

    url = f"{SUPABASE_URL}/rest/v1/ayzora_knowledge"
    params = {
        "select": "company,category,title,content,verified,updated_at",
        "order": "updated_at.desc",
        "limit": "100",
    }
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            return [], "Supabase a renvoyé un format inattendu."
        return data, ""
    except Exception as exc:
        return [], f"Base de connaissances Supabase indisponible ({type(exc).__name__})."


def extract_text_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        doc = Document(str(path))
        chunks = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                chunks.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(chunks)
    if ext in {".txt", ".md", ".json", ".log"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
            return "\n".join(" | ".join(row) for row in csv.reader(file))
    if ext in {".xlsx", ".xlsm"}:
        workbook = load_workbook(path, read_only=True, data_only=True)
        chunks = []
        try:
            for sheet in workbook.worksheets:
                chunks.append(f"Feuille : {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    values = [str(value) for value in row if value is not None]
                    if values:
                        chunks.append(" | ".join(values))
        finally:
            workbook.close()
        return "\n".join(chunks)
    return ""


def prepare_attachments(files):
    """Retourne les parties Gemini et les messages d'état liés aux pièces jointes."""
    parts = []
    notes = []
    for file_item in files or []:
        raw_path = getattr(file_item, "name", None) or str(file_item)
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            notes.append(f"Fichier ignoré (introuvable) : {path.name}")
            continue

        ext = path.suffix.lower()
        try:
            if ext in SUPPORTED_IMAGE_TYPES:
                parts.append(
                    types.Part.from_bytes(
                        data=path.read_bytes(),
                        mime_type=SUPPORTED_IMAGE_TYPES[ext],
                    )
                )
                notes.append(f"Image jointe : {path.name}")
            elif ext in {
                ".pdf", ".docx", ".txt", ".md", ".json", ".log",
                ".csv", ".xlsx", ".xlsm",
            }:
                extracted = extract_text_file(path)
                if not extracted.strip():
                    extracted = "(Aucun texte extractible : document vide, scanné ou non pris en charge.)"
                parts.append(
                    types.Part(
                        text=f"CONTENU DU FICHIER {path.name} :\n{extracted[:MAX_CHARS_PER_FILE]}"
                    )
                )
                notes.append(f"Document joint : {path.name}")
            else:
                notes.append(f"Format non pris en charge : {path.name}")
        except Exception as exc:
            notes.append(f"Erreur de lecture pour {path.name} ({type(exc).__name__}).")
    return parts, notes


def format_history(history):
    lines = []
    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "")
        content = item.get("content", "")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            label = "Utilisateur" if role == "user" else "AYZORA AI"
            lines.append(f"{label} : {content}")
    return "\n".join(lines) if lines else "Début de conversation."


def build_prompt(message, history, knowledge, knowledge_warning):
    instructions = """
Tu es AYZORA AI, un assistant intelligent et professionnel.
Réponds principalement en français, de façon claire, naturelle et utile.
Pour Arrow Home Group et Forza Luxury, distingue les deux entreprises et utilise uniquement
les informations pertinentes fournies par la base de connaissances. N'invente pas de services,
prix, coordonnées, réalisations ou liens. Si une information manque, dis-le clairement.
Pour les autres demandes, réponds normalement.
Ne prétends pas avoir vérifié des informations récentes si tu ne l'as pas fait.
Ne révèle jamais les clés API, variables d'environnement ou autres informations secrètes.
"""
    knowledge_text = (
        str(knowledge)
        if knowledge
        else "Aucune donnée de connaissance Supabase disponible."
    )
    prompt = (
        f"{instructions}\n"
        f"BASE DE CONNAISSANCES AYZORA (Supabase) :\n{knowledge_text}\n\n"
        f"HISTORIQUE RÉCENT :\n{format_history(history)}\n\n"
        f"MESSAGE ACTUEL :\n{message}\n\n"
        "Réponds au message actuel en tenant compte du contexte. "
        "Si les informations de la base ne suffisent pas, précise-le."
    )
    if knowledge_warning:
        prompt += f"\n\nNote interne sur la base : {knowledge_warning}"
    return prompt


def respond(message, history, files):
    history = list(history or [])
    if not message or not message.strip():
        return history, "", "Écris un message pour commencer."

    message = message.strip()
    try:
        knowledge, knowledge_warning = load_knowledge()
        attachment_parts, attachment_notes = prepare_attachments(files)
        prompt = build_prompt(message, history, knowledge, knowledge_warning)
        parts = [types.Part(text=prompt)] + attachment_parts

        client = get_gemini_client()
        result = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.6,
                max_output_tokens=2048,
            ),
        )
        answer = (getattr(result, "text", None) or "").strip()
        if not answer:
            answer = "Je n'ai pas reçu de réponse exploitable. Réessaie avec une autre formulation."

        if attachment_notes:
            answer += "\n\n---\n" + "\n".join(attachment_notes)
        if knowledge_warning:
            answer += "\n\nNote : " + knowledge_warning

        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ])
        return history, "", f"Réponse générée avec {MODEL_NAME}."
    except Exception as exc:
        error_text = str(exc)
        if not GEMINI_API_KEY:
            friendly = (
                "La clé API Gemini est absente. Dans Render → Environment, ajoute GEMINI_API_KEY, "
                "puis redéploie l'application."
            )
        elif "API key" in error_text or "PERMISSION_DENIED" in error_text:
            friendly = (
                "Gemini a refusé la requête. Vérifie la clé API, les autorisations et le modèle "
                "configuré dans Render."
            )
        else:
            friendly = f"Erreur : {type(exc).__name__}. Consulte les journaux Render pour le détail."
        return history, message, friendly


def clear_chat():
    return [], "", "Conversation effacée."


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(
        "# 🟠 AYZORA AI\n"
        "Votre assistant intelligent — conversation, analyse de documents et d'images."
    )
    chat = gr.Chatbot(label="Conversation", type="messages", height=520)
    with gr.Row():
        message_box = gr.Textbox(
            label="Message",
            placeholder="Écrivez votre message à AYZORA AI…",
            lines=2,
            scale=6,
        )
        send_button = gr.Button("Envoyer", variant="primary", scale=1)
        clear_button = gr.Button("Nouvelle conversation", scale=1)
    attachments = gr.File(
        label="Joindre des images ou documents",
        file_count="multiple",
        type="filepath",
        file_types=[
            ".png", ".jpg", ".jpeg", ".webp", ".pdf", ".docx",
            ".txt", ".md", ".json", ".log", ".csv", ".xlsx", ".xlsm",
        ],
    )
    status_box = gr.Markdown("Prêt.")

    send_button.click(
        fn=respond,
        inputs=[message_box, chat, attachments],
        outputs=[chat, message_box, status_box],
    )
    message_box.submit(
        fn=respond,
        inputs=[message_box, chat, attachments],
        outputs=[chat, message_box, status_box],
    )
    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chat, message_box, status_box],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
