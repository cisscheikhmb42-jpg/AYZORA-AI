import os
import csv
from pathlib import Path

import gradio as gr
import requests

from google import genai
from google.genai import types

from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook


# ==========================================
# CONFIGURATION AYZORA AI
# ==========================================

APP_TITLE = "AYZORA AI"

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)

SUPABASE_URL = (
    os.getenv("SUPABASE_URL") or ""
).rstrip("/")

SUPABASE_KEY = (
    os.getenv("SUPABASE_PUBLISHABLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
)

MAX_HISTORY_MESSAGES = 12
MAX_FILE_CHARS = 25000

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


# ==========================================
# CLIENT GEMINI
# ==========================================

def get_gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "La clé API Gemini est absente. "
            "Configure GEMINI_API_KEY dans Render."
        )

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ==========================================
# BASE DE CONNAISSANCES SUPABASE
# ==========================================

def load_knowledge():

    if not SUPABASE_URL or not SUPABASE_KEY:
        return [], (
            "Supabase n'est pas configuré. "
            "Les connaissances générales restent disponibles."
        )

    url = (
        f"{SUPABASE_URL}/rest/v1/"
        "ayzora_knowledge"
    )

    params = {
        "select": (
            "company,category,title,content,"
            "verified,updated_at"
        ),
        "order": "updated_at.desc",
        "limit": "100",
    }

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            return [], (
                "Le format des données Supabase "
                "est inattendu."
            )

        return data, ""

    except Exception as exc:
        return [], (
            "Base Supabase indisponible : "
            f"{type(exc).__name__}."
        )


# ==========================================
# LECTURE DES DOCUMENTS
# ==========================================

def extract_text_file(path):

    path = Path(path)
    extension = path.suffix.lower()

    if extension == ".pdf":

        reader = PdfReader(str(path))

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    elif extension == ".docx":

        document = Document(str(path))

        contents = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        for table in document.tables:
            for row in table.rows:
                contents.append(
                    " | ".join(
                        cell.text.strip()
                        for cell in row.cells
                    )
                )

        return "\n".join(contents)

    elif extension in {
        ".txt",
        ".md",
        ".json",
        ".log",
    }:

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    elif extension == ".csv":

        with open(
            path,
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as file:

            return "\n".join(
                " | ".join(row)
                for row in csv.reader(file)
            )

    elif extension in {".xlsx", ".xlsm"}:

        workbook = load_workbook(
            path,
            read_only=True,
            data_only=True,
        )

        contents = []

        try:
            for sheet in workbook.worksheets:

                contents.append(
                    f"Feuille : {sheet.title}"
                )

                for row in sheet.iter_rows(
                    values_only=True
                ):

                    values = [
                        str(value)
                        for value in row
                        if value is not None
                    ]

                    if values:
                        contents.append(
                            " | ".join(values)
                        )

        finally:
            workbook.close()

        return "\n".join(contents)

    return ""


# ==========================================
# TRAITEMENT DES FICHIERS ET DES IMAGES
# ==========================================

def prepare_attachments(files):

    parts = []
    notes = []

    for item in files or []:

        file_path = Path(str(item))

        if not file_path.exists():
            notes.append(
                f"Fichier introuvable : {file_path.name}"
            )
            continue

        extension = file_path.suffix.lower()

        try:

            # Images : envoi à Gemini
            if extension in IMAGE_TYPES:

                image_part = types.Part.from_bytes(
                    data=file_path.read_bytes(),
                    mime_type=IMAGE_TYPES[extension],
                )

                parts.append(image_part)

                notes.append(
                    f"Image jointe : {file_path.name}"
                )

            # Documents : extraction du texte
            elif extension in {
                ".pdf",
                ".docx",
                ".txt",
                ".md",
                ".json",
                ".log",
                ".csv",
                ".xlsx",
                ".xlsm",
            }:

                extracted_text = extract_text_file(
                    file_path
                )

                if not extracted_text.strip():
                    extracted_text = (
                        "Aucun texte extractible. "
                        "Le document peut être vide "
                        "ou scanné."
                    )

                parts.append(
                    types.Part(
                        text=(
                            f"CONTENU DU DOCUMENT "
                            f"{file_path.name} :\n"
                            f"{extracted_text[:MAX_FILE_CHARS]}"
                        )
                    )
                )

                notes.append(
                    f"Document joint : {file_path.name}"
                )

            else:

                notes.append(
                    f"Format non pris en charge : "
                    f"{file_path.name}"
                )

        except Exception as exc:

            notes.append(
                f"Erreur de lecture de "
                f"{file_path.name} : "
                f"{type(exc).__name__}"
            )

    return parts, notes


# ==========================================
# HISTORIQUE DES CONVERSATIONS
# ==========================================

def format_history(history):

    lines = []

    for item in (
        history or []
    )[-MAX_HISTORY_MESSAGES:]:

        if not isinstance(item, dict):
            continue

        role = item.get("role", "")
        content = item.get("content", "")

        if (
            role in {"user", "assistant"}
            and isinstance(content, str)
            and content.strip()
        ):

            label = (
                "Utilisateur"
                if role == "user"
                else "AYZORA AI"
            )

            lines.append(
                f"{label} : {content}"
            )

    return (
        "\n".join(lines)
        if lines
        else "Début de conversation."
    )


# ==========================================
# INSTRUCTIONS DE L'ASSISTANT
# ==========================================

def build_prompt(
    message,
    history,
    knowledge,
):

    instructions = """
Tu es AYZORA AI, un assistant intelligent,
professionnel et utile.

Réponds principalement en français,
avec des explications claires et naturelles.

Adapte tes réponses à la demande de l'utilisateur.

Pour Arrow Home Group et Forza Luxury :
- Distingue clairement les deux entreprises.
- Utilise les informations disponibles dans
  la base de connaissances lorsqu'elles sont pertinentes.
- N'invente pas de prix, de coordonnées,
  de services ou de réalisations.
- Si une information manque, dis-le clairement.

Pour les autres demandes, réponds normalement.

Ne prétends pas avoir vérifié une information
récente si tu ne l'as pas réellement vérifiée.

Ne révèle jamais les clés API ni les secrets
des variables d'environnement.
"""

    knowledge_text = (
        str(knowledge)
        if knowledge
        else "Aucune connaissance Supabase disponible."
    )

    return f"""
{instructions}

BASE DE CONNAISSANCES AYZORA :

{knowledge_text}

HISTORIQUE DE LA CONVERSATION :

{format_history(history)}

MESSAGE ACTUEL :

{message}

Réponds directement au message actuel.
Tiens compte de l'historique et des documents joints.
"""


# ==========================================
# FONCTION PRINCIPALE DU CHATBOT
# ==========================================

def respond(message, history, files):

    history = list(history or [])

    if not message or not message.strip():

        return (
            history,
            "",
            "Écris un message pour commencer.",
        )

    message = message.strip()

    try:

        # Charger les connaissances
        knowledge, knowledge_warning = (
            load_knowledge()
        )

        # Préparer les documents et images
        attachment_parts, attachment_notes = (
            prepare_attachments(files)
        )

        # Préparer les instructions
        prompt = build_prompt(
            message,
            history,
            knowledge,
        )

        parts = [
            types.Part(text=prompt)
        ] + attachment_parts

        # Interroger Gemini
        client = get_gemini_client()

        result = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=parts,
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.6,
                max_output_tokens=2048,
            ),
        )

        answer = (
            getattr(result, "text", None) or ""
        ).strip()

        if not answer:
            answer = (
                "Je n'ai pas reçu de réponse exploitable. "
                "Réessaie avec une autre formulation."
            )

        # Ajouter les informations sur les fichiers
        if attachment_notes:

            answer += (
                "\n\n---\n"
                + "\n".join(attachment_notes)
            )

        if knowledge_warning:

            answer += (
                "\n\nNote : "
                + knowledge_warning
            )

        # Conserver les messages dans l'historique
        history.extend([
            {
                "role": "user",
                "content": message,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ])

        return (
            history,
            "",
            f"Réponse générée avec {MODEL_NAME}.",
        )

    except Exception as exc:

        if not GEMINI_API_KEY:

            status = (
                "La clé API Gemini est absente. "
                "Ajoute GEMINI_API_KEY dans "
                "les variables d'environnement de Render."
            )

        else:

            status = (
                f"Erreur : {type(exc).__name__}. "
                "Consulte les journaux de Render "
                "pour identifier la cause."
            )

        return (
            history,
            message,
            status,
        )


# ==========================================
# EFFACER LA CONVERSATION
# ==========================================

def clear_chat():

    return (
        [],
        "",
        "Conversation effacée.",
    )


# ==========================================
# INTERFACE GRADIO
# ==========================================

with gr.Blocks(
    title=APP_TITLE
) as demo:

    gr.Markdown(
        """
        # 🟠 AYZORA AI

        Votre assistant intelligent pour discuter,
        analyser des documents et comprendre des images.
        """
    )

    chat = gr.Chatbot(
        label="Conversation",
        type="messages",
        height=520,
    )

    with gr.Row():

        message_box = gr.Textbox(
            label="Votre message",
            placeholder="Écrivez votre message à AYZORA AI...",
            lines=2,
            scale=6,
        )

        send_button = gr.Button(
            "Envoyer",
            variant="primary",
            scale=1,
        )

        clear_button = gr.Button(
            "Nouvelle conversation",
            scale=1,
        )

    attachments = gr.File(
        label="Joindre des documents ou des images",
        file_count="multiple",
        type="filepath",
        file_types=[
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".json",
            ".log",
            ".csv",
            ".xlsx",
            ".xlsm",
        ],
    )

    status_box = gr.Markdown("Prêt.")

    send_button.click(
        fn=respond,
        inputs=[
            message_box,
            chat,
            attachments,
        ],
        outputs=[
            chat,
            message_box,
            status_box,
        ],
    )

    message_box.submit(
        fn=respond,
        inputs=[
            message_box,
            chat,
            attachments,
        ],
        outputs=[
            chat,
            message_box,
            status_box,
        ],
    )

    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[
            chat,
            message_box,
            status_box,
        ],
    )


# ==========================================
# DÉMARRAGE DE L'APPLICATION SUR RENDER
# ==========================================

if __name__ == "__main__":

    port = int(
        os.getenv("PORT", "10000")
    )

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
    )
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
