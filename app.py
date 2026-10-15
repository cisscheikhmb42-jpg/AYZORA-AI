import os
import json
import csv
import io
import base64
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

APP_NAME = "AYZORA AI"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SUPPORTED_IMAGES = {
    ".png", ".jpg", ".jpeg", ".webp"
}

SUPPORTED_DOCUMENTS = {
    ".pdf", ".docx", ".txt", ".md",
    ".json", ".log", ".csv", ".xlsx", ".xlsm"
}


# ==========================================
# CONNEXION GEMINI
# ==========================================

def get_gemini_client():
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "La clé API Gemini est absente. "
            "Configure GEMINI_API_KEY dans Render."
        )

    return genai.Client(api_key=api_key)


# ==========================================
# CONNAISSANCES SUPABASE (OPTIONNEL)
# ==========================================

def load_knowledge(question):
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = (
        os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )

    if not supabase_url or not supabase_key:
        return ""

    try:
        url = (
            f"{supabase_url.rstrip('/')}"
            "/rest/v1/ayzora_knowledge"
        )

        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        }

        params = {
            "select": "company,category,title,content",
            "verified": "eq.true",
            "limit": "10",
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15,
        )

        response.raise_for_status()
        rows = response.json()

        if not rows:
            return ""

        knowledge = []

        for row in rows:
            title = row.get("title", "")
            content = row.get("content", "")
            category = row.get("category", "")
            company = row.get("company", "")

            if content:
                knowledge.append(
                    f"Entreprise : {company}\n"
                    f"Catégorie : {category}\n"
                    f"Titre : {title}\n"
                    f"Contenu : {content}"
                )

        return "\n\n".join(knowledge)

    except Exception as error:
        print(f"Erreur Supabase : {error}")
        return ""


# ==========================================
# EXTRACTION DES DOCUMENTS
# ==========================================

def extract_text_file(file_path):
    path = Path(file_path)
    extension = path.suffix.lower()

    try:
        if extension == ".pdf":
            reader = PdfReader(str(path))
            pages = []

            for page in reader.pages:
                pages.append(page.extract_text() or "")

            return "\n".join(pages)

        if extension == ".docx":
            document = Document(str(path))
            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        if extension in {
            ".txt", ".md", ".json", ".log"
        }:
            return path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        if extension == ".csv":
            with path.open(
                "r",
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as file:
                return file.read()

        if extension in {".xlsx", ".xlsm"}:
            workbook = load_workbook(
                filename=str(path),
                read_only=True,
                data_only=True,
            )

            output = []

            for sheet in workbook.worksheets:
                output.append(
                    f"\nFeuille : {sheet.title}"
                )

                for row in sheet.iter_rows(
                    values_only=True
                ):
                    values = [
                        str(value) if value is not None else ""
                        for value in row
                    ]

                    output.append(" | ".join(values))

            workbook.close()
            return "\n".join(output)

        return (
            f"Le format {extension} n'est pas pris en charge."
        )

    except Exception as error:
        return f"Erreur de lecture du fichier : {error}"


# ==========================================
# PRÉPARATION DES PIÈCES JOINTES
# ==========================================

def prepare_attachments(files):
    parts = []

    if not files:
        return parts

    for file_path in files:
        if not file_path:
            continue

        path = Path(file_path)
        extension = path.suffix.lower()

        if extension in SUPPORTED_IMAGES:
            try:
                mime_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                }

                image_bytes = path.read_bytes()

                parts.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_types[extension],
                    )
                )

            except Exception as error:
                parts.append(
                    types.Part.from_text(
                        text=(
                            f"Impossible de lire l'image "
                            f"{path.name} : {error}"
                        )
                    )
                )

        elif extension in SUPPORTED_DOCUMENTS:
            text_content = extract_text_file(path)

            # Limite de sécurité pour éviter
            # d'envoyer des documents trop volumineux.
            text_content = text_content[:50000]

            parts.append(
                types.Part.from_text(
                    text=(
                        f"\nDOCUMENT : {path.name}\n"
                        f"{text_content}"
                    )
                )
            )

        else:
            parts.append(
                types.Part.from_text(
                    text=(
                        f"Le fichier {path.name} "
                        "n'est pas pris en charge."
                    )
                )
            )

    return parts


# ==========================================
# HISTORIQUE DE CONVERSATION
# ==========================================

def format_history(history):
    formatted = []

    if not history:
        return formatted

    for item in history[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"}:
            continue

        if isinstance(content, str) and content.strip():
            formatted.append(
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[
                        types.Part.from_text(text=content)
                    ],
                )
            )

    return formatted


# ==========================================
# CONSTRUCTION DU PROMPT
# ==========================================

def build_prompt(message, knowledge):
    prompt = f"""
Tu es {APP_NAME}, un assistant IA intelligent,
professionnel, fiable et utile.

Tes objectifs :
- Répondre clairement aux questions.
- Expliquer les sujets complexes simplement.
- Aider à rédiger, analyser et organiser des informations.
- Aider à la création de contenus et à la résolution de problèmes.
- Analyser les documents et les images fournis.
- Répondre en français si l'utilisateur écrit en français.

Règles importantes :
- N'invente pas des informations présentées comme certaines.
- Si tu ne connais pas une réponse, indique-le honnêtement.
- Respecte les instructions de l'utilisateur lorsqu'elles sont sûres.
- Utilise les documents joints lorsqu'ils sont pertinents.
- Distingue les faits des hypothèses.

"""

    if knowledge:
        prompt += f"""
CONNAISSANCES DISPONIBLES :

{knowledge}

Utilise ces informations uniquement lorsqu'elles sont pertinentes.
"""

    prompt += f"""

QUESTION DE L'UTILISATEUR :

{message}
"""

    return prompt


# ==========================================
# RÉPONSE DE L'ASSISTANT
# ==========================================

def respond(message, history, files):
    if not message and not files:
        return "Écris une question ou ajoute un fichier."

    try:
        client = get_gemini_client()

        question = message or (
            "Analyse les fichiers joints et explique leur contenu."
        )

        knowledge = load_knowledge(question)
        prompt = build_prompt(question, knowledge)

        contents = format_history(history)

        user_parts = [
            types.Part.from_text(text=prompt)
        ]

        user_parts.extend(
            prepare_attachments(files)
        )

        contents.append(
            types.Content(
                role="user",
                parts=user_parts,
            )
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=4096,
            ),
        )

        answer = response.text

        if not answer:
            answer = (
                "Je n'ai pas pu générer de réponse. "
                "Essaie de reformuler ta question."
            )

        return answer

    except Exception as error:
        print(f"Erreur AYZORA AI : {error}")

        return (
            "Une erreur est survenue pendant le traitement "
            "de ta demande.\n\n"
            f"Détail : {error}\n\n"
            "Vérifie la clé API Gemini et les journaux "
            "de déploiement sur Render."
        )


# ==========================================
# INTERFACE GRADIO
# ==========================================

with gr.Blocks(title=APP_NAME) as demo:
    gr.Markdown(
        f"""
        # 🟠 {APP_NAME}

        **Ton assistant intelligent propulsé par Google Gemini.**

        Pose une question, ajoute un document ou envoie une image.
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        type="messages",
        height=520,
    )

    message = gr.Textbox(
        label="Ton message",
        placeholder="Pose ta question à AYZORA AI...",
        lines=3,
    )

    files = gr.File(
        label="Ajouter des documents ou des images",
        file_count="multiple",
        type="filepath",
        file_types=[
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".json",
            ".log",
            ".csv",
            ".xlsx",
            ".xlsm",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        ],
    )

    with gr.Row():
        send_button = gr.Button(
            "Envoyer",
            variant="primary",
        )

        clear_button = gr.Button("Effacer la conversation")

    def user_message(user_text, current_history, uploaded_files):
        if current_history is None:
            current_history = []

        if not user_text and not uploaded_files:
            return current_history, ""

        display_text = user_text or (
            "Analyse les fichiers joints."
        )

        new_history = current_history + [
            {
                "role": "user",
                "content": display_text,
            }
        ]

        answer = respond(
            user_text,
            current_history,
            uploaded_files,
        )

        new_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return new_history, ""

    send_button.click(
        fn=user_message,
        inputs=[message, chatbot, files],
        outputs=[chatbot, message],
    )

    message.submit(
        fn=user_message,
        inputs=[message, chatbot, files],
        outputs=[chatbot, message],
    )

    clear_button.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, message],
    )


# ==========================================
# LANCEMENT
# ==========================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
