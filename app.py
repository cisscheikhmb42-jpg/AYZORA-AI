import os
import csv
import json
import mimetypes
from pathlib import Path

import gradio as gr
import requests

from google import genai
from google.genai import types


# =========================================================
# 1. CONFIGURATION AYZORA AI
# =========================================================

APP_NAME = "AYZORA AI"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
).strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()

MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_TEXT_CHARS = 30000

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:
        print(f"Erreur d'initialisation Gemini : {exc}")


# =========================================================
# 2. INSTRUCTIONS DE L'ASSISTANT
# =========================================================

SYSTEM_INSTRUCTION = """
Tu es AYZORA AI, un assistant intelligent polyvalent.

OBJECTIFS :
- Répondre clairement, précisément et utilement.
- Comprendre les besoins de l'utilisateur.
- Aider à analyser des documents et des images.
- Aider à rédiger, apprendre, programmer et résoudre des problèmes.
- Proposer des solutions concrètes et structurées.
- Répondre en français lorsque l'utilisateur écrit en français.

CONTEXTE PROFESSIONNEL :
L'utilisateur peut travailler sur Arrow Home Group et Forza Luxury.
Ne confonds jamais ces deux entreprises.
N'invente pas leurs prix, coordonnées, services ou informations internes.
Utilise les informations disponibles dans les documents fournis
et dans la base de connaissances lorsqu'elles sont pertinentes.

RÈGLES :
- Si une information manque, explique-le honnêtement.
- Ne prétends pas avoir effectué une action que tu n'as pas réalisée.
- Pour le code informatique, fournis des exemples cohérents.
- Pour les documents et les images, analyse réellement leur contenu
  lorsqu'il est disponible.
- Sois professionnel, pédagogique et orienté vers les résultats.
"""


# =========================================================
# 3. BASE DE CONNAISSANCES SUPABASE
# =========================================================

def load_knowledge(question):
    """
    Récupère des informations depuis la table ayzora_knowledge.

    Variables nécessaires :
    SUPABASE_URL
    SUPABASE_ANON_KEY

    Colonnes attendues :
    id, title, content, verified
    """

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return ""

    endpoint = (
        f"{SUPABASE_URL}/rest/v1/ayzora_knowledge"
    )

    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }

    params = {
        "select": "title,content",
        "verified": "eq.true",
        "limit": "8",
    }

    try:
        response = requests.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=10,
        )

        response.raise_for_status()
        records = response.json()

        if not isinstance(records, list):
            return ""

        sections = []

        for record in records:
            title = str(record.get("title") or "").strip()
            content = str(record.get("content") or "").strip()

            if content:
                sections.append(
                    f"Titre : {title}\n"
                    f"Contenu : {content[:4000]}"
                )

        return "\n\n".join(sections)[:MAX_TEXT_CHARS]

    except Exception as exc:
        print(f"Supabase indisponible : {exc}")
        return ""


# =========================================================
# 4. EXTRACTION DU TEXTE DES DOCUMENTS
# =========================================================

def extract_document_text(file_path):
    """
    Extrait le texte des fichiers PDF, DOCX, TXT, MD,
    JSON, CSV et XLSX.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if not path.exists():
        return "[Fichier introuvable]"

    if path.stat().st_size > MAX_FILE_SIZE:
        return (
            f"[Le fichier {path.name} dépasse la limite "
            "de 15 Mo.]"
        )

    try:
        if extension == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = []

            for page in reader.pages[:40]:
                pages.append(page.extract_text() or "")

            return "\n".join(pages)[:MAX_TEXT_CHARS]

        if extension == ".docx":
            from docx import Document

            document = Document(str(path))

            paragraphs = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]

            return "\n".join(paragraphs)[:MAX_TEXT_CHARS]

        if extension in {".txt", ".md", ".log"}:
            return path.read_text(
                encoding="utf-8",
                errors="replace",
            )[:MAX_TEXT_CHARS]

        if extension == ".json":
            with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as file:
                data = json.load(file)

            return json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )[:MAX_TEXT_CHARS]

        if extension == ".csv":
            with path.open(
                "r",
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as file:
                rows = list(csv.reader(file))

            return "\n".join(
                " | ".join(row) for row in rows[:300]
            )[:MAX_TEXT_CHARS]

        if extension in {".xlsx", ".xlsm"}:
            from openpyxl import load_workbook

            workbook = load_workbook(
                str(path),
                read_only=True,
                data_only=True,
            )

            sections = []

            try:
                for sheet in workbook.worksheets:
                    sections.append(
                        f"\nFeuille : {sheet.title}"
                    )

                    for row in sheet.iter_rows(
                        min_row=1,
                        max_row=150,
                        values_only=True,
                    ):
                        values = [
                            str(value)
                            if value is not None
                            else ""
                            for value in row
                        ]

                        sections.append(" | ".join(values))

                        if sum(map(len, sections)) > MAX_TEXT_CHARS:
                            break

                    if sum(map(len, sections)) > MAX_TEXT_CHARS:
                        break

            finally:
                workbook.close()

            return "\n".join(sections)[:MAX_TEXT_CHARS]

        return (
            f"[Le format {extension} n'est pas pris en charge "
            "pour l'extraction de texte.]"
        )

    except Exception as exc:
        print(f"Erreur de lecture du fichier {path.name} : {exc}")
        return f"[Impossible de lire le fichier {path.name}.]"


# =========================================================
# 5. PRÉPARATION DES FICHIERS POUR GEMINI
# =========================================================

def prepare_file_parts(file_paths):
    """
    Prépare les documents et images pour Gemini.
    """

    parts = []

    if not file_paths:
        return parts

    for file_path in file_paths:
        if not file_path:
            continue

        path = Path(file_path)

        if not path.exists():
            parts.append(
                types.Part.from_text(
                    text=f"Fichier introuvable : {path.name}"
                )
            )
            continue

        if path.stat().st_size > MAX_FILE_SIZE:
            parts.append(
                types.Part.from_text(
                    text=f"Fichier trop volumineux : {path.name}"
                )
            )
            continue

        extension = path.suffix.lower()

        image_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }

        if extension in image_extensions:
            mime_type, _ = mimetypes.guess_type(str(path))

            if not mime_type:
                mime_type = "image/jpeg"

            with path.open("rb") as file:
                image_bytes = file.read()

            parts.append(
                types.Part.from_text(
                    text=f"Image fournie : {path.name}"
                )
            )

            parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )
            )

        else:
            extracted_text = extract_document_text(path)

            parts.append(
                types.Part.from_text(
                    text=(
                        f"Document : {path.name}\n"
                        f"Contenu extrait :\n{extracted_text}"
                    )
                )
            )

    return parts


# =========================================================
# 6. CONSTRUCTION DE L'HISTORIQUE
# =========================================================

def build_contents(message, history, file_paths):
    """
    Construit les messages envoyés à Gemini.
    """

    contents = []

    for item in (history or [])[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        gemini_role = "user" if role == "user" else "model"

        contents.append(
            types.Content(
                role=gemini_role,
                parts=[
                    types.Part.from_text(text=content)
                ],
            )
        )

    question = (message or "").strip()

    if not question and not file_paths:
        question = (
            "Bonjour, présente-toi et explique-moi "
            "comment tu peux m'aider."
        )

    knowledge = load_knowledge(question)

    prompt_parts = [
        types.Part.from_text(
            text=SYSTEM_INSTRUCTION
        )
    ]

    if knowledge:
        prompt_parts.append(
            types.Part.from_text(
                text=(
                    "INFORMATIONS DE LA BASE DE CONNAISSANCES :\n"
                    f"{knowledge}\n\n"
                    "Utilise ces informations uniquement "
                    "si elles sont pertinentes."
                )
            )
        )

    prompt_parts.append(
        types.Part.from_text(
            text=(
                "MESSAGE DE L'UTILISATEUR :\n"
                f"{question or 'Analyse les fichiers joints.'}"
            )
        )
    )

    prompt_parts.extend(
        prepare_file_parts(file_paths)
    )

    contents.append(
        types.Content(
            role="user",
            parts=prompt_parts,
        )
    )

    return contents


# =========================================================
# 7. FONCTION PRINCIPALE DE L'ASSISTANT
# =========================================================

def chat(message, history):
    """
    Fonction appelée par l'interface Gradio.
    """

    if not client:
        return (
            "La clé GEMINI_API_KEY n'est pas configurée "
            "ou l'initialisation de Gemini a échoué. "
            "Ajoute ta clé dans les variables d'environnement "
            "de Render, puis redémarre l'application."
        )

    if isinstance(message, dict):
        question = str(message.get("text") or "").strip()
        file_paths = message.get("files") or []
    else:
        question = str(message or "").strip()
        file_paths = []

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    if not question and not file_paths:
        return "Écris une question ou joins un fichier."

    try:
        contents = build_contents(
            question,
            history,
            file_paths,
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.4,
                max_output_tokens=4096,
            ),
        )

        answer = response.text

        if answer and answer.strip():
            return answer.strip()

        return (
            "Je n'ai pas pu produire de réponse textuelle. "
            "Essaie de reformuler ta question."
        )

    except Exception as exc:
        import traceback

        print("ERREUR AYZORA AI :", repr(exc))
        traceback.print_exc()

        return (
            "Erreur Gemini : "
            f"{type(exc).__name__}: {str(exc)}"
        )
# =========================================================
# 8. INTERFACE UTILISATEUR
# =========================================================

DESCRIPTION = """
### Bienvenue sur AYZORA AI

Ton assistant intelligent pour :
- Répondre à tes questions et t'aider à apprendre.
- Rédiger des contenus et préparer des stratégies.
- Analyser des images et des documents.
- T'accompagner dans tes projets professionnels et techniques.

**Comment commencer ?**

Écris ta question dans la zone de discussion.
Tu peux également joindre un document ou une image.
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="AYZORA AI",
) as demo:

    gr.Markdown(
        "# 🟠 AYZORA AI\n"
        "### Ton assistant intelligent"
    )

    gr.Markdown(DESCRIPTION)

    gr.ChatInterface(
        fn=chat,
        type="messages",
        multimodal=True,
        chatbot=gr.Chatbot(
            type="messages",
            height=500,
            placeholder="Pose ta première question à AYZORA AI...",
        ),
        textbox=gr.MultimodalTextbox(
            placeholder=(
                "Écris ta question ou joins un fichier..."
            ),
            file_count="multiple",
            file_types=[
                ".pdf",
                ".docx",
                ".txt",
                ".md",
                ".log",
                ".json",
                ".csv",
                ".xlsx",
                ".xlsm",
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            ],
            sources=["upload"],
        ),
        examples=[
            "Présente-toi et explique-moi tes capacités.",
            "Aide-moi à préparer une stratégie marketing digital.",
            "Explique-moi simplement ce qu'est un agent IA.",
        ],
        save_history=True,
    )

    gr.Markdown(
        "---\n"
        "AYZORA AI · Assistant intelligent"
    )


# =========================================================
# 9. DÉMARRAGE DE L'APPLICATION
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
