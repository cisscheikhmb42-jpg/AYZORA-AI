import os
from pathlib import Path
import csv

import gradio as gr
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook

APP_NAME = "AYZORA AI"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_FILE_CHARS = 18000
MAX_HISTORY_MESSAGES = 12


def get_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "La clé API Gemini est absente. Ajoutez GEMINI_API_KEY dans les variables "
            "d'environnement de Render."
        )
    return genai.Client(api_key=api_key)


def extract_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".docx":
            doc = Document(str(path))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text.strip() for cell in row.cells))
            return "\n".join(parts)
        if suffix in (".xlsx", ".xlsm"):
            workbook = load_workbook(str(path), read_only=True, data_only=True)
            parts = []
            for sheet in workbook.worksheets:
                parts.append(f"Feuille : {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    values = [str(value) for value in row if value is not None]
                    if values:
                        parts.append(" | ".join(values))
            workbook.close()
            return "\n".join(parts)
        if suffix == ".csv":
            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                return "\n".join(" | ".join(row) for row in csv.reader(f))
        if suffix in (".txt", ".md", ".json", ".log"):
            return path.read_text(encoding="utf-8", errors="replace")
        return f"Format non pris en charge : {suffix or 'inconnu'}"
    except Exception as exc:
        return f"Impossible de lire {path.name} : {exc}"


def prepare_uploads(files) -> str:
    if not files:
        return ""
    chunks = []
    for file_path in files:
        if not file_path:
            continue
        text = extract_file_text(file_path)
        if text.strip():
            chunks.append(f"--- DOCUMENT : {Path(file_path).name} ---\n{text[:MAX_FILE_CHARS]}")
    return "\n\n".join(chunks)


def build_prompt(history, user_message, document_text):
    sections = [
        "Tu es AYZORA AI, un assistant IA utile, clair et professionnel.",
        "Réponds dans la langue utilisée par l'utilisateur, sauf demande contraire.",
        "Si tu ne connais pas une réponse ou si les documents ne la contiennent pas, dis-le clairement.",
    ]
    if document_text:
        sections.append(
            "Contenu des documents joints. Utilise-le si pertinent et ne prétends pas trouver "
            "des informations absentes de ces extraits :\n\n" + document_text
        )
    recent = (history or [])[-MAX_HISTORY_MESSAGES:]
    conversation = []
    for item in recent:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            label = "Utilisateur" if role == "user" else "AYZORA AI"
            conversation.append(f"{label} : {content}")
    if conversation:
        sections.append("Historique récent :\n" + "\n".join(conversation))
    sections.append("Nouveau message de l'utilisateur :\n" + user_message)
    sections.append("Réponds directement au nouveau message.")
    return "\n\n".join(sections)


def respond(message, history, files, document_text):
    history = list(history or [])
    if not message or not message.strip():
        return history, "", document_text or "", "Écris un message pour commencer."
    try:
        if files:
            document_text = prepare_uploads(files)
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=build_prompt(history, message.strip(), document_text or ""),
            config=types.GenerateContentConfig(temperature=0.6, max_output_tokens=2048),
        )
        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            answer = "Je n'ai pas pu générer de réponse. Réessaie avec une autre formulation."
        history.extend([
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": answer},
        ])
        return history, "", document_text or "", f"Réponse générée avec {MODEL_NAME}."
    except Exception as exc:
        error = str(exc)
        if "GEMINI_API_KEY" in error or "API key" in error:
            friendly = (
                "La clé API Gemini n'est pas configurée. Dans Render, ouvre Environment, "
                "ajoute GEMINI_API_KEY avec ta clé API, puis redéploie."
            )
        else:
            friendly = f"Une erreur est survenue : {error}"
        return history, message, document_text or "", friendly


def clear_chat():
    return [], "", None, "", "Conversation effacée."


with gr.Blocks(title=APP_NAME) as demo:
    gr.Markdown(
        f"# 🤖 {APP_NAME}\n"
        "Votre assistant IA pour discuter, analyser des documents et vous aider au quotidien."
    )
    with gr.Row():
        with gr.Column(scale=4):
            chat = gr.Chatbot(label="Conversation", type="messages", height=520)
            message_box = gr.Textbox(
                label="Votre message",
                placeholder="Écrivez votre question ici…",
                lines=2,
            )
            with gr.Row():
                send_button = gr.Button("Envoyer", variant="primary")
                clear_button = gr.Button("Effacer la conversation")
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Joindre des documents",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".xlsx", ".xlsm", ".csv", ".txt", ".md", ".json"],
                type="filepath",
            )
            status_box = gr.Markdown("Prêt.")
            gr.Markdown(
                "**Formats acceptés :** PDF, Word, Excel, CSV et fichiers texte.\n\n"
                "La clé Gemini doit être configurée dans les variables d'environnement de Render."
            )
    document_state = gr.State("")
    send_button.click(
        fn=respond,
        inputs=[message_box, chat, file_input, document_state],
        outputs=[chat, message_box, document_state, status_box],
    )
    message_box.submit(
        fn=respond,
        inputs=[message_box, chat, file_input, document_state],
        outputs=[chat, message_box, document_state, status_box],
    )
    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chat, message_box, file_input, document_state, status_box],
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
