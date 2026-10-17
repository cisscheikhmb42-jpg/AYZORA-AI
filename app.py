import os
import csv
import json
import mimetypes
import time
import uuid
import traceback
from pathlib import Path
from datetime import datetime, timezone

import gradio as gr
import requests
from google import genai
from google.genai import types


# =========================================================
# 1. CONFIGURATION
# =========================================================

APP_NAME = "AYZORA AI"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

# Ordre de secours configurable depuis Render.
FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.6-flash,gemini-3.5-flash-lite"
    ).split(",")
    if model.strip()
]
MODEL_CANDIDATES = list(dict.fromkeys([GEMINI_MODEL] + FALLBACK_MODELS))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = (
    os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
    or os.getenv("SUPABASE_ANON_KEY", "").strip()
)

MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_TEXT_CHARS = 30000
REST_TIMEOUT = 15
GEMINI_RETRIES = 2

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:
        print("Erreur d'initialisation Gemini :", repr(exc))


# =========================================================
# 2. INSTRUCTIONS
# =========================================================

SYSTEM_INSTRUCTION = """
Tu es AYZORA AI, un assistant intelligent polyvalent.
Réponds clairement, précisément et utilement, en français lorsque l'utilisateur
écrit en français. Aide à rédiger, apprendre, programmer, analyser des fichiers,
préparer des stratégies et résoudre des problèmes.

CONTEXTE PROFESSIONNEL :
L'utilisateur peut travailler sur Arrow Home Group et Forza Luxury.
Ne confonds jamais ces deux entreprises. N'invente pas leurs prix, coordonnées,
services ou informations internes. Utilise les informations fournies par
l'utilisateur, les documents, la mémoire et la base de connaissances.

MÉMOIRE ET CONFIDENTIALITÉ :
Utilise uniquement les souvenirs du compte actuellement connecté.
Ne révèle jamais les souvenirs d'un autre utilisateur.
Ne prétends pas avoir sauvegardé une information si la sauvegarde n'a pas réussi.
N'invente pas d'informations et signale clairement les incertitudes.
Ne prétends pas avoir effectué une action que tu n'as pas réalisée.
"""


# =========================================================
# 3. SUPABASE REST
# =========================================================

def supabase_headers(access_token=None, prefer=None):
    if not SUPABASE_KEY:
        return {}
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {access_token or SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def supabase_request(method, table, access_token, params=None, payload=None, prefer=None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Supabase non configuré.")
        return None
    try:
        response = requests.request(
            method=method,
            url=f"{SUPABASE_URL}/rest/v1/{table}",
            headers=supabase_headers(access_token, prefer),
            params=params,
            json=payload,
            timeout=REST_TIMEOUT,
        )
        if not response.ok:
            print(f"Erreur Supabase {table}:", response.status_code, response.text[:500])
            return None
        if response.status_code == 204 or not response.content:
            return []
        return response.json()
    except Exception as exc:
        print(f"Erreur Supabase {table} :", repr(exc))
        return None


def get_authenticated_user(access_token):
    if not SUPABASE_URL or not SUPABASE_KEY or not access_token:
        return None
    try:
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers=supabase_headers(access_token),
            timeout=REST_TIMEOUT,
        )
        if response.ok:
            return response.json()
        print("Vérification utilisateur échouée :", response.status_code, response.text[:300])
    except Exception as exc:
        print("Erreur vérification utilisateur :", repr(exc))
    return None


# =========================================================
# 4. AUTHENTIFICATION
# =========================================================

def auth_request(endpoint, payload):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None, "Supabase n'est pas configuré sur Render."
    try:
        response = requests.post(
            f"{SUPABASE_URL}/auth/v1/{endpoint}",
            headers=supabase_headers(),
            json=payload,
            timeout=REST_TIMEOUT,
        )
        data = response.json() if response.content else {}
        if not response.ok:
            return None, (
                data.get("msg") or data.get("message")
                or data.get("error_description") or "Erreur d'authentification."
            )
        return data, ""
    except Exception as exc:
        print("Erreur Supabase Auth :", repr(exc))
        return None, "Impossible de joindre Supabase. Réessaie."


def sign_up(email, password):
    email = (email or "").strip().lower()
    if not email or not password:
        return None, "Saisis une adresse e-mail et un mot de passe."
    if len(password) < 12:
        return None, "Le mot de passe doit contenir au moins 12 caractères."
    data, error = auth_request("signup", {"email": email, "password": password})
    if error:
        return None, error
    if not data.get("access_token"):
        return None, "Inscription envoyée. Confirme ton adresse e-mail, puis connecte-toi."
    return {
        "access_token": data["access_token"],
        "user": data.get("user", {}),
        "conversation_id": str(uuid.uuid4()),
    }, "Compte créé et connecté."


def sign_in(email, password):
    email = (email or "").strip().lower()
    if not email or not password:
        return None, "Saisis ton adresse e-mail et ton mot de passe."
    data, error = auth_request(
        "token?grant_type=password",
        {"email": email, "password": password},
    )
    if error:
        print("Échec connexion :", error)
        return None, "Connexion impossible. Vérifie tes identifiants et la confirmation de ton e-mail."
    if not data.get("access_token") or not data.get("user"):
        return None, "Supabase n'a pas renvoyé de session valide."
    return {
        "access_token": data["access_token"],
        "user": data["user"],
        "conversation_id": str(uuid.uuid4()),
    }, "Connexion réussie."


def login_action(email, password, mode):
    session, message = (
        sign_up(email, password) if mode == "Créer un compte"
        else sign_in(email, password)
    )
    if not session:
        return (
            None, f"**{message}**", gr.update(visible=True),
            gr.update(visible=False), [], None, gr.update(choices=[], value=None),
        )
    if not session.get("user", {}).get("id"):
        return (
            None, "**Impossible de vérifier le compte utilisateur.**",
            gr.update(visible=True), gr.update(visible=False), [], None,
            gr.update(choices=[], value=None),
        )
    choices = get_conversation_choices(session)
    return (
        session, f"**{message}**", gr.update(visible=False),
        gr.update(visible=True), [], session["conversation_id"],
        gr.update(choices=choices, value=None),
    )


def logout_action():
    return (
        None, "**Vous êtes déconnecté.**", gr.update(visible=True),
        gr.update(visible=False), [], None, gr.update(choices=[], value=None),
        "",
    )


# =========================================================
# 5. KNOWLEDGE BASE AND MEMORY
# =========================================================

def load_knowledge(question, access_token):
    records = supabase_request(
        "GET", "ayzora_knowledge", access_token,
        params={"select": "title,content", "verified": "eq.true", "limit": "8"},
    )
    if not isinstance(records, list):
        return ""
    sections = []
    for record in records:
        title = str(record.get("title") or "").strip()
        content = str(record.get("content") or "").strip()
        if content:
            sections.append(f"Titre : {title}\nContenu : {content[:4000]}")
    return "\n\n".join(sections)[:MAX_TEXT_CHARS]


def load_memories(access_token):
    records = supabase_request(
        "GET", "ayzora_memories", access_token,
        params={
            "select": "id,category,title,content,updated_at",
            "order": "updated_at.desc", "limit": "30",
        },
    )
    return records if isinstance(records, list) else []


def save_memory(access_token, category, title, content):
    user = get_authenticated_user(access_token)
    if not user or not user.get("id"):
        return False
    payload = {
        "user_id": user["id"],
        "category": str(category or "general")[:80],
        "title": str(title or "")[:200],
        "content": str(content or "")[:3000],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    result = supabase_request(
        "POST", "ayzora_memories", access_token,
        payload=payload, prefer="return=minimal",
    )
    return result is not None


def retrieve_memories_for_prompt(memories):
    sections = []
    for memory in (memories or [])[:20]:
        title = str(memory.get("title") or "").strip()
        content = str(memory.get("content") or "").strip()
        category = str(memory.get("category") or "general").strip()
        if title and content:
            sections.append(f"- Catégorie : {category}\n  Souvenir : {title}\n  Information : {content}")
    return "\n".join(sections)[:12000]


def summarize_and_store_memories(access_token, user_text, assistant_text):
    # Best-effort : une panne pendant l'extraction ne doit jamais annuler la réponse.
    if not client or not user_text.strip():
        return
    prompt = f"""
Extrais au maximum 3 informations durables explicitement données par l'utilisateur.
N'infère aucun fait. N'enregistre aucun mot de passe, clé API, secret ou donnée sensible.
Ne conserve pas les questions ponctuelles. Si rien n'est durable, renvoie [].
Réponds uniquement en JSON sous forme de liste avec les clés category, title, content.

Message utilisateur :
{user_text[:2500]}

Réponse de l'assistant :
{assistant_text[:1000]}
"""
    try:
        result = generate_with_fallback(
            prompt, temperature=0.1, max_output_tokens=800,
            model_candidates=[GEMINI_MODEL] + FALLBACK_MODELS,
        )
        raw = (result.text or "").strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "", 1).replace("```", "").strip()
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end < start:
            return
        items = json.loads(raw[start:end + 1])
        if not isinstance(items, list):
            return
        for item in items[:3]:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                content = str(item.get("content") or "").strip()
                if title and content:
                    save_memory(access_token, item.get("category", "general"), title, content)
    except Exception as exc:
        print("Extraction de mémoire ignorée :", repr(exc))


def refresh_memory(session):
    if not session or not session.get("access_token"):
        return "Connecte-toi pour consulter ta mémoire."
    memories = load_memories(session["access_token"])
    if not memories:
        return "Aucun souvenir enregistré pour le moment. Tu peux demander à AYZORA de retenir une préférence ou une information durable."
    return "\n\n---\n\n".join(
        f"**{m.get('title', 'Souvenir')}**\n_Catégorie : {m.get('category', 'general')}_\n\n{m.get('content', '')}"
        for m in memories
    )


# =========================================================
# 6. CONVERSATION HISTORY
# =========================================================

def save_conversation(access_token, conversation_id, messages, title=None):
    user = get_authenticated_user(access_token)
    if not user or not user.get("id"):
        return False
    payload = {
        "id": conversation_id,
        "user_id": user["id"],
        "title": (title or "Conversation AYZORA")[:200],
        "messages": messages[-80:],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    result = supabase_request(
        "POST", "ayzora_conversations", access_token,
        params={"on_conflict": "id"}, payload=payload,
        prefer="resolution=merge-duplicates,return=minimal",
    )
    return result is not None


def get_conversation_choices(session):
    if not session or not session.get("access_token"):
        return []
    records = supabase_request(
        "GET", "ayzora_conversations", session["access_token"],
        params={
            "select": "id,title,updated_at",
            "order": "updated_at.desc", "limit": "50",
        },
    )
    if not isinstance(records, list):
        return []
    return [
        (f"{r.get('title') or 'Conversation'} — {str(r.get('updated_at') or '')[:16]}", r["id"])
        for r in records if r.get("id")
    ]


def refresh_conversation_choices(session):
    return gr.update(choices=get_conversation_choices(session), value=None)


def load_conversation(session, conversation_id):
    if not session or not session.get("access_token"):
        return [], None, "Connecte-toi pour ouvrir une conversation."
    if not conversation_id:
        return [], None, "Choisis une conversation dans la liste."
    records = supabase_request(
        "GET", "ayzora_conversations", session["access_token"],
        params={
            "select": "id,title,messages",
            "id": f"eq.{conversation_id}", "limit": "1",
        },
    )
    if not isinstance(records, list) or not records:
        return [], None, "Conversation introuvable."
    record = records[0]
    messages = record.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    # Ne conserver que le format messages accepté par Gradio Chatbot.
    cleaned = []
    for item in messages:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            content = item.get("content", "")
            if isinstance(content, str):
                cleaned.append({"role": item["role"], "content": content})
    return cleaned, record.get("id"), f"Conversation ouverte : **{record.get('title') or 'Conversation'}**"


def new_conversation(session):
    new_id = str(uuid.uuid4())
    if session:
        session["conversation_id"] = new_id
    return [], new_id, "Nouvelle conversation prête."


# =========================================================
# 7. FILE TEXT EXTRACTION
# =========================================================

def extract_document_text(file_path):
    path = Path(file_path)
    extension = path.suffix.lower()
    if not path.exists():
        return "[Fichier introuvable]"
    if path.stat().st_size > MAX_FILE_SIZE:
        return f"[Le fichier {path.name} dépasse la limite de 15 Mo.]"
    try:
        if extension == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages[:40])[:MAX_TEXT_CHARS]
        if extension == ".docx":
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())[:MAX_TEXT_CHARS]
        if extension in {".txt", ".md", ".log"}:
            return path.read_text(encoding="utf-8", errors="replace")[:MAX_TEXT_CHARS]
        if extension == ".json":
            with path.open("r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            return json.dumps(data, ensure_ascii=False, indent=2)[:MAX_TEXT_CHARS]
        if extension == ".csv":
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                rows = list(csv.reader(f))
            return "\n".join(" | ".join(row) for row in rows[:300])[:MAX_TEXT_CHARS]
        if extension in {".xlsx", ".xlsm"}:
            from openpyxl import load_workbook
            workbook = load_workbook(str(path), read_only=True, data_only=True)
            sections = []
            total = 0
            try:
                for sheet in workbook.worksheets:
                    sections.append(f"\nFeuille : {sheet.title}")
                    for row in sheet.iter_rows(min_row=1, max_row=150, values_only=True):
                        line = " | ".join("" if v is None else str(v) for v in row)
                        sections.append(line)
                        total += len(line)
                        if total >= MAX_TEXT_CHARS:
                            break
                    if total >= MAX_TEXT_CHARS:
                        break
            finally:
                workbook.close()
            return "\n".join(sections)[:MAX_TEXT_CHARS]
        return f"[Le format {extension} n'est pas pris en charge pour l'extraction de texte.]"
    except Exception as exc:
        print(f"Erreur de lecture du fichier {path.name} :", repr(exc))
        return f"[Impossible de lire le fichier {path.name}.]"


def prepare_file_parts(file_paths):
    parts = []
    for file_path in file_paths or []:
        if not file_path:
            continue
        path = Path(file_path)
        if not path.exists():
            parts.append(types.Part.from_text(text=f"Fichier introuvable : {path.name}"))
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            parts.append(types.Part.from_text(text=f"Fichier trop volumineux : {path.name}"))
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
            parts.append(types.Part.from_text(text=f"Image fournie : {path.name}"))
            parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))
        else:
            parts.append(types.Part.from_text(
                text=f"Document : {path.name}\nContenu extrait :\n{extract_document_text(path)}"
            ))
    return parts


# =========================================================
# 8. GEMINI RETRIES AND MODEL FALLBACK
# =========================================================

def generate_with_fallback(contents, temperature=0.4, max_output_tokens=4096, model_candidates=None):
    if not client:
        raise RuntimeError("La clé GEMINI_API_KEY n'est pas configurée.")
    candidates = model_candidates or MODEL_CANDIDATES
    errors = []
    for model_name in list(dict.fromkeys(candidates)):
        for attempt in range(GEMINI_RETRIES):
            try:
                print(f"Appel Gemini : modèle={model_name}, tentative={attempt + 1}")
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
            except Exception as exc:
                errors.append(f"{model_name}: {type(exc).__name__}: {str(exc)[:180]}")
                text = str(exc).lower()
                retryable = any(code in text for code in (
                    "503", "unavailable", "429", "resource_exhausted",
                    "500", "internal", "502", "504", "deadline",
                    "timed out", "timeout",
                ))
                print(f"Échec Gemini {model_name} tentative {attempt + 1} :", repr(exc))
                if not retryable:
                    # Une erreur de modèle peut justifier le passage au modèle suivant.
                    if any(term in text for term in ("not found", "unsupported", "model")):
                        break
                    raise
                if attempt < GEMINI_RETRIES - 1:
                    time.sleep(1.5 * (2 ** attempt))
                else:
                    break
    raise RuntimeError("Tous les modèles Gemini ont échoué. " + " | ".join(errors[-6:]))


# =========================================================
# 9. CHAT
# =========================================================

def chat(message, history, session, conversation_id):
    if not session or not session.get("access_token"):
        return "Connecte-toi pour utiliser AYZORA AI."
    if not client:
        return "La clé GEMINI_API_KEY n'est pas configurée sur Render."

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

    token = session["access_token"]
    try:
        user = get_authenticated_user(token)
        if not user:
            return "Ta session a expiré. Déconnecte-toi, puis reconnecte-toi."

        memories = load_memories(token)
        memory_text = retrieve_memories_for_prompt(memories)
        knowledge = load_knowledge(question, token)

        contents = []
        for item in (history or [])[-12:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            # L'historique de ChatInterface peut contenir des objets fichiers/images.
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                contents.append(types.Content(
                    role="user" if role == "user" else "model",
                    parts=[types.Part.from_text(text=content)],
                ))

        prompt_parts = [types.Part.from_text(text=SYSTEM_INSTRUCTION)]
        if memory_text:
            prompt_parts.append(types.Part.from_text(
                text="MÉMOIRE PERSISTANTE DE CET UTILISATEUR :\n"
                     f"{memory_text}\nUtilise-la seulement si elle est pertinente."
            ))
        if knowledge:
            prompt_parts.append(types.Part.from_text(
                text="BASE DE CONNAISSANCES VÉRIFIÉE :\n" + knowledge
            ))
        prompt_parts.append(types.Part.from_text(
            text="MESSAGE DE L'UTILISATEUR :\n" + (question or "Analyse les fichiers joints.")
        ))
        prompt_parts.extend(prepare_file_parts(file_paths))
        contents.append(types.Content(role="user", parts=prompt_parts))

        response = generate_with_fallback(contents)
        answer = (response.text or "").strip()
        if not answer:
            answer = "Je n'ai pas reçu de réponse textuelle. Essaie de reformuler ta question."

        # Persistance de la conversation.
        current_history = list(history or [])
        current_history.extend([
            {"role": "user", "content": question or "[Fichier joint]"},
            {"role": "assistant", "content": answer},
        ])
        if not conversation_id:
            conversation_id = session.get("conversation_id") or str(uuid.uuid4())
        session["conversation_id"] = conversation_id
        title = question or "Analyse de fichier"
        if not save_conversation(token, conversation_id, current_history, title=title[:100]):
            print("Attention : la conversation n'a pas été sauvegardée.")

        # Extraction de mémoire facultative. Ne doit pas modifier la réponse déjà générée.
        if question:
            summarize_and_store_memories(token, question, answer)

        return answer

    except Exception as exc:
        print("ERREUR AYZORA AI :", repr(exc))
        traceback.print_exc()
        error_text = str(exc).lower()
        if "503" in error_text or "unavailable" in error_text:
            return (
                "Gemini est temporairement indisponible, même après les tentatives "
                "et modèles de secours. Attends un peu puis réessaie."
            )
        if "429" in error_text or "resource_exhausted" in error_text:
            return "La limite de requêtes Gemini a été atteinte. Attends avant de réessayer."
        return (
            f"Une erreur est survenue ({type(exc).__name__}). "
            "Consulte les logs Render pour le détail."
        )


# =========================================================
# 10. INTERFACE
# =========================================================

DESCRIPTION = """
### Bienvenue sur AYZORA AI

Ton assistant intelligent pour :
- répondre à tes questions et t'aider à apprendre ;
- rédiger des contenus et préparer des stratégies ;
- analyser des images et des documents ;
- t'accompagner dans tes projets professionnels et techniques.

Connecte-toi pour accéder à tes conversations et à ta mémoire personnelle.
"""

with gr.Blocks(theme=gr.themes.Soft(), title="AYZORA AI") as demo:
    session_state = gr.State(None)
    conversation_state = gr.State(None)

    gr.Markdown("# 🟠 AYZORA AI\n### Ton assistant intelligent")
    gr.Markdown(DESCRIPTION)

    with gr.Column(visible=True) as auth_panel:
        gr.Markdown("## Connexion / inscription")
        mode = gr.Radio(
            ["Se connecter", "Créer un compte"],
            value="Se connecter", label="Action",
        )
        email = gr.Textbox(label="Adresse e-mail", placeholder="vous@exemple.com")
        password = gr.Textbox(label="Mot de passe (12 caractères minimum)", type="password")
        auth_button = gr.Button("Continuer", variant="primary")
        auth_status = gr.Markdown("Les nouveaux comptes peuvent nécessiter une confirmation par e-mail.")

    with gr.Column(visible=False) as app_panel:
        login_status = gr.Markdown()
        with gr.Row():
            new_chat_button = gr.Button("➕ Nouvelle conversation", variant="primary")
            memory_button = gr.Button("🧠 Voir ma mémoire")
            logout_button = gr.Button("Se déconnecter")

        with gr.Row():
            conversation_picker = gr.Dropdown(
                label="Conversations précédentes",
                choices=[],
                value=None,
                interactive=True,
                scale=4,
            )
            refresh_history_button = gr.Button("↻ Actualiser")
            open_history_button = gr.Button("Ouvrir")

        history_status = gr.Markdown()
        memory_view = gr.Markdown()

        chatbot = gr.ChatInterface(
            fn=chat,
            type="messages",
            multimodal=True,
            chatbot=gr.Chatbot(
                type="messages",
                height=500,
                placeholder="Pose ta question à AYZORA AI...",
            ),
            textbox=gr.MultimodalTextbox(
                placeholder="Écris ta question ou joins un fichier...",
                file_count="multiple",
                file_types=[
                    ".pdf", ".docx", ".txt", ".md", ".log", ".json", ".csv",
                    ".xlsx", ".xlsm", ".png", ".jpg", ".jpeg", ".webp",
                ],
                sources=["upload"],
            ),
            additional_inputs=[session_state, conversation_state],
            examples=[
                ["Présente-toi et explique-moi tes capacités.", None, None],
                ["Aide-moi à préparer une stratégie marketing digital.", None, None],
                ["Explique-moi simplement ce qu'est un agent IA.", None, None],
                ["Je souhaite que tu retiennes que mon objectif est de développer AYZORA AI.", None, None],
            ],
            save_history=False,
        )

    auth_button.click(
        fn=login_action,
        inputs=[email, password, mode],
        outputs=[
            session_state, auth_status, auth_panel, app_panel,
            chatbot.chatbot, conversation_state, conversation_picker,
        ],
    )
    logout_button.click(
        fn=logout_action,
        inputs=[],
        outputs=[
            session_state, auth_status, auth_panel, app_panel,
            chatbot.chatbot, conversation_state, conversation_picker, memory_view,
        ],
    )
    memory_button.click(
        fn=refresh_memory, inputs=[session_state], outputs=[memory_view],
    )
    refresh_history_button.click(
        fn=refresh_conversation_choices,
        inputs=[session_state], outputs=[conversation_picker],
    )
    open_history_button.click(
        fn=load_conversation,
        inputs=[session_state, conversation_picker],
        outputs=[chatbot.chatbot, conversation_state, history_status],
    )
    new_chat_button.click(
        fn=new_conversation,
        inputs=[session_state],
        outputs=[chatbot.chatbot, conversation_state, history_status],
    )

    gr.Markdown("---\nAYZORA AI · Assistant intelligent")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
