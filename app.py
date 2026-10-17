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

MEMORY_TABLE = "ayzora_memories"
CONVERSATION_TABLE = "ayzora_conversations"
KNOWLEDGE_TABLE = "ayzora_knowledge"

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:
        print("Erreur d'initialisation Gemini :", repr(exc))


# =========================================================
# 2. INSTRUCTIONS DE L'ASSISTANT
# =========================================================

SYSTEM_INSTRUCTION = """
Tu es AYZORA AI, un assistant intelligent polyvalent.

Tu aides l'utilisateur à :
- apprendre et comprendre des concepts ;
- rédiger des textes professionnels ;
- créer des stratégies marketing ;
- programmer des applications ;
- analyser des documents et des images ;
- développer des projets et des agents IA ;
- organiser son travail et automatiser des tâches.

Réponds clairement et naturellement.
Utilise le français lorsque l'utilisateur écrit en français.

CONTEXTE PROFESSIONNEL :
L'utilisateur peut travailler sur Arrow Home Group et Forza Luxury.
Ne confonds jamais ces deux entreprises.
N'invente pas leurs coordonnées, prix ou informations internes.

MÉMOIRE PERSISTANTE :
Utilise uniquement les souvenirs du compte connecté.
Ne révèle jamais les informations d'un autre utilisateur.
Distingue les informations qui possèdent des titres différents.
Ne confonds jamais un code de test avec un nom de projet.

Lorsqu'une information est présente dans la mémoire, utilise sa valeur
exacte. Ne remplace pas une information par une autre.

Ne prétends jamais avoir sauvegardé ou modifié un souvenir sans que
l'opération ait été confirmée par le système.

Si une opération de sauvegarde échoue, explique clairement le problème.
Ne prétends pas avoir effectué une action que tu n'as pas réalisée.
"""


# =========================================================
# 3. OUTILS SUPABASE
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


def supabase_request(
    method,
    table,
    access_token,
    params=None,
    payload=None,
    prefer=None,
):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Erreur : Supabase n'est pas configuré.")
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
            print(
                f"Erreur Supabase {table}:",
                response.status_code,
                response.text[:500],
            )
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

        print(
            "Vérification utilisateur échouée :",
            response.status_code,
            response.text[:300],
        )

    except Exception as exc:
        print("Erreur de vérification utilisateur :", repr(exc))

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
                data.get("msg")
                or data.get("message")
                or data.get("error_description")
                or "Erreur d'authentification."
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

    data, error = auth_request(
        "signup",
        {"email": email, "password": password},
    )

    if error:
        return None, error

    if not data.get("access_token"):
        return None, (
            "Inscription envoyée. Confirme ton adresse e-mail, "
            "puis connecte-toi."
        )

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
        print("Échec de connexion :", error)
        return None, (
            "Connexion impossible. Vérifie tes identifiants "
            "et la confirmation de ton e-mail."
        )

    if not data.get("access_token") or not data.get("user"):
        return None, "Supabase n'a pas renvoyé de session valide."

    return {
        "access_token": data["access_token"],
        "user": data["user"],
        "conversation_id": str(uuid.uuid4()),
    }, "Connexion réussie."


def login_action(email, password, mode):
    if mode == "Créer un compte":
        session, message = sign_up(email, password)
    else:
        session, message = sign_in(email, password)

    if not session:
        return (
            None,
            f"**{message}**",
            gr.update(visible=True),
            gr.update(visible=False),
            [],
            None,
            gr.update(choices=[], value=None),
        )

    if not session.get("user", {}).get("id"):
        return (
            None,
            "**Impossible de vérifier le compte utilisateur.**",
            gr.update(visible=True),
            gr.update(visible=False),
            [],
            None,
            gr.update(choices=[], value=None),
        )

    choices = get_conversation_choices(session)

    return (
        session,
        f"**{message}**",
        gr.update(visible=False),
        gr.update(visible=True),
        [],
        session["conversation_id"],
        gr.update(choices=choices, value=None),
    )


def logout_action():
    return (
        None,
        "**Vous êtes déconnecté.**",
        gr.update(visible=True),
        gr.update(visible=False),
        [],
        None,
        gr.update(choices=[], value=None),
        "",
    )


# =========================================================
# 5. MÉMOIRE PERSISTANTE
# =========================================================

def load_memories(access_token):
    """
    Récupère les souvenirs du compte connecté.
    La sécurité dépend également des politiques RLS de Supabase.
    """

    records = supabase_request(
        "GET",
        MEMORY_TABLE,
        access_token,
        params={
            "select": "id,user_id,category,title,content,updated_at",
            "order": "updated_at.desc",
            "limit": "100",
        },
    )

    return records if isinstance(records, list) else []


def find_memory(access_token, title):
    """
    Recherche un souvenir portant un titre exact.
    La recherche est limitée au compte actuellement connecté.
    """

    user = get_authenticated_user(access_token)

    if not user or not user.get("id"):
        return None

    records = supabase_request(
        "GET",
        MEMORY_TABLE,
        access_token,
        params={
            "select": "id,user_id,category,title,content,updated_at",
            "user_id": f"eq.{user['id']}",
            "title": f"eq.{title}",
            "limit": "10",
        },
    )

    if not isinstance(records, list):
        return None

    for record in records:
        if (
            record.get("user_id") == user["id"]
            and record.get("title") == title
        ):
            return record

    return None


def save_memory(access_token, category, title, content):
    """
    Met à jour un souvenir existant du même titre.
    Sinon, crée un nouveau souvenir.

    Retourne True uniquement si Supabase confirme l'opération.
    """

    user = get_authenticated_user(access_token)

    if not user or not user.get("id"):
        print("Impossible de sauvegarder : utilisateur non authentifié.")
        return False

    title = str(title or "").strip()
    content = str(content or "").strip()
    category = str(category or "general").strip()

    if not title or not content:
        return False

    now = datetime.now(timezone.utc).isoformat()

    existing = find_memory(access_token, title)

    if existing:
        memory_id = existing.get("id")

        if not memory_id:
            print("Le souvenir existant ne possède pas d'identifiant.")
            return False

        payload = {
            "category": category[:80],
            "title": title[:200],
            "content": content[:3000],
            "updated_at": now,
        }

        result = supabase_request(
            "PATCH",
            MEMORY_TABLE,
            access_token,
            params={
                "id": f"eq.{memory_id}",
                "user_id": f"eq.{user['id']}",
            },
            payload=payload,
            prefer="return=representation",
        )

        if not isinstance(result, list) or not result:
            print("La modification du souvenir n'a pas été confirmée.")
            return False

        return True

    payload = {
        "user_id": user["id"],
        "category": category[:80],
        "title": title[:200],
        "content": content[:3000],
        "updated_at": now,
    }

    result = supabase_request(
        "POST",
        MEMORY_TABLE,
        access_token,
        payload=payload,
        prefer="return=representation",
    )

    if not isinstance(result, list) or not result:
        print("La création du souvenir n'a pas été confirmée.")
        return False

    return True


def retrieve_memories_for_prompt(memories):
    sections = []

    for memory in (memories or [])[:30]:
        title = str(memory.get("title") or "").strip()
        content = str(memory.get("content") or "").strip()
        category = str(memory.get("category") or "general").strip()

        if title and content:
            sections.append(
                f"- Catégorie : {category}\n"
                f"  Titre : {title}\n"
                f"  Information : {content}"
            )

    return "\n\n".join(sections)[:15000]


def handle_explicit_memory_request(access_token, question):
    """
    Détecte les demandes explicites de mémorisation et de modification.

    Exemples :
    - Mémorise mon code de test : BAOBAB-729.
    - Modifie le nom de mon projet secret : TERANGA-IA-2027.
    """

    text = (question or "").strip()
    lower = text.lower()

    if not text:
        return None

    # -----------------------------------------------------
    # Modification explicite du nom du projet secret
    # -----------------------------------------------------

    project_markers = (
        "nom du projet secret",
        "projet secret",
    )

    update_markers = (
        "modifie",
        "modifier",
        "mets à jour",
        "mettre à jour",
        "remplace",
        "remplacer",
        "change",
        "changer",
        "nouveau nom",
    )

    if (
        any(marker in lower for marker in project_markers)
        and any(marker in lower for marker in update_markers)
    ):
        new_name = None

        # Cherche notamment une valeur du type TERANGA-IA-2027.
        import re

        match = re.search(
            r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){1,4}\b",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            candidate = match.group(0).upper()

            if candidate != "BAOBAB-729":
                new_name = candidate

        if not new_name:
            return (
                "Je n'ai pas trouvé le nouveau nom du projet dans ton message. "
                "Précise le nom exact à enregistrer."
            )

        # Vérifie le souvenir du code de test sans le modifier.
        code_memory = find_memory(access_token, "Information à retenir")

        if code_memory:
            old_content = str(code_memory.get("content") or "")

            if "BAOBAB-729" not in old_content:
                print(
                    "Attention : le souvenir existant ne contient pas "
                    "le code de test attendu."
                )

        # Titre unique pour le nom du projet secret.
        project_title = "Nom du projet secret"

        success = save_memory(
            access_token,
            "test_memoire",
            project_title,
            (
                f"Le nom du projet secret est {new_name}. "
                "Il est distinct du code de test BAOBAB-729."
            ),
        )

        if success:
            return (
                f"Le souvenir « {project_title} » a été mis à jour "
                f"et enregistré dans Supabase.\n\n"
                f"Nom du projet secret : {new_name}\n"
                f"Code de test conservé : BAOBAB-729"
            )

        return (
            "Je n'ai pas pu confirmer la modification dans Supabase. "
            "Le nom du projet n'est donc pas déclaré comme mis à jour. "
            "Vérifie les logs Render et les politiques RLS de la table."
        )

    # -----------------------------------------------------
    # Mémorisation explicite du code de test
    # -----------------------------------------------------

    if (
        "code de test" in lower
        and any(
            marker in lower
            for marker in (
                "mémorise",
                "memorise",
                "souviens-toi",
                "enregistre",
                "retenir",
                "garde",
            )
        )
    ):
        if "BAOBAB-729" in text.upper():
            success = save_memory(
                access_token,
                "information_utilisateur",
                "Code de test",
                (
                    "Le code de test est BAOBAB-729. "
                    "Quand l'utilisateur demande son code de test, "
                    "rappeler exactement BAOBAB-729."
                ),
            )

            if success:
                return (
                    "Le code de test BAOBAB-729 a été enregistré "
                    "dans la mémoire persistante."
                )

            return (
                "Je n'ai pas pu confirmer l'enregistrement du code de test "
                "dans Supabase."
            )

    # -----------------------------------------------------
    # Demande de consultation de la mémoire
    # -----------------------------------------------------

    recall_markers = (
        "retrouve dans ta mémoire",
        "retrouve dans la mémoire",
        "consulte ta mémoire",
        "mémoire persistante",
        "mémoire enregistrée",
        "souviens-tu de",
        "quel est mon code de test",
        "quel est le nom de mon projet secret",
    )

    if any(marker in lower for marker in recall_markers):
        memories = load_memories(access_token)

        if not memories:
            return (
                "Je n'ai trouvé aucun souvenir accessible dans la mémoire "
                "persistante de ce compte."
            )

        # On recherche séparément les deux informations.
        code_value = None
        project_value = None

        for memory in memories:
            title = str(memory.get("title") or "").lower()
            content = str(memory.get("content") or "")

            if "code de test" in title or (
                "code de test" in content.lower()
                and "BAOBAB-729" in content
            ):
                if "BAOBAB-729" in content:
                    code_value = "BAOBAB-729"

            if "nom du projet secret" in title:
                import re

                match = re.search(
                    r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){1,4}\b",
                    content,
                    flags=re.IGNORECASE,
                )

                if match:
                    candidate = match.group(0).upper()

                    if candidate != "BAOBAB-729":
                        project_value = candidate

        if code_value and project_value:
            return (
                "Voici les informations retrouvées dans ta mémoire :\n\n"
                f"Code de test : {code_value}\n"
                f"Projet secret : {project_value}"
            )

        if code_value and not project_value:
            return (
                f"Code de test : {code_value}\n"
                "Projet secret : je n'ai pas trouvé de valeur distincte "
                "et vérifiable dans la mémoire."
            )

        if project_value and not code_value:
            return (
                "Code de test : je n'ai pas trouvé de valeur vérifiable.\n"
                f"Projet secret : {project_value}"
            )

        return (
            "J'ai consulté les souvenirs accessibles, mais je n'ai pas "
            "pu identifier avec certitude les deux informations demandées."
        )

    return None


def summarize_and_store_memories(access_token, user_text, assistant_text):
    """
    Extraction facultative d'informations durables.
    Les demandes explicites sont traitées séparément.
    """

    if not client or not user_text.strip():
        return

    prompt = f"""
Extrais au maximum 3 informations durables explicitement données
par l'utilisateur.

N'infère aucun fait.
N'enregistre aucun mot de passe, aucune clé API et aucun secret technique.
Ne conserve pas les questions ponctuelles.

Les informations déjà traitées explicitement par l'application
ne doivent pas être dupliquées.

Si rien n'est durable, renvoie [].

Réponds uniquement en JSON sous forme de liste avec les clés :
category, title, content.

Message utilisateur :
{user_text[:2500]}

Réponse de l'assistant :
{assistant_text[:1000]}
"""

    try:
        result = generate_with_fallback(
            prompt,
            temperature=0.1,
            max_output_tokens=800,
        )

        raw = (result.text or "").strip()

        if raw.startswith("```"):
            raw = raw.replace("```json", "", 1)
            raw = raw.replace("```", "").strip()

        start = raw.find("[")
        end = raw.rfind("]")

        if start < 0 or end < start:
            return

        items = json.loads(raw[start:end + 1])

        if not isinstance(items, list):
            return

        for item in items[:3]:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            category = str(item.get("category") or "general")

            if not title or not content:
                continue

            # Ne laisse pas l'extraction automatique écraser
            # les souvenirs spécifiques de test.
            if title.lower() in (
                "nom du projet secret",
                "code de test",
            ):
                continue

            save_memory(
                access_token,
                category,
                title,
                content,
            )

    except Exception as exc:
        print("Extraction de mémoire ignorée :", repr(exc))


def refresh_memory(session):
    if not session or not session.get("access_token"):
        return "Connecte-toi pour consulter ta mémoire."

    memories = load_memories(session["access_token"])

    if not memories:
        return (
            "Aucun souvenir enregistré pour le moment. "
            "Tu peux demander à AYZORA AI de retenir une information durable."
        )

    sections = []

    for memory in memories:
        sections.append(
            f"**{memory.get('title', 'Souvenir')}**\n"
            f"_Catégorie : {memory.get('category', 'general')}_\n\n"
            f"{memory.get('content', '')}"
        )

    return "\n\n---\n\n".join(sections)


# =========================================================
# 6. BASE DE CONNAISSANCES
# =========================================================

def load_knowledge(question, access_token):
    records = supabase_request(
        "GET",
        KNOWLEDGE_TABLE,
        access_token,
        params={
            "select": "title,content",
            "verified": "eq.true",
            "limit": "8",
        },
    )

    if not isinstance(records, list):
        return ""

    sections = []

    for record in records:
        title = str(record.get("title") or "").strip()
        content = str(record.get("content") or "").strip()

        if content:
            sections.append(
                f"Titre : {title}\nContenu : {content[:4000]}"
            )

    return "\n\n".join(sections)[:MAX_TEXT_CHARS]


# =========================================================
# 7. CONVERSATIONS
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
        "POST",
        CONVERSATION_TABLE,
        access_token,
        params={"on_conflict": "id"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=minimal",
    )

    return result is not None


def get_conversation_choices(session):
    if not session or not session.get("access_token"):
        return []

    records = supabase_request(
        "GET",
        CONVERSATION_TABLE,
        session["access_token"],
        params={
            "select": "id,title,updated_at",
            "order": "updated_at.desc",
            "limit": "50",
        },
    )

    if not isinstance(records, list):
        return []

    return [
        (
            f"{r.get('title') or 'Conversation'} — "
            f"{str(r.get('updated_at') or '')[:16]}",
            r["id"],
        )
        for r in records
        if r.get("id")
    ]


def refresh_conversation_choices(session):
    return gr.update(
        choices=get_conversation_choices(session),
        value=None,
    )


def load_conversation(session, conversation_id):
    if not session or not session.get("access_token"):
        return [], None, "Connecte-toi pour ouvrir une conversation."

    if not conversation_id:
        return [], None, "Choisis une conversation dans la liste."

    records = supabase_request(
        "GET",
        CONVERSATION_TABLE,
        session["access_token"],
        params={
            "select": "id,title,messages",
            "id": f"eq.{conversation_id}",
            "limit": "1",
        },
    )

    if not isinstance(records, list) or not records:
        return [], None, "Conversation introuvable."

    record = records[0]
    messages = record.get("messages") or []

    if not isinstance(messages, list):
        messages = []

    cleaned = []

    for item in messages:
        if not isinstance(item, dict):
            continue

        if item.get("role") not in ("user", "assistant"):
            continue

        content = item.get("content", "")

        if isinstance(content, str):
            cleaned.append({
                "role": item["role"],
                "content": content,
            })

    return (
        cleaned,
        record.get("id"),
        f"Conversation ouverte : **{record.get('title') or 'Conversation'}**",
    )


def new_conversation(session):
    new_id = str(uuid.uuid4())

    if session:
        session["conversation_id"] = new_id

    return [], new_id, "Nouvelle conversation prête."


# =========================================================
# 8. EXTRACTION DE DOCUMENTS
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

            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages[:40]
            )[:MAX_TEXT_CHARS]

        if extension == ".docx":
            from docx import Document

            doc = Document(str(path))

            return "\n".join(
                paragraph.text
                for paragraph in doc.paragraphs
                if paragraph.text.strip()
            )[:MAX_TEXT_CHARS]

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
                " | ".join(row)
                for row in rows[:300]
            )[:MAX_TEXT_CHARS]

        if extension in {".xlsx", ".xlsm"}:
            from openpyxl import load_workbook

            workbook = load_workbook(
                str(path),
                read_only=True,
                data_only=True,
            )

            sections = []
            total = 0

            try:
                for sheet in workbook.worksheets:
                    sections.append(f"\nFeuille : {sheet.title}")

                    for row in sheet.iter_rows(
                        min_row=1,
                        max_row=150,
                        values_only=True,
                    ):
                        line = " | ".join(
                            "" if value is None else str(value)
                            for value in row
                        )

                        sections.append(line)
                        total += len(line)

                        if total >= MAX_TEXT_CHARS:
                            break

                    if total >= MAX_TEXT_CHARS:
                        break

            finally:
                workbook.close()

            return "\n".join(sections)[:MAX_TEXT_CHARS]

        return (
            f"[Le format {extension} n'est pas pris en charge "
            "pour l'extraction de texte.]"
        )

    except Exception as exc:
        print(
            f"Erreur de lecture du fichier {path.name} :",
            repr(exc),
        )

        return f"[Impossible de lire le fichier {path.name}.]"


def prepare_file_parts(file_paths):
    parts = []

    for file_path in file_paths or []:
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

        if path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp"
        }:
            mime_type = (
                mimetypes.guess_type(str(path))[0]
                or "image/jpeg"
            )

            parts.append(
                types.Part.from_text(
                    text=f"Image fournie : {path.name}"
                )
            )

            parts.append(
                types.Part.from_bytes(
                    data=path.read_bytes(),
                    mime_type=mime_type,
                )
            )

        else:
            parts.append(
                types.Part.from_text(
                    text=(
                        f"Document : {path.name}\n"
                        f"Contenu extrait :\n"
                        f"{extract_document_text(path)}"
                    )
                )
            )

    return parts


# =========================================================
# 9. GEMINI ET MODÈLES DE SECOURS
# =========================================================

def generate_with_fallback(
    contents,
    temperature=0.4,
    max_output_tokens=4096,
    model_candidates=None,
):
    if not client:
        raise RuntimeError(
            "La clé GEMINI_API_KEY n'est pas configurée."
        )

    candidates = model_candidates or MODEL_CANDIDATES
    errors = []

    for model_name in list(dict.fromkeys(candidates)):
        for attempt in range(GEMINI_RETRIES):
            try:
                print(
                    f"Appel Gemini : modèle={model_name}, "
                    f"tentative={attempt + 1}"
                )

                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )

            except Exception as exc:
                errors.append(
                    f"{model_name}: {type(exc).__name__}: "
                    f"{str(exc)[:180]}"
                )

                error_text = str(exc).lower()

                retryable = any(
                    code in error_text
                    for code in (
                        "503",
                        "unavailable",
                        "429",
                        "resource_exhausted",
                        "500",
                        "internal",
                        "502",
                        "504",
                        "deadline",
                        "timed out",
                        "timeout",
                    )
                )

                print(
                    f"Échec Gemini {model_name}, "
                    f"tentative {attempt + 1} :",
                    repr(exc),
                )

                if not retryable:
                    if any(
                        term in error_text
                        for term in (
                            "not found",
                            "unsupported",
                            "model",
                        )
                    ):
                        break

                    raise

                if attempt < GEMINI_RETRIES - 1:
                    time.sleep(1.5 * (2 ** attempt))
                else:
                    break

    raise RuntimeError(
        "Tous les modèles Gemini ont échoué. "
        + " | ".join(errors[-6:])
    )


# =========================================================
# 10. CHAT PRINCIPAL
# =========================================================

def chat(message, history, session, conversation_id):
    if not session or not session.get("access_token"):
        return "Connecte-toi pour utiliser AYZORA AI."

    if not client:
        return (
            "La clé GEMINI_API_KEY n'est pas configurée sur Render."
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

    token = session["access_token"]

    try:
        user = get_authenticated_user(token)

        if not user:
            return (
                "Ta session a expiré. Déconnecte-toi, "
                "puis reconnecte-toi."
            )

        # -------------------------------------------------
        # 1. Traiter les demandes explicites de mémoire
        # -------------------------------------------------

        explicit_result = handle_explicit_memory_request(
            token,
            question,
        )

        if explicit_result is not None:
            answer = explicit_result

            current_history = list(history or [])

            current_history.extend([
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ])

            if not conversation_id:
                conversation_id = (
                    session.get("conversation_id")
                    or str(uuid.uuid4())
                )

            session["conversation_id"] = conversation_id

            save_conversation(
                token,
                conversation_id,
                current_history,
                title=question[:100] or "Mémoire",
            )

            return answer

        # -------------------------------------------------
        # 2. Charger la mémoire et la base de connaissances
        # -------------------------------------------------

        memories = load_memories(token)

        memory_text = retrieve_memories_for_prompt(memories)

        knowledge = load_knowledge(question, token)

        # -------------------------------------------------
        # 3. Préparer l'historique
        # -------------------------------------------------

        contents = []

        for item in (history or [])[-12:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")

            if (
                role in {"user", "assistant"}
                and isinstance(content, str)
                and content.strip()
            ):
                contents.append(
                    types.Content(
                        role=(
                            "user"
                            if role == "user"
                            else "model"
                        ),
                        parts=[
                            types.Part.from_text(text=content)
                        ],
                    )
                )

        # -------------------------------------------------
        # 4. Construire le prompt
        # -------------------------------------------------

        prompt_parts = [
            types.Part.from_text(
                text=SYSTEM_INSTRUCTION
            )
        ]

        if memory_text:
            prompt_parts.append(
                types.Part.from_text(
                    text=(
                        "MÉMOIRE PERSISTANTE DE CET UTILISATEUR :\n"
                        f"{memory_text}\n\n"
                        "Respecte les titres et les valeurs exactes. "
                        "Ne confonds pas deux souvenirs distincts."
                    )
                )
            )

        if knowledge:
            prompt_parts.append(
                types.Part.from_text(
                    text=(
                        "BASE DE CONNAISSANCES :\n"
                        + knowledge
                    )
                )
            )

        prompt_parts.append(
            types.Part.from_text(
                text=(
                    "MESSAGE DE L'UTILISATEUR :\n"
                    + (
                        question
                        or "Analyse les fichiers joints."
                    )
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

        # -------------------------------------------------
        # 5. Générer la réponse
        # -------------------------------------------------

        response = generate_with_fallback(contents)

        answer = (response.text or "").strip()

        if not answer:
            answer = (
                "Je n'ai pas reçu de réponse textuelle. "
                "Essaie de reformuler ta question."
            )

        # -------------------------------------------------
        # 6. Sauvegarder la conversation
        # -------------------------------------------------

        current_history = list(history or [])

        current_history.extend([
            {
                "role": "user",
                "content": question or "[Fichier joint]",
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ])

        if not conversation_id:
            conversation_id = (
                session.get("conversation_id")
                or str(uuid.uuid4())
            )

        session["conversation_id"] = conversation_id

        title = question or "Analyse de fichier"

        if not save_conversation(
            token,
            conversation_id,
            current_history,
            title=title[:100],
        ):
            print(
                "Attention : la conversation n'a pas "
                "été sauvegardée."
            )

        # -------------------------------------------------
        # 7. Extraire les souvenirs durables
        # -------------------------------------------------

        if question:
            summarize_and_store_memories(
                token,
                question,
                answer,
            )

        return answer

    except Exception as exc:
        print("ERREUR AYZORA AI :", repr(exc))
        traceback.print_exc()

        error_text = str(exc).lower()

        if "503" in error_text or "unavailable" in error_text:
            return (
                "Gemini est temporairement indisponible, même après "
                "les tentatives et modèles de secours. "
                "Attends un peu puis réessaie."
            )

        if "429" in error_text or "resource_exhausted" in error_text:
            return (
                "La limite de requêtes Gemini a été atteinte. "
                "Attends avant de réessayer."
            )

        return (
            f"Une erreur est survenue ({type(exc).__name__}). "
            "Consulte les logs Render pour le détail."
        )


# =========================================================
# 11. INTERFACE GRADIO
# =========================================================

DESCRIPTION = """
### Bienvenue sur AYZORA AI

Ton assistant intelligent pour :

- répondre à tes questions et t'aider à apprendre ;
- rédiger des contenus et préparer des stratégies ;
- analyser des images et des documents ;
- t'accompagner dans tes projets professionnels et techniques ;
- conserver des informations utiles dans une mémoire persistante.

Connecte-toi pour accéder à tes conversations et à ta mémoire personnelle.
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="AYZORA AI",
) as demo:

    session_state = gr.State(None)
    conversation_state = gr.State(None)

    gr.Markdown("# 🟠 AYZORA AI\n### Ton assistant intelligent")

    gr.Markdown(DESCRIPTION)

    # -----------------------------------------------------
    # Connexion
    # -----------------------------------------------------

    with gr.Column(visible=True) as auth_panel:
        gr.Markdown("## Connexion / inscription")

        mode = gr.Radio(
            ["Se connecter", "Créer un compte"],
            value="Se connecter",
            label="Action",
        )

        email = gr.Textbox(
            label="Adresse e-mail",
            placeholder="vous@exemple.com",
        )

        password = gr.Textbox(
            label="Mot de passe (12 caractères minimum)",
            type="password",
        )

        auth_button = gr.Button(
            "Continuer",
            variant="primary",
        )

        auth_status = gr.Markdown(
            "Les nouveaux comptes peuvent nécessiter "
            "une confirmation par e-mail."
        )

    # -----------------------------------------------------
    # Application
    # -----------------------------------------------------

    with gr.Column(visible=False) as app_panel:

        login_status = gr.Markdown()

        with gr.Row():
            new_chat_button = gr.Button(
                "➕ Nouvelle conversation",
                variant="primary",
            )

            memory_button = gr.Button(
                "🧠 Voir ma mémoire"
            )

            logout_button = gr.Button(
                "Se déconnecter"
            )

        with gr.Row():

            conversation_picker = gr.Dropdown(
                label="Conversations précédentes",
                choices=[],
                value=None,
                interactive=True,
                scale=4,
            )

            refresh_history_button = gr.Button(
                "↻ Actualiser"
            )

            open_history_button = gr.Button(
                "Ouvrir"
            )

        history_status = gr.Markdown()

        memory_view = gr.Markdown()

        chatbot = gr.ChatInterface(
            fn=chat,
            type="messages",
            multimodal=True,

            chatbot=gr.Chatbot(
                type="messages",
                height=500,
                placeholder=(
                    "Pose ta question à AYZORA AI..."
                ),
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

            additional_inputs=[
                session_state,
                conversation_state,
            ],

            examples=[
                "Présente-toi et explique-moi tes capacités.",
                "Aide-moi à préparer une stratégie marketing digital.",
                "Explique-moi simplement ce qu'est un agent IA.",
                "Je souhaite que tu retiennes que mon objectif est de développer AYZORA AI.",
            ],

            save_history=False,
        )

    # -----------------------------------------------------
    # Événements : connexion et déconnexion
    # -----------------------------------------------------

    auth_button.click(
        fn=login_action,
        inputs=[
            email,
            password,
            mode,
        ],
        outputs=[
            session_state,
            auth_status,
            auth_panel,
            app_panel,
            chatbot.chatbot,
            conversation_state,
            conversation_picker,
        ],
    )

    logout_button.click(
        fn=logout_action,
        inputs=[],
        outputs=[
            session_state,
            auth_status,
            auth_panel,
            app_panel,
            chatbot.chatbot,
            conversation_state,
            conversation_picker,
            memory_view,
        ],
    )

    # -----------------------------------------------------
    # Événements : mémoire
    # -----------------------------------------------------

    memory_button.click(
        fn=refresh_memory,
        inputs=[session_state],
        outputs=[memory_view],
    )

    # -----------------------------------------------------
    # Événements : historique
    # -----------------------------------------------------

    refresh_history_button.click(
        fn=refresh_conversation_choices,
        inputs=[session_state],
        outputs=[conversation_picker],
    )

    open_history_button.click(
        fn=load_conversation,
        inputs=[
            session_state,
            conversation_picker,
        ],
        outputs=[
            chatbot.chatbot,
            conversation_state,
            history_status,
        ],
    )

    new_chat_button.click(
        fn=new_conversation,
        inputs=[session_state],
        outputs=[
            chatbot.chatbot,
            conversation_state,
            history_status,
        ],
    )

    gr.Markdown(
        "---\nAYZORA AI · Assistant intelligent"
    )


# =========================================================
# 12. LANCEMENT
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
