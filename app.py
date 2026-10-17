import os
import time
import uuid
import requests
import gradio as gr
from google import genai
from google.genai import types

# ============================================================
# AYZORA AI — APPLICATION PRINCIPALE
# ============================================================

APP_NAME = "AYZORA AI"
APP_DESCRIPTION = (
    "Votre assistant IA pour apprendre, créer, réfléchir et travailler."
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()

MODEL_CANDIDATES = [
    model.strip()
    for model in os.getenv(
        "GEMINI_MODELS",
        "gemini-2.5-flash,gemini-2.0-flash"
    ).split(",")
    if model.strip()
]

SYSTEM_INSTRUCTION = """
Tu es AYZORA AI, un assistant intelligent, utile, clair et professionnel.

Tu réponds dans la langue utilisée par l'utilisateur, notamment en français.

Tu aides les utilisateurs à :
- Comprendre les concepts de l'intelligence artificielle.
- Créer du contenu et des stratégies marketing.
- Développer des projets et des applications.
- Programmer et résoudre des problèmes techniques.
- Organiser leurs activités professionnelles.
- Apprendre de nouvelles compétences.
- Générer des idées innovantes.

Tu expliques les sujets complexes simplement, avec des exemples concrets.

Tu ne prétends jamais avoir réalisé une action externe si ce n'est pas le cas.

Si une information est incertaine, tu le précises.
"""

_client = None


# ============================================================
# CONNEXION GEMINI
# ============================================================

def get_gemini_client():
    global _client

    if not GEMINI_API_KEY:
        return None

    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def call_gemini(prompt, history=None):
    client = get_gemini_client()

    if client is None:
        return (
            "⚠️ La clé GEMINI_API_KEY n'est pas configurée. "
            "Ajoute-la dans les variables d'environnement de Render."
        )

    history = history or []
    conversation = []

    for item in history[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if not isinstance(content, str):
            continue

        if role == "user":
            conversation.append(
                "Utilisateur : " + content
            )

        elif role == "assistant":
            conversation.append(
                "AYZORA AI : " + content
            )

    full_prompt = (
        SYSTEM_INSTRUCTION
        + "\n\n"
        + "\n".join(conversation)
        + "\n\nUtilisateur : "
        + str(prompt)
        + "\n\nRéponds comme AYZORA AI :"
    )

    last_error = None

    for model in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )

            answer = (response.text or "").strip()

            if answer:
                return answer

        except Exception as error:
            last_error = error
            error_message = str(error).lower()

            if any(term in error_message for term in [
                "429",
                "resource_exhausted",
                "quota",
                "rate limit",
            ]):
                continue

            if any(term in error_message for term in [
                "503",
                "unavailable",
                "overloaded",
                "500",
                "502",
                "504",
            ]):
                time.sleep(1)
                continue

    error_message = str(last_error or "").lower()

    if any(term in error_message for term in [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
    ]):
        return (
            "⚠️ Le quota Gemini semble atteint. "
            "Aucun des modèles configurés n'a pu répondre. "
            "Vérifie les limites de ton projet Google AI Studio "
            "ou attends le renouvellement du quota."
        )

    return (
        "Je n'ai pas réussi à obtenir une réponse de Gemini. "
        "Vérifie ta clé API, les modèles disponibles et les journaux Render."
    )
