"""
Answer Generation Module.

Supports two modes:
1. LLM-based (Google Gemini API via google-genai) — rich, natural-language answers
2. Extractive fallback — highlights the most relevant sentences from retrieved chunks
   (works without any API key)
"""
import re
from google.genai import types

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from config import GEMINI_API_KEY, GEMINI_MODEL


# ═══════════════════════════════════════════════════════════════════════════════
#  Extractive Answer (no API needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_sentence(sentence: str, query_words: set) -> float:
    """Score a sentence by how many query words it contains (normalized)."""
    sent_words = set(sentence.lower().split())
    if not sent_words:
        return 0.0
    overlap = len(query_words & sent_words)
    return overlap / len(query_words) if query_words else 0.0


def generate_extractive_answer(question: str, retrieved_chunks: list) -> dict:
    """
    Extract the most relevant sentences from the retrieved chunks.
    No external API required.
    Returns dict with 'answer', 'sources', 'method'.
    """
    query_words = set(question.lower().split())
    stopwords = {"what", "is", "the", "a", "an", "how", "can", "do", "does",
                 "are", "of", "in", "for", "to", "and", "or", "if", "by", "at"}
    query_words -= stopwords

    scored_sentences = []
    for chunk in retrieved_chunks:
        text = chunk["text"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            score = _score_sentence(sent, query_words)
            if score > 0:
                scored_sentences.append((sent, score, chunk))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = scored_sentences[:5]

    if not top_sentences:
        answer = "No directly relevant sentences found. Please review the retrieved chunks below."
    else:
        answer_parts = []
        for sent, score, chunk in top_sentences:
            page = chunk.get("start_page", "?")
            section = chunk.get("section_title", "")
            ref = f"(Page {page}" + (f", {section}" if section else "") + ")"
            answer_parts.append(f"- {sent} {ref}")
        answer = "**Relevant excerpts from the handbook:**\n\n" + "\n\n".join(answer_parts)

    sources = []
    seen_ids = set()
    for chunk in retrieved_chunks:
        cid = chunk.get("chunk_id", id(chunk))
        if cid not in seen_ids:
            seen_ids.add(cid)
            sources.append({
                "page": chunk.get("start_page", "?"),
                "section": chunk.get("section_title", "Unknown"),
                "preview": chunk["text"][:150] + "...",
            })

    return {"answer": answer, "sources": sources, "method": "extractive"}


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM-based Answer (Google Gemini via google-genai)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_genai_client(api_key: str = ""):
    """Create a google.genai client."""
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai is not installed. Run: pip install google-genai")
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=key)


def _is_quota_exhausted(error: Exception) -> bool:
    """
    True  → hard daily quota exhaustion (free-tier limit hit, cannot retry).
    False → transient per-minute rate limit (safe to retry after a wait).
    """
    msg = str(error).lower()
    return "free_tier" in msg or "exceeded your current quota" in msg


def _friendly_error(error: Exception) -> str:
    """User-facing message for quota/API errors."""
    if _is_quota_exhausted(error):
        return (
            "Gemini free-tier daily quota exhausted — extractive answer shown below. "
            "Resets at midnight PT. Get a new key or upgrade at https://aistudio.google.com"
        )
    return f"Gemini API error: {str(error)[:200]}"


def generate_llm_answer(question: str, retrieved_chunks: list, api_key: str = "") -> dict:
    """
    Generate a grounded answer via Gemini.
    Retries 2x with exponential back-off for transient rate limits.
    Raises the last exception so the unified caller can fall back to extractive.
    """
    import time

    client = _get_genai_client(api_key)

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source {i+1} | Page {chunk.get('start_page','?')} | {chunk.get('section_title','Unknown')}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an academic advisor for NUST.
Answer the student's question using ONLY the context below.
If the answer is not in the context, say "I cannot find this information in the handbook."
Cite sources (e.g. [Source 1]) and mention page numbers where possible.

CONTEXT:
{context}

STUDENT QUESTION: {question}

ANSWER:"""

    last_error = None
    backoff = 5  # seconds

    for attempt in range(3):   # initial attempt + 2 retries
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                ),
            )
            sources = [
                {
                    "page": c.get("start_page", "?"),
                    "section": c.get("section_title", "Unknown"),
                    "preview": c["text"][:150] + "...",
                }
                for c in retrieved_chunks
            ]
            return {"answer": response.text, "sources": sources, "method": "gemini"}

        except Exception as e:
            last_error = e
            if _is_quota_exhausted(e):
                break           # hard limit — don't retry
            if attempt < 2:
                time.sleep(backoff)
                backoff *= 2    # 5s → 10s

    raise last_error


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified interface
# ═══════════════════════════════════════════════════════════════════════════════

def generate_answer(question: str, retrieved_chunks: list, api_key: str = "", use_llm: bool = True) -> dict:
    """
    Try LLM answer first; fall back to extractive on any failure.

    Returns dict with keys: 'answer', 'sources', 'method', optionally 'llm_error'.
    """
    if use_llm and (api_key or GEMINI_API_KEY):
        try:
            return generate_llm_answer(question, retrieved_chunks, api_key)
        except Exception as e:
            result = generate_extractive_answer(question, retrieved_chunks)
            result["llm_error"] = _friendly_error(e)
            return result

    return generate_extractive_answer(question, retrieved_chunks)
