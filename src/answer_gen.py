"""
Answer Generation Module.

Supports two modes:
1. LLM-based (Google Gemini API via google-genai) — rich, natural-language answers
2. Extractive fallback — highlights the most relevant sentences from retrieved chunks
   (works without any API key)
"""
import re

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from config import GEMINI_API_KEY, GEMINI_MODEL


# ═══════════════════════════════════════════════════════════════════════════════
#  Extractive Answer (no API needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_sentence(sentence: str, query_words: set[str]) -> float:
    """Score a sentence by how many query words it contains (normalized)."""
    sent_words = set(sentence.lower().split())
    if not sent_words:
        return 0.0
    overlap = len(query_words & sent_words)
    return overlap / len(query_words) if query_words else 0.0


def generate_extractive_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Extract the most relevant sentences from the retrieved chunks
    that best answer the question. No external API required.

    Returns dict with 'answer', 'sources', 'method'.
    """
    query_words = set(question.lower().split())
    # Remove common stopwords for better matching
    stopwords = {"what", "is", "the", "a", "an", "how", "can", "do", "does",
                 "are", "of", "in", "for", "to", "and", "or", "if", "by", "at"}
    query_words -= stopwords

    # Score every sentence across all chunks
    scored_sentences = []
    for chunk in retrieved_chunks:
        text = chunk["text"]
        # Split into sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:  # skip very short fragments
                continue
            score = _score_sentence(sent, query_words)
            if score > 0:
                scored_sentences.append((sent, score, chunk))

    # Sort by score descending, take top 5 sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = scored_sentences[:5]

    if not top_sentences:
        answer = "No directly relevant sentences found. Please review the retrieved chunks below for the answer."
    else:
        answer_parts = []
        for i, (sent, score, chunk) in enumerate(top_sentences):
            page = chunk.get("start_page", "?")
            section = chunk.get("section_title", "")
            ref = f"(Page {page}" + (f", {section}" if section else "") + ")"
            answer_parts.append(f"- {sent} {ref}")
        answer = "**Relevant excerpts from the handbook:**\n\n" + "\n\n".join(answer_parts)

    # Build sources list
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

    return {
        "answer": answer,
        "sources": sources,
        "method": "extractive",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM-based Answer (Google Gemini via google-genai)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_genai_client(api_key: str = ""):
    """Create a google.genai client."""
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai package is not installed. Run: pip install google-genai")
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Set the environment variable or pass it directly."
        )
    return genai.Client(api_key=key)


def generate_llm_answer(question: str, retrieved_chunks: list[dict], api_key: str = "") -> dict:
    """
    Generate an answer using Gemini (google-genai), grounded in the retrieved chunks.

    Returns dict with 'answer', 'sources', 'method'.
    """
    client = _get_genai_client(api_key)

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        section = chunk.get("section_title", "Unknown")
        page = chunk.get("start_page", "?")
        context_parts.append(
            f"[Source {i+1} | Page {page} | {section}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an academic advisor for NUST (National University of Sciences and Technology).
Answer the student's question using ONLY the information provided in the context below.
If the answer is not in the context, say "I cannot find this information in the handbook."

Rules:
1. Be accurate and cite which source(s) you used (e.g., [Source 1], [Source 2]).
2. Be concise but complete.
3. If there are specific rules, quote them directly.
4. Mention the relevant page number(s) when possible.

CONTEXT:
{context}

STUDENT QUESTION: {question}

ANSWER:"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    # Build sources list
    sources = []
    for chunk in retrieved_chunks:
        sources.append({
            "page": chunk.get("start_page", "?"),
            "section": chunk.get("section_title", "Unknown"),
            "preview": chunk["text"][:150] + "...",
        })

    return {
        "answer": response.text,
        "sources": sources,
        "method": "gemini",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified interface
# ═══════════════════════════════════════════════════════════════════════════════

def generate_answer(question: str, retrieved_chunks: list[dict], api_key: str = "", use_llm: bool = True) -> dict:
    """
    Generate an answer — tries LLM first, falls back to extractive if LLM fails.

    Args:
        question: user's question
        retrieved_chunks: list of chunk dicts
        api_key: optional Gemini API key
        use_llm: whether to attempt LLM generation

    Returns:
        dict with 'answer', 'sources', 'method'
    """
    if use_llm and (api_key or GEMINI_API_KEY):
        try:
            return generate_llm_answer(question, retrieved_chunks, api_key)
        except Exception as e:
            # Fall back to extractive with a note about the error
            result = generate_extractive_answer(question, retrieved_chunks)
            result["llm_error"] = str(e)
            return result

    return generate_extractive_answer(question, retrieved_chunks)
