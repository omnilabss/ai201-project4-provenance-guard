from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from dotenv import load_dotenv
import sqlite3
import uuid
import json
import os
import re
import math
import statistics
from datetime import datetime, timezone

load_dotenv()

app = Flask(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_log.db")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db():
    """Create the audit_log table if it does not already exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id        TEXT NOT NULL UNIQUE,
            creator_id        TEXT NOT NULL,
            timestamp         TEXT NOT NULL,
            attribution       TEXT NOT NULL,
            confidence        REAL NOT NULL,
            llm_score         REAL NOT NULL,
            stylometric_score REAL NOT NULL,
            status            TEXT NOT NULL,
            appeal_reasoning  TEXT,
            appeal_timestamp  TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def db_insert(content_id, creator_id, attribution, confidence, llm_score,
              stylometric_score):
    """Insert a new classified submission into the audit log."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO audit_log
            (content_id, creator_id, timestamp, attribution, confidence,
             llm_score, stylometric_score, status, appeal_reasoning, appeal_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'classified', NULL, NULL)
        """,
        (
            content_id,
            creator_id,
            datetime.now(timezone.utc).isoformat(),
            attribution,
            confidence,
            llm_score,
            stylometric_score,
        ),
    )
    conn.commit()
    conn.close()


def db_get(content_id):
    """Fetch one row by content_id; returns dict or None."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM audit_log WHERE content_id = ? LIMIT 1", (content_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def db_appeal(content_id, reasoning):
    """Update an existing audit log row with appeal information."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        UPDATE audit_log
        SET status = 'under_review',
            appeal_reasoning = ?,
            appeal_timestamp = ?
        WHERE content_id = ?
        """,
        (reasoning, datetime.now(timezone.utc).isoformat(), content_id),
    )
    conn.commit()
    conn.close()


def db_recent(limit=50):
    """Return the most recent audit log entries as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Detection Signal 1 — LLM via Groq
# ---------------------------------------------------------------------------

def llm_signal(text: str) -> float:
    """
    Call Groq's llama-3.3-70b-versatile to assess whether text is AI-generated.

    Returns a float 0.0–1.0 where 1.0 = definitely AI-generated.
    Falls back to 0.5 on any error or parse failure.
    """
    system_prompt = (
        "You are a forensic text analyst specializing in distinguishing AI-generated "
        "content from human-written content. Analyze the provided text carefully and "
        "assess the probability that it was produced by an AI writing tool.\n\n"
        "Consider: formulaic transitions, unnaturally uniform sentence lengths, absence "
        "of personal voice or lived experience, hedged non-committal phrasing, "
        "repetitive vocabulary, and overly structured organization.\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"ai_probability": 0.82, "reasoning": "Brief explanation here."}\n\n'
        "Where ai_probability is a float from 0.0 (definitely human) to 1.0 (definitely AI). "
        "Do not include any text outside of the JSON object."
    )

    user_message = f"Analyze this text for AI authorship:\n\n{text}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON — handle cases where the model wraps it in markdown fences
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = float(data.get("ai_probability", 0.5))
            return max(0.0, min(1.0, score))

        # Attempt to parse the entire response as JSON
        data = json.loads(raw)
        score = float(data.get("ai_probability", 0.5))
        return max(0.0, min(1.0, score))

    except Exception:
        # Any failure — API error, parse error, key error — returns neutral 0.5
        return 0.5


# ---------------------------------------------------------------------------
# Detection Signal 2 — Stylometric Heuristics (pure Python)
# ---------------------------------------------------------------------------

def stylometric_signal(text: str) -> dict:
    """
    Compute four stylometric heuristics and combine them into a single score.

    Returns a dict with keys:
        score                           - combined stylometric score (0.0–1.0)
        sentence_length_variance_score
        type_token_ratio_score
        punctuation_density_score
        avg_sentence_length_score
    """
    # Tokenize sentences and words
    raw_sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())

    if not sentences or not words:
        return {
            "score": 0.5,
            "sentence_length_variance_score": 0.5,
            "type_token_ratio_score": 0.5,
            "punctuation_density_score": 0.5,
            "avg_sentence_length_score": 0.5,
        }

    sentence_word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences]

    # --- Metric 1: sentence_length_variance ---
    # Lower std dev → more uniform → more AI-like → higher score
    # AI texts have std dev < 3, human texts often have std dev > 5
    # Normalize: score = max(0, 1 - std_dev / 8)
    if len(sentence_word_counts) >= 2:
        try:
            std_dev = statistics.stdev(sentence_word_counts)
        except statistics.StatisticsError:
            std_dev = 0.0
    else:
        std_dev = 0.0

    variance_score = max(0.0, 1.0 - std_dev / 8.0)

    # --- Metric 2: type_token_ratio ---
    # Lower TTR → less vocabulary diversity → more AI-like → higher score
    # AI texts ~0.4–0.55, human ~0.55–0.75
    # Score = max(0, 1 - (ttr - 0.3) / 0.5)
    ttr = len(set(words)) / len(words)
    ttr_score = max(0.0, min(1.0, 1.0 - (ttr - 0.3) / 0.5))

    # --- Metric 3: punctuation_density ---
    # AI texts have more uniform, lower punctuation density
    # Count non-alphanumeric/space chars / total chars
    # Score = max(0, 1 - (density - 0.03) / 0.12)
    total_chars = len(text) if text else 1
    punct_chars = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    density = punct_chars / total_chars
    punct_score = max(0.0, min(1.0, 1.0 - (density - 0.03) / 0.12))

    # --- Metric 4: avg_sentence_length ---
    # AI tends to write longer, more uniform sentences
    # Score = min(1, avg_len / 25)
    avg_len = sum(sentence_word_counts) / len(sentence_word_counts)
    len_score = min(1.0, avg_len / 25.0)

    # Combine with equal weights
    combined = (variance_score + ttr_score + punct_score + len_score) / 4.0

    return {
        "score": round(combined, 4),
        "sentence_length_variance_score": round(variance_score, 4),
        "type_token_ratio_score": round(ttr_score, 4),
        "punctuation_density_score": round(punct_score, 4),
        "avg_sentence_length_score": round(len_score, 4),
    }


# ---------------------------------------------------------------------------
# Confidence Scoring
# ---------------------------------------------------------------------------

def compute_confidence(llm_score: float, stylometric_score: float) -> float:
    """
    Combine LLM and stylometric signals into a single confidence score.

    Weighted combination: llm_weight=0.65, stylometric_weight=0.35.
    If the two signals DISAGREE strongly (|llm - stylometric| > 0.4),
    reduce confidence by pulling toward 0.5:
        combined = 0.5 + (combined - 0.5) * 0.6

    Returns a float in [0, 1] representing probability of AI authorship.
    """
    combined = llm_score * 0.65 + stylometric_score * 0.35

    if abs(llm_score - stylometric_score) > 0.4:
        combined = 0.5 + (combined - 0.5) * 0.6

    return round(max(0.0, min(1.0, combined)), 4)


# ---------------------------------------------------------------------------
# Attribution and Label Generation
# ---------------------------------------------------------------------------

def get_attribution(confidence: float) -> str:
    """Map confidence score to an attribution string."""
    if confidence > 0.70:
        return "likely_ai"
    elif confidence < 0.35:
        return "likely_human"
    else:
        return "uncertain"


def generate_label(confidence: float, attribution: str) -> str:
    """Generate a human-readable transparency label for the given confidence."""
    pct = round(confidence * 100)
    human_pct = round((1.0 - confidence) * 100)

    if attribution == "likely_ai":
        return (
            f"⚠️ Likely AI-Generated — Our analysis found strong indicators that this "
            f"content was produced by an AI writing tool. Confidence: {pct}%. "
            f"The author may appeal this classification if they believe it is incorrect."
        )
    elif attribution == "likely_human":
        return (
            f"✓ Likely Human-Written — Our analysis found strong indicators that this "
            f"content was written by a person. Confidence: {human_pct}% human. "
            f"No action required."
        )
    else:
        return (
            f"◐ Attribution Uncertain — Our analysis could not determine with confidence "
            f"whether this content was written by a person or generated by AI. "
            f"Confidence: {pct}% AI indicators. "
            f"Authors may appeal if this label is inaccurate."
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    """
    POST /submit
    Body: { "text": "...", "creator_id": "..." }
    Returns: { content_id, attribution, confidence, signal_scores, label, status }
    """
    data = request.get_json(silent=True) or {}

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing required field: text"}), 400

    creator_id = data.get("creator_id", "").strip()
    if not creator_id:
        return jsonify({"error": "Missing required field: creator_id"}), 400

    # Run both detection signals
    llm_score = llm_signal(text)
    stylo_result = stylometric_signal(text)
    stylometric_score = stylo_result["score"]

    # Combine into a single confidence score
    confidence = compute_confidence(llm_score, stylometric_score)

    # Determine attribution and generate label
    attribution = get_attribution(confidence)
    label = generate_label(confidence, attribution)

    # Generate a unique content ID
    content_id = str(uuid.uuid4())

    # Persist to audit log
    db_insert(
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        confidence=confidence,
        llm_score=llm_score,
        stylometric_score=stylometric_score,
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "confidence": confidence,
            "signal_scores": {
                "llm_score": round(llm_score, 4),
                "stylometric_score": stylometric_score,
                "stylometric_breakdown": {
                    "sentence_length_variance_score": stylo_result[
                        "sentence_length_variance_score"
                    ],
                    "type_token_ratio_score": stylo_result["type_token_ratio_score"],
                    "punctuation_density_score": stylo_result[
                        "punctuation_density_score"
                    ],
                    "avg_sentence_length_score": stylo_result[
                        "avg_sentence_length_score"
                    ],
                },
            },
            "label": label,
            "status": "classified",
        }
    ), 200


@app.route("/appeal", methods=["POST"])
@limiter.limit("20 per minute;200 per day")
def appeal():
    """
    POST /appeal
    Body: { "content_id": "...", "creator_reasoning": "..." }
    Returns: { message, content_id, status }
    """
    data = request.get_json(silent=True) or {}

    content_id = data.get("content_id", "").strip()
    if not content_id:
        return jsonify({"error": "Missing required field: content_id"}), 400

    creator_reasoning = data.get("creator_reasoning", "").strip()
    if not creator_reasoning:
        return jsonify({"error": "Missing required field: creator_reasoning"}), 400

    row = db_get(content_id)
    if row is None:
        return jsonify({"error": f"content_id not found: {content_id}"}), 404

    db_appeal(content_id, creator_reasoning)

    return jsonify(
        {
            "message": "Appeal received and is under review.",
            "content_id": content_id,
            "status": "under_review",
        }
    ), 200


@app.route("/log", methods=["GET"])
def get_log():
    """
    GET /log
    Returns the last 50 audit log entries as a JSON array, newest first.
    """
    entries = db_recent(limit=50)
    return jsonify(entries), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(429)
def rate_limit_handler(e):
    return jsonify(
        {
            "error": "Rate limit exceeded. Please wait before submitting again.",
            "retry_after": str(e.description),
        }
    ), 429


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------

init_db()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
