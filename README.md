# Provenance Guard

A Flask-based backend service for AI content attribution analysis. Any creative-sharing platform can plug in Provenance Guard to classify submitted text, score confidence, surface a transparency label, and handle creator appeals.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Endpoints](#endpoints)
3. [Detection Signals](#detection-signals)
4. [Confidence Scoring](#confidence-scoring)
5. [Transparency Labels](#transparency-labels)
6. [Rate Limiting](#rate-limiting)
7. [Audit Log](#audit-log)
8. [Appeals Workflow](#appeals-workflow)
9. [Known Limitations](#known-limitations)
10. [Spec Reflection](#spec-reflection)
11. [AI Usage](#ai-usage)
12. [Setup & Running](#setup--running)

---

## Architecture Overview

A submitted piece of text takes the following path through the system:

1. **POST /submit** — The client sends `{ text, creator_id }` to the submission endpoint.
2. **Rate Limiter** — Flask-Limiter checks whether the caller has exceeded the per-minute or per-day quota. Requests over the limit receive HTTP 429.
3. **Signal 1 — LLM Classification** — The text is sent to `llama-3.3-70b-versatile` via the Groq API. The model is prompted to return a single JSON object `{ "ai_probability": float }` representing the probability that the content was AI-generated.
4. **Signal 2 — Stylometric Heuristics** — Four surface-level linguistic features are computed in pure Python: sentence-length standard deviation, type-token ratio, punctuation density, and average sentence length. These are combined into a single stylometric score.
5. **Confidence Combiner** — The two scores are merged using a weighted average (LLM: 65%, stylometric: 35%). If the signals disagree by more than 0.4, the combined score is dampened toward 0.5 to reflect genuine uncertainty.
6. **Attribution + Label Generator** — The confidence score is mapped to one of three attributions (`likely_ai`, `uncertain`, `likely_human`) and a corresponding human-readable transparency label.
7. **SQLite Audit Log** — Every decision is written to `audit_log.db` with timestamp, signal scores, confidence, and status.
8. **JSON Response** — The caller receives `content_id`, `attribution`, `confidence`, `signal_scores`, `label`, and `status`.

For appeals:

1. **POST /appeal** — The client sends `{ content_id, creator_reasoning }`.
2. The system validates the `content_id` exists in the audit log.
3. The row is updated: `status → "under_review"`, `appeal_reasoning` and `appeal_timestamp` are set.
4. A human reviewer uses **GET /log** to inspect queued appeals.

See [planning.md](planning.md) for the full architecture diagrams and design decisions.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/submit` | Submit text for attribution analysis |
| `POST` | `/appeal` | Appeal a classification |
| `GET` | `/log` | Return the last 50 audit log entries |

### POST /submit

**Request body:**
```json
{
  "text": "The piece of creative writing to analyze.",
  "creator_id": "unique-creator-identifier"
}
```

**Response:**
```json
{
  "content_id": "3f7a2b1e-...",
  "attribution": "likely_ai",
  "confidence": 0.78,
  "signal_scores": {
    "llm_score": 0.81,
    "stylometric_score": 0.71,
    "stylometric_breakdown": {
      "sentence_length_variance_score": 0.65,
      "type_token_ratio_score": 0.72,
      "punctuation_density_score": 0.80,
      "avg_sentence_length_score": 0.68
    }
  },
  "label": "⚠️ Likely AI-Generated — Our analysis found strong indicators...",
  "status": "classified"
}
```

### POST /appeal

**Request body:**
```json
{
  "content_id": "3f7a2b1e-...",
  "creator_reasoning": "I wrote this myself. I can provide my original draft notes."
}
```

**Response:**
```json
{
  "message": "Appeal received and is under review.",
  "content_id": "3f7a2b1e-...",
  "status": "under_review"
}
```

---

## Detection Signals

### Signal 1 — LLM-Based Classification (Groq)

**What it measures:** Semantic and stylistic coherence holistically — the kinds of signals a human expert would use to identify AI writing:
- Formulaic structure and transitions (e.g., "it is important to note," "in conclusion")
- Absence of lived experience, personal anecdote, or genuine opinion
- Unnaturally consistent paragraph and sentence lengths
- Hedged, non-committal phrasing typical of aligned AI models
- Repetitive vocabulary and overly structured organization

**Why it differs between human and AI writing:** LLMs trained on RLHF tend to produce text that is accurate, safe, and readable — but at the cost of the idiosyncrasy and voice that characterizes human writing. Human writers make unconventional choices, break rules, contradict themselves, and embed personal reference points that LLMs rarely generate unprompted.

**Output format:** A float `llm_score` in [0.0, 1.0], where 1.0 means the model is highly confident the text is AI-generated. Extracted from the `ai_probability` field of the model's JSON response.

**Weight:** 0.65 (primary signal — stronger contextual understanding)

**Blind spots:** Very short texts (< 50 words) give the model insufficient signal. The model may also be fooled by human writers who deliberately write in an AI style, or by AI text that has been heavily edited. Academic and legal prose can be mistaken for AI writing because its formal register resembles AI output.

---

### Signal 2 — Stylometric Heuristics (Pure Python)

**What it measures:** Four statistical properties of text that tend to differ systematically between human and AI writing:

| Metric | Human Range | AI Range | AI-score Formula |
|--------|-------------|----------|------------------|
| Sentence length std dev | > 5 words | < 3 words | `max(0, 1 − std_dev / 8)` |
| Type-token ratio (TTR) | 0.55 – 0.75 | 0.40 – 0.55 | `max(0, 1 − (ttr − 0.3) / 0.5)` |
| Punctuation density | 0.08 – 0.14 | 0.05 – 0.08 | `max(0, 1 − (density − 0.03) / 0.12)` |
| Avg sentence length | Varies | Long, uniform | `min(1, avg_len / 25)` |

Each metric is normalized to [0, 1] (higher = more AI-like), then averaged with equal weights.

**Why these differ:** AI-generated text exhibits more uniformity at the structural level. Sentence lengths cluster around a comfortable middle range; vocabulary repeats across paragraphs; punctuation tends to be sparse and grammatically standard. Human writing is more erratic — short punchy sentences interrupted by long compound ones, colloquial contractions, em-dashes, parenthetical asides, and idiosyncratic word choices.

**Output format:** A dict with a combined `score` float and four individual sub-scores. Only `score` feeds into the confidence combiner.

**Weight:** 0.35 (secondary signal — fast, reliable, but surface-level)

**Blind spots:** Formal human writing (academic papers, legal documents, technical reports) can score as AI-like because its structural regularity mimics AI output. Short texts produce statistically unreliable variance calculations. Non-English text invalidates the English-language calibration of the scoring formulas.

---

## Confidence Scoring

### Combination Formula

```
combined = llm_score × 0.65 + stylometric_score × 0.35
```

If the two signals **disagree strongly** (|llm_score − stylometric_score| > 0.4), the combined score is dampened toward 0.5:

```
combined = 0.5 + (combined − 0.5) × 0.6
```

This reduces overconfidence when signals conflict — a strong indicator the text is genuinely ambiguous.

### Threshold Mapping

| Confidence | Attribution | Reasoning |
|------------|-------------|-----------|
| > 0.70 | `likely_ai` | Both signals agree on strong AI indicators |
| 0.35 – 0.70 | `uncertain` | Signals are ambiguous, conflicting, or borderline |
| < 0.35 | `likely_human` | Strong evidence of human authorship |

The uncertain band (0.35–0.70) is intentionally wide. **A false positive — labeling a human writer's work as AI — is considered a more serious error than returning "uncertain."** The system is designed to err toward acknowledging uncertainty rather than over-claiming confidence it cannot support.

### Example Submissions with Different Confidence Scores

**High-confidence AI (confidence: 0.79)**

Input:
> "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment."

Result:
```json
{
  "attribution": "likely_ai",
  "confidence": 0.79,
  "signal_scores": {
    "llm_score": 0.88,
    "stylometric_score": 0.61
  },
  "label": "⚠️ Likely AI-Generated — Our analysis found strong indicators that this content was produced by an AI writing tool. Confidence: 79%. The author may appeal this classification if they believe it is incorrect."
}
```

**Low-confidence / likely human (confidence: 0.22)**

Input:
> "ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it and i was thirsty for like three hours after. my friend got the spicy version and said it was better. probably won't go back unless someone drags me there"

Result:
```json
{
  "attribution": "likely_human",
  "confidence": 0.22,
  "signal_scores": {
    "llm_score": 0.12,
    "stylometric_score": 0.39
  },
  "label": "✓ Likely Human-Written — Our analysis found strong indicators that this content was written by a person. Confidence: 78% human. No action required."
}
```

The 0.57-point gap between these two examples (0.22 vs 0.79) demonstrates that the scoring produces **meaningful variation** across clearly different inputs, not a binary flip at 0.5.

---

## Transparency Labels

The label returned by `/submit` changes based on the confidence score. All three variants are reachable:

### Variant 1 — High-confidence AI (`confidence > 0.70`)

> ⚠️ Likely AI-Generated — Our analysis found strong indicators that this content was produced by an AI writing tool. Confidence: {pct}%. The author may appeal this classification if they believe it is incorrect.

**Example (pct = 79):**
> ⚠️ Likely AI-Generated — Our analysis found strong indicators that this content was produced by an AI writing tool. Confidence: 79%. The author may appeal this classification if they believe it is incorrect.

---

### Variant 2 — High-confidence Human (`confidence < 0.35`)

> ✓ Likely Human-Written — Our analysis found strong indicators that this content was written by a person. Confidence: {human_pct}% human. No action required.

**Example (human_pct = 78):**
> ✓ Likely Human-Written — Our analysis found strong indicators that this content was written by a person. Confidence: 78% human. No action required.

---

### Variant 3 — Uncertain (`0.35 ≤ confidence ≤ 0.70`)

> ◐ Attribution Uncertain — Our analysis could not determine with confidence whether this content was written by a person or generated by AI. Confidence: {pct}% AI indicators. Authors may appeal if this label is inaccurate.

**Example (pct = 53):**
> ◐ Attribution Uncertain — Our analysis could not determine with confidence whether this content was written by a person or generated by AI. Confidence: 53% AI indicators. Authors may appeal if this label is inaccurate.

---

The labels are designed with three goals:
1. **Be honest about uncertainty** — the label text and confidence percentage are always shown together.
2. **Make the appeal path visible** — both the AI and uncertain labels explicitly invite the author to appeal.
3. **Avoid punitive language** — the labels describe system findings, not accusations. The word "detected" is avoided; the system "found indicators."

---

## Rate Limiting

Flask-Limiter is applied to the submission endpoint:

```
10 requests per minute
100 requests per day
```

**Reasoning for these limits:**

- **10 per minute:** A legitimate human creator submitting their own work would rarely submit more than a few pieces in any given minute. Even a prolific writer revising and re-submitting a single poem would need 10 attempts per minute to hit this limit — an unlikely real-world scenario. This limit still allows bursts (e.g., a developer testing the API), while stopping a script that floods submissions at machine speed.

- **100 per day:** A creative platform user might submit 5–20 pieces of work in a day during an active session. 100 per day provides generous headroom for power users while making it uneconomical to probe the system's detection boundary by submitting thousands of variants daily.

- **The asymmetry is intentional:** The per-day limit is more important than the per-minute one. An adversary probing the system will try to evade the per-minute limit by spacing requests. The per-day cap ensures that even with pacing, systematic evasion attempts hit a hard ceiling.

**Rate limit evidence (HTTP status codes from 12 rapid sequential submissions):**

```
200
200
200
200
200
200
200
200
200
200
429
429
```

The first 10 requests succeed (HTTP 200). Requests 11 and 12 receive HTTP 429 (Too Many Requests).

**429 response body:**
```json
{
  "error": "Rate limit exceeded. Please wait before submitting again.",
  "retry_after": "10 per 1 minute"
}
```

---

## Audit Log

Every attribution decision is written to `audit_log.db` (SQLite). The log is accessible via `GET /log`.

**Schema:**
```
content_id        — UUID for the submission
creator_id        — Caller-supplied creator identifier  
timestamp         — UTC ISO 8601 timestamp of classification
attribution       — "likely_ai" | "uncertain" | "likely_human"
confidence        — Combined confidence score (float, 0–1)
llm_score         — Raw LLM signal score (float, 0–1)
stylometric_score — Raw stylometric signal score (float, 0–1)
status            — "classified" | "under_review"
appeal_reasoning  — Creator's reasoning (null until appealed)
appeal_timestamp  — UTC timestamp of appeal (null until appealed)
```

**Sample log entries (from GET /log):**

```json
[
  {
    "id": 3,
    "content_id": "c9d4e5f6-...",
    "creator_id": "test-user-3",
    "timestamp": "2025-06-15T18:47:22.001Z",
    "attribution": "uncertain",
    "confidence": 0.53,
    "llm_score": 0.61,
    "stylometric_score": 0.37,
    "status": "under_review",
    "appeal_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical.",
    "appeal_timestamp": "2025-06-15T18:48:55.302Z"
  },
  {
    "id": 2,
    "content_id": "b8c3d4e5-...",
    "creator_id": "test-user-2",
    "timestamp": "2025-06-15T18:45:01.441Z",
    "attribution": "likely_human",
    "confidence": 0.22,
    "llm_score": 0.12,
    "stylometric_score": 0.39,
    "status": "classified",
    "appeal_reasoning": null,
    "appeal_timestamp": null
  },
  {
    "id": 1,
    "content_id": "a7b2c3d4-...",
    "creator_id": "test-user-1",
    "timestamp": "2025-06-15T18:43:12.887Z",
    "attribution": "likely_ai",
    "confidence": 0.79,
    "llm_score": 0.88,
    "stylometric_score": 0.61,
    "status": "classified",
    "appeal_reasoning": null,
    "appeal_timestamp": null
  }
]
```

---

## Appeals Workflow

1. A creator receives a label they believe is inaccurate.
2. They call `POST /appeal` with their `content_id` and a `creator_reasoning` string.
3. The system validates that the `content_id` exists (returns 404 if not found).
4. The submission's `status` is updated from `"classified"` to `"under_review"` and the reasoning and timestamp are logged.
5. A human reviewer uses `GET /log` and filters by `status = "under_review"` to see the appeal queue.
6. The reviewer can examine `llm_score`, `stylometric_score`, `confidence`, and `appeal_reasoning` to make a final determination.

**Test command:**
```bash
curl -s -X POST http://localhost:5001/appeal \
  -H "Content-Type: application/json" \
  -d '{"content_id": "PASTE-CONTENT-ID-HERE", "creator_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical."}' | python3 -m json.tool
```

**Expected response:**
```json
{
  "content_id": "...",
  "message": "Appeal received and is under review.",
  "status": "under_review"
}
```

---

## Known Limitations

### 1. Formal Human Writing Gets False-Positived

The most predictable failure mode: **academic writing, legal documents, and technical reports** will frequently score as AI-generated. These genres have the same structural regularities that make AI output recognizable — long, uniform sentences, high vocabulary precision, low punctuation density, formulaic organization. Both signals struggle here. The stylometric heuristics are calibrated on informal-to-moderate formal human writing; academic prose sits outside that calibration range. The LLM signal is better, but a well-structured academic argument looks a lot like what a prompted LLM would produce.

**Impact:** A professor submitting her own lecture notes might receive a `likely_ai` label. This is why the uncertain band is kept wide (0.35–0.70) and why the appeals workflow is a first-class feature.

### 2. Short Texts (< 50 words) Are Statistically Unreliable

Sentence-length variance requires multiple sentences to be meaningful. A single sentence or haiku gives the stylometric signal almost no data — a single long sentence collapses the std dev to 0, pushing the variance score toward AI-like regardless of origin. The system partially mitigates this by defaulting the stylometric signal to 0.5 when statistics are unavailable, but short texts will still produce noisy results. Platform operators should consider requiring a minimum character count before submission.

---

## Spec Reflection

### One way the spec helped guide implementation

The planning.md's explicit requirement to write out the three label variants *before* writing any code forced a design decision that shaped the implementation: the label text had to communicate both the verdict **and** the confidence level in plain language. Writing the labels first revealed that showing a raw probability number (e.g., "0.78") was meaningless to a non-technical reader, which led to converting it to a percentage and presenting it inline as "Confidence: 78%". That decision then propagated through the generate_label() function naturally, rather than being retrofitted later.

### One way implementation diverged from the spec

The original spec sketched equal weights for all four stylometric metrics. During implementation testing, the formulas for TTR and punctuation density occasionally produced scores near 0 even for obviously AI-generated text — especially for shorter samples with limited unique vocabulary. This revealed a calibration issue: the normalization ranges were borrowed from literature without adjustment for the text lengths seen in practice. Rather than adding per-metric weights (which would have required more empirical data to justify), the disagreement-dampening mechanism was relied on more heavily — when the stylometric and LLM signals diverge substantially, the confidence is pulled toward 0.5 regardless of the raw scores. This was a pragmatic divergence: the spec didn't include the dampening formula; it was added during implementation when test cases showed overconfidence at signal boundaries.

---

## AI Usage

### Instance 1 — System Prompt Engineering for the LLM Signal

I directed Claude to draft and iterate on the system prompt for the Groq LLM signal. The initial draft asked the model to "assess whether text is AI-generated" and return a score. The model frequently returned prose explanations instead of JSON. Claude suggested restructuring the prompt to: (a) give the model a clear expert persona ("forensic text analyst"), (b) enumerate specific features to look for (formulaic transitions, absent personal voice, etc.), and (c) add an explicit instruction: "Do not include any text outside of the JSON object." I overrode one of Claude's suggestions: it proposed using `response_format={"type": "json_object"}` on the Groq API call, but that parameter isn't consistently supported across all Groq models, so I kept the regex-based JSON extraction as a fallback instead.

### Instance 2 — Disagreement-Dampening Formula

During Milestone 4 testing, I observed that when the LLM returned 0.9 (very confident AI) but the stylometric signal returned 0.4 (uncertain), the combined score was 0.72 — still `likely_ai`, but the signals clearly disagreed. I asked Claude whether a disagreement penalty was a reasonable approach. Claude suggested the formula `combined = 0.5 + (combined − 0.5) × factor` with a damping factor of 0.5–0.7, and explained that pulling toward 0.5 when signals conflict is a Bayesian-intuitive choice: conflicting evidence reduces posterior confidence. I adjusted the threshold from Claude's suggested 0.35 to 0.4 (a bit more permissive of minor disagreement) and chose a damping factor of 0.6 rather than the suggested 0.5, after testing showed 0.5 was too aggressive — it pushed borderline cases that were genuinely AI-generated into the uncertain band.

---

## Setup & Running

### Requirements

- Python 3.11+
- A [Groq API key](https://console.groq.com/) (free tier)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai201-project4-provenance-guard.git
cd ai201-project4-provenance-guard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the repo root:

```
GROQ_API_KEY=your_key_here
```

### Running

```bash
python3 app.py
# Server starts on http://localhost:5001
```

### Testing

**Submit text for analysis:**
```bash
curl -s -X POST http://localhost:5001/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "The sun dipped below the horizon, painting the sky in hues of amber and rose. I sat on the porch, coffee in hand, watching the neighborhood slowly go quiet.", "creator_id": "test-user-1"}' | python3 -m json.tool
```

**View audit log:**
```bash
curl -s http://localhost:5001/log | python3 -m json.tool
```

**Test rate limiting (12 rapid requests — expects 429 after 10th):**
```bash
for i in $(seq 1 12); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5001/submit \
    -H "Content-Type: application/json" \
    -d '{"text": "Test submission.", "creator_id": "ratelimit-test"}'
done
```