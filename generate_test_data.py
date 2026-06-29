#!/usr/bin/env python3
"""
generate_test_data.py

Run this AFTER starting app.py to populate the audit log with test entries
and capture rate-limit evidence for the README.

Usage:
    python3 app.py &
    python3 generate_test_data.py
"""

import json
import time
import urllib.request
import urllib.error

BASE = "http://localhost:5001"


def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(path):
    req = urllib.request.Request(BASE + path, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ─── Test inputs ─────────────────────────────────────────────────────────────

TESTS = [
    {
        "label": "Clearly AI-generated",
        "creator_id": "test-user-1",
        "text": (
            "Artificial intelligence represents a transformative paradigm shift "
            "in modern society. It is important to note that while the benefits "
            "of AI are numerous, it is equally essential to consider the ethical "
            "implications. Furthermore, stakeholders across various sectors must "
            "collaborate to ensure responsible deployment."
        ),
    },
    {
        "label": "Clearly human-written (casual)",
        "creator_id": "test-user-2",
        "text": (
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium "
            "in it and i was thirsty for like three hours after. my friend got "
            "the spicy version and said it was better. probably won't go back "
            "unless someone drags me there"
        ),
    },
    {
        "label": "Borderline: formal human writing",
        "creator_id": "test-user-3",
        "text": (
            "The relationship between monetary policy and asset price inflation "
            "has been extensively studied in the literature. Central banks face "
            "a fundamental tension between their mandate for price stability and "
            "the unintended consequences of prolonged low interest rates on "
            "equity and real estate valuations."
        ),
    },
    {
        "label": "Borderline: lightly edited AI output",
        "creator_id": "test-user-4",
        "text": (
            "I've been thinking a lot about remote work lately. There are genuine "
            "tradeoffs — flexibility and no commute on one side, isolation and "
            "blurred work-life boundaries on the other. Studies show productivity "
            "varies widely by individual and role type."
        ),
    },
]


def main():
    print("=" * 60)
    print("Provenance Guard — Test Data Generator")
    print("=" * 60)

    content_ids = []

    for t in TESTS:
        print(f"\n[SUBMIT] {t['label']}")
        status, resp = post("/submit", {"text": t["text"], "creator_id": t["creator_id"]})
        print(f"  HTTP {status}")
        if status == 200:
            cid = resp["content_id"]
            content_ids.append(cid)
            print(f"  content_id : {cid}")
            print(f"  attribution: {resp['attribution']}")
            print(f"  confidence : {resp['confidence']}")
            print(f"  llm_score  : {resp['signal_scores']['llm_score']}")
            print(f"  stylo_score: {resp['signal_scores']['stylometric_score']}")
            print(f"  label      : {resp['label'][:80]}...")
        else:
            print(f"  ERROR: {resp}")
        time.sleep(1)  # be polite to the Groq API

    # ── File an appeal on the first submission ────────────────────────────────
    if content_ids:
        print(f"\n[APPEAL] Filing appeal on content_id={content_ids[0]}")
        status, resp = post(
            "/appeal",
            {
                "content_id": content_ids[0],
                "creator_reasoning": (
                    "I wrote this myself from personal experience. "
                    "I am a non-native English speaker and my writing style "
                    "may appear more formal than typical."
                ),
            },
        )
        print(f"  HTTP {status}: {resp}")

    # ── Fetch and print audit log ─────────────────────────────────────────────
    print("\n[LOG] Current audit log (newest first):")
    entries = get("/log")
    print(json.dumps(entries, indent=2))

    # ── Rate limit test ───────────────────────────────────────────────────────
    print("\n[RATE LIMIT] Sending 12 rapid submissions (expect 429 after 10th)...")
    codes = []
    for i in range(12):
        status, _ = post(
            "/submit",
            {
                "text": "This is a test submission for rate limit testing purposes only.",
                "creator_id": "ratelimit-test",
            },
        )
        codes.append(status)
        print(f"  Request {i+1:2d}: HTTP {status}")

    print("\n  Status code summary:", codes)
    assert 429 in codes, "ERROR: Rate limiting did not trigger!"
    print("  ✓ Rate limiting is working correctly")

    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
