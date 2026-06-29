#!/usr/bin/env python3
"""
populate_audit_log.py

Directly calls the app functions (no HTTP server needed) to populate the
SQLite audit log with at least 3 entries + 1 appeal, and verify all works.
"""
import sys
import os

# Make the app importable from this script
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv()

from app import (
    init_db, db_insert, db_appeal, db_recent, db_get,
    llm_signal, stylometric_signal, compute_confidence,
    get_attribution, generate_label
)
import uuid
import json

def run_full_classification(text, creator_id, label_desc):
    print(f"\n{'='*60}")
    print(f"[{label_desc}]")
    print(f"Creator: {creator_id}")
    print(f"Text preview: {text[:80]}...")
    
    llm_score = llm_signal(text)
    stylo_result = stylometric_signal(text)
    stylometric_score = stylo_result["score"]
    confidence = compute_confidence(llm_score, stylometric_score)
    attribution = get_attribution(confidence)
    label = generate_label(confidence, attribution)
    content_id = str(uuid.uuid4())

    db_insert(
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        confidence=confidence,
        llm_score=llm_score,
        stylometric_score=stylometric_score,
    )

    print(f"  content_id : {content_id}")
    print(f"  llm_score  : {llm_score}")
    print(f"  stylo_score: {stylometric_score}")
    print(f"  confidence : {confidence}")
    print(f"  attribution: {attribution}")
    print(f"  label      : {label}")
    return content_id

def main():
    print("Initializing database...")
    init_db()

    TESTS = [
        (
            "Clearly AI-generated",
            "test-user-1",
            "Artificial intelligence represents a transformative paradigm shift "
            "in modern society. It is important to note that while the benefits "
            "of AI are numerous, it is equally essential to consider the ethical "
            "implications. Furthermore, stakeholders across various sectors must "
            "collaborate to ensure responsible deployment."
        ),
        (
            "Clearly human-written (casual)",
            "test-user-2",
            "ok so i finally tried that new ramen place downtown and honestly? "
            "underwhelming. the broth was fine but they put WAY too much sodium "
            "in it and i was thirsty for like three hours after. my friend got "
            "the spicy version and said it was better. probably won't go back "
            "unless someone drags me there"
        ),
        (
            "Borderline formal human writing",
            "test-user-3",
            "The relationship between monetary policy and asset price inflation "
            "has been extensively studied in the literature. Central banks face "
            "a fundamental tension between their mandate for price stability and "
            "the unintended consequences of prolonged low interest rates on "
            "equity and real estate valuations."
        ),
        (
            "Lightly edited AI output",
            "test-user-4",
            "I've been thinking a lot about remote work lately. There are genuine "
            "tradeoffs — flexibility and no commute on one side, isolation and "
            "blurred work-life boundaries on the other. Studies show productivity "
            "varies widely by individual and role type."
        ),
    ]

    content_ids = []
    for label_desc, creator_id, text in TESTS:
        cid = run_full_classification(text, creator_id, label_desc)
        content_ids.append(cid)

    # File an appeal on the first submission
    if content_ids:
        print(f"\n[APPEAL] Filing appeal on {content_ids[0]}")
        db_appeal(
            content_ids[0],
            "I wrote this myself from personal experience. "
            "I am a non-native English speaker and my writing style "
            "may appear more formal than typical."
        )
        row = db_get(content_ids[0])
        print(f"  Status after appeal: {row['status']}")
        print(f"  Appeal reasoning: {row['appeal_reasoning'][:60]}...")

    # Show all log entries
    entries = db_recent(limit=10)
    print(f"\n{'='*60}")
    print(f"[AUDIT LOG] {len(entries)} entries:")
    print(json.dumps(entries, indent=2))

    print(f"\n{'='*60}")
    print("ALL DONE. Audit log populated successfully.")

if __name__ == "__main__":
    main()
