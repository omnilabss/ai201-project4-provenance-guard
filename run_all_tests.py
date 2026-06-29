#!/usr/bin/env python3
"""
run_all_tests.py

All-in-one script that:
  1. Populates the SQLite audit log with 4 real entries + 1 appeal (Groq API calls)
  2. Tests rate limiting via Flask test client and records 429 evidence
  3. Runs the full unit/integration test suite
  4. Prints a summary of all results for the README

Run with:
    python3 run_all_tests.py
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from io import StringIO

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv()

# ─── Step 1: Populate the Audit Log with real Groq API calls ─────────────────

def step1_populate_audit_log():
    print("\n" + "="*70)
    print("STEP 1: Populating Audit Log (real Groq API calls)")
    print("="*70)

    from app import (
        init_db, db_insert, db_appeal, db_recent, db_get,
        llm_signal, stylometric_signal, compute_confidence,
        get_attribution, generate_label
    )
    import uuid

    init_db()

    TESTS = [
        (
            "test-user-1",
            "Clearly AI-generated",
            (
                "Artificial intelligence represents a transformative paradigm shift "
                "in modern society. It is important to note that while the benefits "
                "of AI are numerous, it is equally essential to consider the ethical "
                "implications. Furthermore, stakeholders across various sectors must "
                "collaborate to ensure responsible deployment of these technologies."
            ),
        ),
        (
            "test-user-2",
            "Clearly human-written (casual)",
            (
                "ok so i finally tried that new ramen place downtown and honestly? "
                "underwhelming. the broth was fine but they put WAY too much sodium "
                "in it and i was thirsty for like three hours after. my friend got "
                "the spicy version and said it was better. probably won't go back "
                "unless someone drags me there"
            ),
        ),
        (
            "test-user-3",
            "Borderline: formal human writing",
            (
                "The relationship between monetary policy and asset price inflation "
                "has been extensively studied in the literature. Central banks face "
                "a fundamental tension between their mandate for price stability and "
                "the unintended consequences of prolonged low interest rates on "
                "equity and real estate valuations."
            ),
        ),
        (
            "test-user-4",
            "Borderline: lightly edited AI output",
            (
                "I've been thinking a lot about remote work lately. There are genuine "
                "tradeoffs — flexibility and no commute on one side, isolation and "
                "blurred work-life boundaries on the other. Studies show productivity "
                "varies widely by individual and role type."
            ),
        ),
    ]

    content_ids = []
    results = []

    for creator_id, label, text in TESTS:
        print(f"\n  [{label}]")
        print(f"  Calling Groq API...", end="", flush=True)
        t0 = time.time()
        llm_score = llm_signal(text)
        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s)")

        stylo_result = stylometric_signal(text)
        stylometric_score = stylo_result["score"]
        confidence = compute_confidence(llm_score, stylometric_score)
        attribution = get_attribution(confidence)
        label_text = generate_label(confidence, attribution)
        content_id = str(uuid.uuid4())

        db_insert(
            content_id=content_id,
            creator_id=creator_id,
            attribution=attribution,
            confidence=confidence,
            llm_score=llm_score,
            stylometric_score=stylometric_score,
        )

        content_ids.append(content_id)
        results.append({
            "label": label,
            "content_id": content_id,
            "llm_score": llm_score,
            "stylometric_score": stylometric_score,
            "confidence": confidence,
            "attribution": attribution,
            "label_text": label_text,
        })

        print(f"  llm={llm_score:.4f}  stylo={stylometric_score:.4f}  "
              f"confidence={confidence:.4f}  → {attribution}")
        print(f"  label: {label_text[:90]}...")

    # File appeal on first entry
    if content_ids:
        print(f"\n  [APPEAL] Filing on content_id={content_ids[0][:8]}...")
        db_appeal(
            content_ids[0],
            "I wrote this myself from personal experience. "
            "I am a non-native English speaker and my writing style "
            "may appear more formal than typical."
        )
        row = db_get(content_ids[0])
        print(f"  Status after appeal: {row['status']}")

    print("\n  [AUDIT LOG] All entries:")
    entries = db_recent(limit=10)
    print(json.dumps(entries, indent=2))

    return entries, results


# ─── Step 2: Rate Limit Test via Flask Test Client ────────────────────────────

def step2_rate_limit_test():
    print("\n" + "="*70)
    print("STEP 2: Rate Limit Test (Flask test client — no server needed)")
    print("="*70)

    # Import fresh to avoid state pollution
    import app as app_module
    flask_app = app_module.app
    flask_app.config["TESTING"] = True

    status_codes = []

    with flask_app.test_client() as client:
        # We mock the LLM to avoid 12 API calls here — rate limiting
        # works at the HTTP layer, independent of detection logic
        with patch("app.llm_signal", return_value=0.75):
            print("\n  Sending 12 rapid POST /submit requests...")
            for i in range(1, 13):
                resp = client.post(
                    "/submit",
                    data=json.dumps({
                        "text": "This is a test submission for rate limit testing purposes only.",
                        "creator_id": "ratelimit-test"
                    }),
                    content_type="application/json"
                )
                status_codes.append(resp.status_code)
                marker = "✓" if resp.status_code == 200 else "✗ 429"
                print(f"  Request {i:2d}: HTTP {resp.status_code} {marker}")

                # Capture the 429 body
                if resp.status_code == 429 and i == 11:
                    body = json.loads(resp.data)
                    print(f"  429 body: {json.dumps(body, indent=2)}")

    print(f"\n  Status code sequence: {status_codes}")
    ok_count = sum(1 for c in status_codes if c == 200)
    throttled_count = sum(1 for c in status_codes if c == 429)
    print(f"  200 responses: {ok_count}  |  429 responses: {throttled_count}")

    if 429 in status_codes:
        print("  ✅ Rate limiting is working correctly!")
    else:
        print("  ⚠️  WARNING: Rate limiting did not trigger as expected")

    return status_codes


# ─── Step 3: Unit/Integration Test Suite ─────────────────────────────────────

def step3_run_tests():
    print("\n" + "="*70)
    print("STEP 3: Running Unit & Integration Test Suite")
    print("="*70 + "\n")

    # Use a separate temp DB so tests don't pollute the real audit log
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db.close()

    import app as app_module
    original_db = app_module.DB_PATH
    app_module.DB_PATH = tmp_db.name
    app_module.init_db()

    try:
        loader = unittest.TestLoader()
        suite = loader.discover(
            start_dir=os.path.join(PROJECT_DIR, "tests"),
            pattern="test_*.py"
        )

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    finally:
        app_module.DB_PATH = original_db
        os.unlink(tmp_db.name)

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "★"*70)
    print("  Provenance Guard — All-in-One Test Runner")
    print("★"*70)

    # Step 3 FIRST: Unit tests (before rate limit is exhausted)
    test_result = step3_run_tests()

    # Step 2: Rate limiting test (exhausts the limiter)
    status_codes = step2_rate_limit_test()

    # Step 1: Real Groq API calls to populate DB (runs after rate limiter resets)
    entries, results = step1_populate_audit_log()

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"\n  Audit log entries created : {len(entries)}")
    print(f"  Confidence range observed : "
          f"{min(r['confidence'] for r in results):.4f} – "
          f"{max(r['confidence'] for r in results):.4f}")

    ok = sum(1 for c in status_codes if c == 200)
    bad = sum(1 for c in status_codes if c == 429)
    print(f"  Rate limit test           : {ok} × 200, {bad} × 429")
    print(f"  Unit tests                : "
          f"{test_result.testsRun} run, "
          f"{len(test_result.failures)} failures, "
          f"{len(test_result.errors)} errors")

    all_ok = (
        len(entries) >= 3
        and 429 in status_codes
        and test_result.wasSuccessful()
    )
    print(f"\n  Overall result: {'✅ ALL GOOD' if all_ok else '⚠️  CHECK OUTPUT ABOVE'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
