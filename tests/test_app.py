"""
tests/test_app.py

Unit tests for Provenance Guard.
Run with: python3 -m pytest tests/ -v

These tests do NOT make Groq API calls — the LLM signal is mocked.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Make the project importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch DB_PATH before importing the app so tests use a temp database
_TEMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TEMP_DB.close()

import app as app_module

app_module.DB_PATH = _TEMP_DB.name
app_module.init_db()

from app import (
    app,
    compute_confidence,
    get_attribution,
    generate_label,
    stylometric_signal,
)

app.config["TESTING"] = True


class TestStylometricSignal(unittest.TestCase):
    """Tests for the pure-Python stylometric heuristics."""

    def test_empty_text_returns_neutral(self):
        result = stylometric_signal("")
        self.assertEqual(result["score"], 0.5)

    def test_uniform_ai_like_text_scores_higher(self):
        ai_text = (
            "Artificial intelligence represents a transformative paradigm. "
            "It is important to ensure responsible deployment. "
            "Stakeholders must collaborate across various sectors. "
            "The benefits of AI must be weighed against ethical implications."
        )
        casual_text = (
            "ok so i tried that ramen place and honestly? underwhelming. "
            "broth was fine but WAY too salty. my friend said it was better "
            "with the spicy version. probably won't go back lol"
        )
        ai_score = stylometric_signal(ai_text)["score"]
        human_score = stylometric_signal(casual_text)["score"]
        self.assertGreater(ai_score, human_score)

    def test_returns_all_expected_keys(self):
        result = stylometric_signal("Hello world. This is a test sentence.")
        for key in [
            "score",
            "sentence_length_variance_score",
            "type_token_ratio_score",
            "punctuation_density_score",
            "avg_sentence_length_score",
        ]:
            self.assertIn(key, result)

    def test_score_in_valid_range(self):
        text = "Some normal human text here. It has a few sentences! What do you think?"
        result = stylometric_signal(text)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)


class TestComputeConfidence(unittest.TestCase):
    """Tests for the confidence combining logic."""

    def test_high_both_signals_gives_high_confidence(self):
        conf = compute_confidence(0.9, 0.85)
        self.assertGreater(conf, 0.70)

    def test_low_both_signals_gives_low_confidence(self):
        conf = compute_confidence(0.1, 0.15)
        self.assertLess(conf, 0.35)

    def test_disagreement_dampens_toward_neutral(self):
        # LLM says 0.9 AI, stylometric says 0.3 human (disagree by 0.6 > 0.4)
        dampened = compute_confidence(0.9, 0.3)
        # Without dampening: 0.9*0.65 + 0.3*0.35 = 0.585 + 0.105 = 0.69
        # With dampening: 0.5 + (0.69-0.5)*0.6 = 0.5 + 0.114 = 0.614
        self.assertLess(dampened, 0.70)  # dampened below likely_ai threshold

    def test_output_in_valid_range(self):
        for llm, stylo in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.3, 0.8)]:
            conf = compute_confidence(llm, stylo)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)


class TestGetAttribution(unittest.TestCase):
    """Tests for the attribution threshold logic."""

    def test_high_confidence_is_likely_ai(self):
        self.assertEqual(get_attribution(0.85), "likely_ai")
        self.assertEqual(get_attribution(0.71), "likely_ai")

    def test_low_confidence_is_likely_human(self):
        self.assertEqual(get_attribution(0.20), "likely_human")
        self.assertEqual(get_attribution(0.34), "likely_human")

    def test_middle_is_uncertain(self):
        self.assertEqual(get_attribution(0.50), "uncertain")
        self.assertEqual(get_attribution(0.35), "uncertain")
        self.assertEqual(get_attribution(0.70), "uncertain")


class TestGenerateLabel(unittest.TestCase):
    """Tests for transparency label generation."""

    def test_likely_ai_label_contains_warning(self):
        label = generate_label(0.85, "likely_ai")
        self.assertIn("⚠️", label)
        self.assertIn("AI-Generated", label)
        self.assertIn("appeal", label)

    def test_likely_human_label_contains_checkmark(self):
        label = generate_label(0.20, "likely_human")
        self.assertIn("✓", label)
        self.assertIn("Human-Written", label)

    def test_uncertain_label_contains_uncertain_symbol(self):
        label = generate_label(0.50, "uncertain")
        self.assertIn("◐", label)
        self.assertIn("Uncertain", label)
        self.assertIn("appeal", label)

    def test_label_contains_percentage(self):
        label = generate_label(0.78, "likely_ai")
        self.assertIn("78%", label)


class TestSubmitEndpoint(unittest.TestCase):
    """Integration tests for POST /submit."""

    def setUp(self):
        self.client = app.test_client()

    def _mock_llm(self, score):
        """Return a patcher that makes llm_signal return a fixed score."""
        return patch("app.llm_signal", return_value=score)

    def test_missing_text_returns_400(self):
        resp = self.client.post(
            "/submit",
            data=json.dumps({"creator_id": "user1"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_missing_creator_id_returns_400(self):
        resp = self.client.post(
            "/submit",
            data=json.dumps({"text": "some text"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_valid_submission_returns_200_with_required_fields(self):
        with self._mock_llm(0.85):
            resp = self.client.post(
                "/submit",
                data=json.dumps({"text": "Test text here.", "creator_id": "test"}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        for field in ["content_id", "attribution", "confidence", "signal_scores", "label", "status"]:
            self.assertIn(field, body)

    def test_high_llm_score_produces_likely_ai(self):
        with self._mock_llm(0.95):
            resp = self.client.post(
                "/submit",
                data=json.dumps(
                    {
                        "text": (
                            "It is important to note that this content demonstrates "
                            "various indicators. Furthermore, stakeholders must ensure "
                            "responsible implementation across all sectors."
                        ),
                        "creator_id": "test",
                    }
                ),
                content_type="application/json",
            )
        body = json.loads(resp.data)
        self.assertEqual(body["attribution"], "likely_ai")

    def test_low_llm_score_produces_likely_human(self):
        with self._mock_llm(0.05):
            resp = self.client.post(
                "/submit",
                data=json.dumps(
                    {"text": "ok i'm not sure what to say lol. this is weird!", "creator_id": "test"}
                ),
                content_type="application/json",
            )
        body = json.loads(resp.data)
        # Stylometric may push up slightly but with llm=0.05 it should be human or uncertain
        self.assertIn(body["attribution"], ["likely_human", "uncertain"])


class TestAppealEndpoint(unittest.TestCase):
    """Integration tests for POST /appeal."""

    def setUp(self):
        self.client = app.test_client()

    def _create_submission(self):
        with patch("app.llm_signal", return_value=0.85):
            resp = self.client.post(
                "/submit",
                data=json.dumps(
                    {
                        "text": "It is important to note that various stakeholders must collaborate.",
                        "creator_id": "appeal-test",
                    }
                ),
                content_type="application/json",
            )
        return json.loads(resp.data)["content_id"]

    def test_appeal_missing_content_id_returns_400(self):
        resp = self.client.post(
            "/appeal",
            data=json.dumps({"creator_reasoning": "I wrote this myself."}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_appeal_unknown_content_id_returns_404(self):
        resp = self.client.post(
            "/appeal",
            data=json.dumps(
                {"content_id": "nonexistent-id", "creator_reasoning": "I wrote this."}
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)

    def test_valid_appeal_returns_under_review(self):
        cid = self._create_submission()
        resp = self.client.post(
            "/appeal",
            data=json.dumps(
                {"content_id": cid, "creator_reasoning": "I wrote this myself."}
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertEqual(body["status"], "under_review")

    def test_missing_reasoning_returns_400(self):
        cid = self._create_submission()
        resp = self.client.post(
            "/appeal",
            data=json.dumps({"content_id": cid, "creator_reasoning": ""}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)


class TestLogEndpoint(unittest.TestCase):
    """Tests for GET /log."""

    def setUp(self):
        self.client = app.test_client()

    def test_log_returns_list(self):
        resp = self.client.get("/log")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data)
        self.assertIsInstance(body, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
