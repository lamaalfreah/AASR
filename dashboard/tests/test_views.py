import json
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings
from django.urls import reverse

from dashboard.services.model_service import ModelTimeoutError


class AnalyzeApiTests(SimpleTestCase):
    def post(self, payload, content_type="application/json"):
        body = json.dumps(payload) if not isinstance(payload, str) else payload
        return self.client.post(reverse("analyze_api"), data=body, content_type=content_type)

    def test_dashboard_page_loads(self):
        response = self.client.get(reverse("dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "reasoning-depth")
        self.assertContains(response, "sources-card")
        self.assertContains(response, "limitations-card")

    @patch("dashboard.views.analyze_spatial_question")
    def test_returns_model_result(self, analyze):
        analyze.return_value = {
            "status": "answered",
            "task_type": "cardinal_direction",
            "reasoning_depth": "concise",
            "data_mode": "user_provided",
            "answer": {"value": "شرق", "text": "تقع ب شرق أ."},
            "sources": [],
            "limitations": [],
        }

        response = self.post({"question": " أين تقع ب من أ؟ "})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"]["value"], "شرق")
        analyze.assert_called_once_with("أين تقع ب من أ؟")

    def test_rejects_non_json_content_type(self):
        response = self.client.post(reverse("analyze_api"), data={"question": "x"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_content_type")

    def test_rejects_invalid_json(self):
        response = self.post("{")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_json")

    def test_rejects_non_string_question(self):
        response = self.post({"question": 123})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_question")

    @override_settings(ASAR_MAX_QUESTION_CHARS=5)
    def test_rejects_question_over_limit(self):
        response = self.post({"question": "123456"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "question_too_long")

    @patch("dashboard.views.analyze_spatial_question", side_effect=ModelTimeoutError())
    def test_maps_model_timeout_to_safe_error(self, _analyze):
        response = self.post({"question": "سؤال مكاني"})

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["error"]["code"], "model_timeout")
        self.assertNotIn("traceback", response.content.decode().lower())
