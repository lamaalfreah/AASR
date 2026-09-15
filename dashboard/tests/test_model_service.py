import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import TestCase, override_settings

from dashboard.models import QueryLog
from dashboard.services.model_service import (
    ModelConfigurationError,
    ModelTimeoutError,
    analyze_spatial_question,
)
from dashboard.services.geo.client import GeoServiceError


def structured_result(**overrides):
    result = {
        "status": "answered",
        "task_type": "cardinal_direction",
        "reasoning_depth": "concise",
        "data_mode": "user_provided",
        "answer": {"value": "شرق", "text": "تقع النقطة ب شرق النقطة أ."},
        "explanation": "خط طول النقطة ب أكبر من خط طول النقطة أ.",
        "visualization": "direction",
        "anchor": {
            "name": "أ",
            "lat": 24.7,
            "lng": 46.6,
            "role": "anchor",
            "category": None,
            "distance_km": None,
            "step": None,
            "score": None,
            "source_ref": None,
            "source": "user",
            "osm_type": None,
            "osm_id": None,
        },
        "locations": [
            {
                "name": "أ",
                "lat": 24.7,
                "lng": 46.6,
                "role": "anchor",
                "category": None,
                "distance_km": None,
                "step": None,
                "score": None,
                "source_ref": None,
                "source": "user",
                "osm_type": None,
                "osm_id": None,
            },
            {
                "name": "ب",
                "lat": 24.7,
                "lng": 46.7,
                "role": "answer",
                "category": None,
                "distance_km": None,
                "step": None,
                "score": None,
                "source_ref": None,
                "source": "user",
                "osm_type": None,
                "osm_id": None,
            },
        ],
        "metrics": {
            "distance_km": None,
            "radius_km": None,
            "nearest_distance_km": None,
            "count": None,
            "direction": "شرق",
            "suitability": None,
        },
        "evidence": [{"label": "فرق خط الطول", "value": "موجب"}],
        "comparison": [],
        "constraints": [],
        "sources": [],
        "limitations": [],
    }
    result.update(overrides)
    return result


@override_settings(
    OPENAI_API_KEY="test-key",
    ASAR_LLM_MODEL="test-spatial-model",
    ASAR_LLM_TIMEOUT_SECONDS=12,
)
class ModelServiceTests(TestCase):
    @patch("dashboard.services.model_service._create_client")
    def test_calls_one_configurable_model_and_logs_result(self, create_client):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            output_text=json.dumps(structured_result(), ensure_ascii=False)
        )
        create_client.return_value = client

        result = analyze_spatial_question("أين تقع ب من أ؟")

        call = client.responses.create.call_args.kwargs
        self.assertEqual(call["model"], "test-spatial-model")
        self.assertEqual(call["reasoning"], {"effort": "medium"})
        self.assertEqual(call["text"]["verbosity"], "low")
        self.assertEqual(call["text"]["format"]["type"], "json_schema")
        self.assertFalse(call["store"])
        self.assertEqual(call["tools"][0]["name"], "analyze_spatial_query")
        self.assertFalse(call["parallel_tool_calls"])
        self.assertEqual(client.responses.create.call_count, 1)
        self.assertNotIn("router", call)
        self.assertEqual(result["reasoning_depth"], "concise")
        self.assertNotIn("explanation", result)
        self.assertEqual(QueryLog.objects.count(), 1)

    @patch("dashboard.services.model_service._create_client")
    def test_insufficient_information_skips_geometry_checks(self, create_client):
        payload = structured_result(
            status="insufficient_information",
            answer={"value": None, "text": "يلزم توفير الإحداثيات."},
            visualization="none",
            anchor=None,
            locations=[],
            reasoning_depth="concise",
        )
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            output_text=json.dumps(payload, ensure_ascii=False)
        )
        create_client.return_value = client

        result = analyze_spatial_question("ما أقرب مستشفى؟")

        self.assertIsNone(result["verification"]["passed"])
        self.assertEqual(result["verification"]["checks"], [])

    @patch("dashboard.services.model_service._create_client")
    def test_timeout_is_mapped_without_leaking_provider_error(self, create_client):
        api_timeout = type("APITimeoutError", (Exception,), {})
        client = Mock()
        client.responses.create.side_effect = api_timeout("provider detail")
        create_client.return_value = client

        with self.assertRaises(ModelTimeoutError):
            analyze_spatial_question("سؤال")

    @override_settings(OPENAI_API_KEY="")
    def test_requires_api_key(self):
        with self.assertRaises(ModelConfigurationError):
            analyze_spatial_question("سؤال")

    @patch("dashboard.services.geo.tools.analyze_spatial_query")
    @patch("dashboard.services.model_service._create_client")
    def test_geo_tool_uses_same_model_twice_and_returns_grounded_points(self, create_client, analyze_geo):
        tool_result = geo_tool_result()
        analyze_geo.return_value = tool_result
        tool_call = SimpleNamespace(
            type="function_call",
            name="analyze_spatial_query",
            call_id="call-1",
            arguments=json.dumps({
                "operation": "spatial_multi_constraint",
                "reference_place": "جامعة الملك سعود",
                "target_place": None,
                "candidate_places": None,
                "category": None,
                "categories": ["hospital", "clinic", "health_center"],
                "supporting_categories": ["school"],
                "radius_m": 5000,
                "candidate_count": 2,
                "country_code": "sa",
            }),
        )
        tampered_anchor = {**tool_result["anchor"], "lat": 0.0}
        tampered_candidate = {**tool_result["locations"][-1], "lng": 0.0, "score": 0.99}
        final = structured_result(
            task_type="spatial_multi_constraint",
            reasoning_depth="deep",
            data_mode="osm_assisted",
            visualization="service_coverage",
            anchor=tampered_anchor,
            locations=[tampered_anchor, tampered_candidate],
            sources=tool_result["sources"],
            limitations=tool_result["limitations"],
        )
        client = Mock()
        client.responses.create.side_effect = [
            SimpleNamespace(output=[tool_call], output_text=""),
            SimpleNamespace(output=[], output_text=json.dumps(final, ensure_ascii=False)),
        ]
        create_client.return_value = client

        result = analyze_spatial_question("ما أفضل موقع قريب من جامعة الملك سعود؟")

        self.assertEqual(client.responses.create.call_count, 2)
        calls = client.responses.create.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "test-spatial-model")
        self.assertEqual(calls[1].kwargs["model"], "test-spatial-model")
        self.assertNotIn("tools", calls[1].kwargs)
        self.assertEqual(result["data_mode"], "osm_assisted")
        self.assertEqual(result["anchor"]["lat"], 24.722)
        self.assertEqual(result["locations"][-1]["lng"], 46.61)
        self.assertEqual(result["locations"][-1]["score"], 0.72)
        self.assertEqual(len(result["locations"]), 3)
        self.assertEqual(result["metrics"]["count"], 1)

    @patch("dashboard.services.geo.tools.analyze_spatial_query")
    @patch("dashboard.services.model_service._create_client")
    def test_geo_failure_returns_safe_insufficient_result(self, create_client, analyze_geo):
        analyze_geo.side_effect = GeoServiceError("geo_timeout", "private provider detail")
        tool_call = SimpleNamespace(
            type="function_call",
            name="analyze_spatial_query",
            call_id="call-2",
            arguments=json.dumps({
                "operation": "nearest_category",
                "reference_place": "جامعة الملك سعود",
                "target_place": None,
                "candidate_places": None,
                "category": "clinic",
                "categories": None,
                "supporting_categories": None,
                "radius_m": 3000,
                "candidate_count": 1,
                "country_code": "sa",
            }),
        )
        client = Mock()
        client.responses.create.side_effect = [
            SimpleNamespace(output=[tool_call], output_text=""),
            SimpleNamespace(output=[], output_text=json.dumps(structured_result(), ensure_ascii=False)),
        ]
        create_client.return_value = client

        result = analyze_spatial_question("أقرب عيادة إلى جامعة الملك سعود")

        self.assertEqual(client.responses.create.call_count, 2)
        self.assertEqual(result["status"], "insufficient_information")
        self.assertEqual(result["locations"], [])
        self.assertNotIn("private provider detail", json.dumps(result, ensure_ascii=False))


def geo_tool_result():
    anchor = {
        "name": "جامعة الملك سعود، الرياض",
        "lat": 24.722,
        "lng": 46.627,
        "role": "anchor",
        "category": None,
        "distance_km": 0.0,
        "step": None,
        "score": None,
        "source_ref": "nominatim:way:1",
        "source": "nominatim",
        "osm_type": "way",
        "osm_id": "1",
    }
    facility = {
        "name": "عيادة",
        "lat": 24.73,
        "lng": 46.63,
        "role": "facility",
        "category": "clinic",
        "distance_km": 0.94,
        "step": None,
        "score": None,
        "source_ref": "osm:node:2",
        "source": "openstreetmap",
        "osm_type": "node",
        "osm_id": "2",
    }
    candidate = {
        "name": "نقطة تغطية تحليلية 1",
        "lat": 24.70,
        "lng": 46.61,
        "role": "answer",
        "category": "service_coverage_candidate",
        "distance_km": 3.0,
        "step": 1,
        "score": 0.72,
        "source_ref": "candidate:1",
        "source": "derived",
        "osm_type": None,
        "osm_id": None,
    }
    return {
        "ok": True,
        "operation": "spatial_multi_constraint",
        "query": {"operation": "spatial_multi_constraint"},
        "answer_value": "نقطة تغطية تحليلية 1",
        "visualization": "service_coverage",
        "anchor": anchor,
        "locations": [anchor, facility, candidate],
        "metrics": {
            "distance_km": 3.0,
            "radius_km": 5.0,
            "nearest_distance_km": 0.94,
            "count": 1,
            "direction": None,
            "suitability": 0.72,
        },
        "evidence": [{"label": "مرافق OSM داخل النطاق", "value": 1}],
        "comparison": [{
            "label": "نقطة تغطية تحليلية 1",
            "value": "يبعد 3.0 كم عن المرجع",
            "score": 0.72,
        }],
        "constraints": [],
        "sources": [{
            "provider": "OpenStreetMap",
            "attribution": "© OpenStreetMap contributors",
            "url": "https://www.openstreetmap.org/copyright",
            "retrieved_at": "2026-01-01T00:00:00+00:00",
            "data_timestamp": None,
        }],
        "limitations": ["نقطة تحليلية وليست موقع بناء معتمدًا."],
    }
