from django.test import SimpleTestCase, override_settings

from dashboard.services.geo.analysis import (
    calculate_direction,
    calculate_distance,
    count_within_radius,
    generate_service_candidates,
)
from dashboard.services.geo.tools import (
    ANALYZE_SPATIAL_QUERY_TOOL,
    GeoToolContext,
    analyze_spatial_query,
    ground_geo_response,
)
from dashboard.services.agents.verifier import verify_result


def named_place(name, lat, lng, osm_id):
    return {
        "source_ref": f"nominatim:way:{osm_id}",
        "name": name,
        "lat": lat,
        "lng": lng,
        "osm_type": "way",
        "osm_id": str(osm_id),
        "importance": 0.8,
        "address": {},
    }


NAMED_PLACES = {
    "المرجع": named_place("المرجع", 24.700, 46.600, 1),
    "الهدف": named_place("الهدف", 24.710, 46.600, 2),
    "المرشح أ": named_place("المرشح أ", 24.701, 46.600, 3),
    "المرشح ب": named_place("المرشح ب", 24.720, 46.600, 4),
}

POIS = [
    {
        "source_ref": "osm:node:10",
        "name": "مستشفى قريب",
        "lat": 24.703,
        "lng": 46.600,
        "category": "hospital",
        "osm_type": "node",
        "osm_id": "10",
    },
    {
        "source_ref": "osm:node:11",
        "name": "عيادة",
        "lat": 24.708,
        "lng": 46.600,
        "category": "clinic",
        "osm_type": "node",
        "osm_id": "11",
    },
    {
        "source_ref": "osm:node:12",
        "name": "مدرسة",
        "lat": 24.704,
        "lng": 46.600,
        "category": "school",
        "osm_type": "node",
        "osm_id": "12",
    },
]


class FakeGeoClient:
    def geocode_place(self, name, _country_code):
        place = NAMED_PLACES.get(name)
        return [dict(place)] if place else []

    def search_nearby_pois(self, _lat, _lng, categories, _radius_m):
        return {
            "items": [dict(point) for point in POIS if point["category"] in categories],
            "data_timestamp": "2026-01-01T00:00:00Z",
        }


def tool_arguments(operation, **overrides):
    arguments = {
        "operation": operation,
        "reference_place": "المرجع",
        "target_place": None,
        "candidate_places": None,
        "category": None,
        "categories": None,
        "supporting_categories": None,
        "radius_m": 5000,
        "candidate_count": 2,
        "country_code": "sa",
    }
    arguments.update(overrides)
    return arguments


@override_settings(
    ASAR_CANDIDATE_GAP_WEIGHT=0.50,
    ASAR_CANDIDATE_ANCHOR_WEIGHT=0.30,
    ASAR_CANDIDATE_SUPPORT_WEIGHT=0.20,
    ASAR_GEO_MAX_RADIUS_M=15000,
    ASAR_GEO_DEFAULT_COUNTRY_CODE="sa",
    ASAR_NOMINATIM_URL="https://nominatim.example",
)
class GeoAnalysisTests(SimpleTestCase):
    def setUp(self):
        self.anchor = {"lat": 24.722, "lng": 46.627}
        self.client = FakeGeoClient()

    def test_distance_direction_and_radius_count_are_deterministic(self):
        near = {"lat": 24.723, "lng": 46.627}
        far = {"lat": 24.80, "lng": 46.627}

        self.assertAlmostEqual(calculate_distance(24.722, 46.627, 24.723, 46.627), 0.111, places=2)
        self.assertEqual(calculate_direction(self.anchor, near), "شمال")
        self.assertEqual(count_within_radius([near, far], self.anchor, 1000), 1)

    def test_candidate_generation_is_bounded_ranked_and_repeatable(self):
        facilities = [{"lat": 24.73, "lng": 46.627}]
        supporting = [{"lat": 24.722, "lng": 46.64}]

        first = generate_service_candidates(self.anchor, facilities, supporting, 5000, 3)
        second = generate_service_candidates(self.anchor, facilities, supporting, 5000, 3)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertGreaterEqual(first[0]["score"], first[1]["score"])
        self.assertTrue(all(0 <= item["score"] <= 1 for item in first))

    def test_single_tool_schema_exposes_all_eight_operations(self):
        self.assertEqual(ANALYZE_SPATIAL_QUERY_TOOL["name"], "analyze_spatial_query")
        enum = ANALYZE_SPATIAL_QUERY_TOOL["parameters"]["properties"]["operation"]["enum"]
        self.assertEqual(len(enum), 8)

    def test_all_eight_operations_return_grounded_deterministic_results(self):
        scenarios = [
            (
                "nearest_category",
                tool_arguments("nearest_category", category="hospital"),
                "مستشفى قريب",
                "nearest",
            ),
            (
                "cardinal_direction",
                tool_arguments("cardinal_direction", target_place="الهدف", radius_m=None),
                "شمال",
                "direction",
            ),
            (
                "within_radius_yes_no",
                tool_arguments("within_radius_yes_no", target_place="الهدف", radius_m=2000),
                True,
                "radius_yes_no",
            ),
            (
                "closer_of_two",
                tool_arguments(
                    "closer_of_two",
                    candidate_places=["المرشح أ", "المرشح ب"],
                    radius_m=None,
                ),
                "المرشح أ",
                "comparison",
            ),
            (
                "count_within_radius",
                tool_arguments("count_within_radius", category="hospital", radius_m=5000),
                1,
                "count",
            ),
            (
                "nearest_of_two_categories",
                tool_arguments("nearest_of_two_categories", categories=["hospital", "clinic"]),
                "مستشفى قريب",
                "category_comparison",
            ),
            (
                "two_hop_nearest",
                tool_arguments("two_hop_nearest", categories=["hospital", "school"]),
                "مدرسة",
                "two_hop",
            ),
            (
                "spatial_multi_constraint",
                tool_arguments(
                    "spatial_multi_constraint",
                    categories=["hospital"],
                    supporting_categories=["school"],
                ),
                "نقطة تغطية تحليلية 1",
                "service_coverage",
            ),
        ]

        for operation, arguments, answer, visualization in scenarios:
            with self.subTest(operation=operation):
                result = analyze_spatial_query(arguments, client=self.client)
                self.assertTrue(result["ok"])
                self.assertEqual(result["operation"], operation)
                self.assertEqual(result["answer_value"], answer)
                self.assertEqual(result["visualization"], visualization)
                self.assertEqual(result["anchor"]["source"], "nominatim")
                self.assertTrue(
                    all(point["source_ref"] for point in result["locations"])
                )
                context = GeoToolContext(
                    called=True,
                    result=result,
                    registry={point["source_ref"]: point for point in result["locations"]},
                )
                model_result = {
                    "status": "answered",
                    "task_type": operation,
                    "reasoning_depth": "deep",
                    "data_mode": "osm_assisted",
                    "answer": {"value": "قيمة غير موثوقة", "text": "إجابة عربية موجزة."},
                    "reasoning": "تفسير موجز.",
                    "visualization": "none",
                    "anchor": {**result["anchor"], "lat": 0.0},
                    "locations": result["locations"],
                    "metrics": {},
                    "evidence": [],
                    "comparison": [],
                    "constraints": [],
                    "sources": [],
                    "limitations": [],
                }
                grounded = ground_geo_response(model_result, context)
                verified = verify_result(grounded)
                self.assertEqual(grounded["answer"]["value"], answer)
                self.assertNotEqual(verified["verification"]["passed"], False)

    @override_settings(
        ASAR_CANDIDATE_GAP_WEIGHT=0,
        ASAR_CANDIDATE_ANCHOR_WEIGHT=0,
        ASAR_CANDIDATE_SUPPORT_WEIGHT=0,
    )
    def test_zero_weights_use_safe_documented_fallback(self):
        candidates = generate_service_candidates(self.anchor, [], [], 1000, 1)
        self.assertEqual(len(candidates), 1)
        self.assertGreaterEqual(candidates[0]["score"], 0)
