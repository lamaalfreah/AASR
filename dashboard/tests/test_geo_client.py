from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings

from dashboard.services.geo.client import OSMClient


TEST_CACHES = {
    "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
    "geospatial": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@override_settings(
    ASAR_GEO_USER_AGENT="ASAR-test/1.0 (test@example.com)",
    ASAR_GEO_CACHE_SECONDS=60,
    ASAR_GEO_MAX_RADIUS_M=15000,
    ASAR_GEO_MAX_RESULTS=80,
    ASAR_NOMINATIM_URL="https://nominatim.example",
    ASAR_OVERPASS_URL="https://overpass.example/api/interpreter",
    ASAR_OVERPASS_QUERY_TIMEOUT_SECONDS=15,
    CACHES=TEST_CACHES,
)
class GeoClientTests(SimpleTestCase):
    @patch("dashboard.services.geo.client._wait_for_nominatim_slot")
    def test_geocode_uses_policy_headers_country_filter_and_cache(self, wait_slot):
        http = Mock()
        http.get.return_value = FakeResponse([{
            "lat": "24.722",
            "lon": "46.627",
            "display_name": "جامعة الملك سعود، الرياض",
            "osm_type": "way",
            "osm_id": 123,
            "importance": 0.7,
            "address": {"city": "الرياض"},
        }])
        client = OSMClient(http_client=http)

        first = client.geocode_place("جامعة الملك سعود", "sa")
        second = client.geocode_place("جامعة   الملك سعود", "sa")

        self.assertEqual(first, second)
        self.assertEqual(http.get.call_count, 1)
        self.assertEqual(wait_slot.call_count, 1)
        kwargs = http.get.call_args.kwargs
        self.assertEqual(kwargs["params"]["countrycodes"], "sa")
        self.assertIn("ASAR-test", kwargs["headers"]["User-Agent"])
        self.assertEqual(first[0]["source_ref"], "nominatim:way:123")

    def test_overpass_combines_allowlisted_categories_and_normalizes_centers(self):
        http = Mock()
        http.post.return_value = FakeResponse({
            "osm3s": {"timestamp_osm_base": "2026-01-01T00:00:00Z"},
            "elements": [
                {
                    "type": "node",
                    "id": 10,
                    "lat": 24.72,
                    "lon": 46.63,
                    "tags": {"amenity": "hospital", "name:ar": "مستشفى أ"},
                },
                {
                    "type": "way",
                    "id": 11,
                    "center": {"lat": 24.73, "lon": 46.64},
                    "tags": {"amenity": "school", "name": "School B"},
                },
            ],
        })
        client = OSMClient(http_client=http)

        result = client.search_nearby_pois(24.722, 46.627, ["hospital", "school"], 5000)

        query = http.post.call_args.kwargs["data"]["data"]
        self.assertIn('["amenity"="hospital"]', query)
        self.assertIn('["amenity"="school"]', query)
        self.assertNotIn("{{", query)
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["items"][1]["lat"], 24.73)
        self.assertEqual(result["items"][0]["category"], "hospital")
        self.assertEqual(result["data_timestamp"], "2026-01-01T00:00:00Z")
