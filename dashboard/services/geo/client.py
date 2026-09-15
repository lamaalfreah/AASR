"""Low-volume, policy-aware clients for Nominatim and Overpass."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass

from django.conf import settings
from django.core.cache import caches


class GeoServiceError(Exception):
    """A safe, classified failure from an external geographic service."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


POI_FILTERS = {
    "hospital": (("amenity", "hospital"), ("healthcare", "hospital")),
    "clinic": (("amenity", "clinic"), ("healthcare", "clinic")),
    "health_center": (
        ("amenity", "clinic"),
        ("healthcare", "clinic"),
        ("healthcare", "health_post"),
    ),
    "doctors": (("amenity", "doctors"), ("healthcare", "doctor")),
    "pharmacy": (("amenity", "pharmacy"), ("healthcare", "pharmacy")),
    "school": (("amenity", "school"),),
    "kindergarten": (("amenity", "kindergarten"),),
    "college": (("amenity", "college"),),
    "university": (("amenity", "university"),),
    "metro_station": (
        ("railway", "station"),
        ("station", "subway"),
        ("railway", "subway_entrance"),
    ),
    "mosque": (
        ("amenity", "place_of_worship"),
    ),
}


_nominatim_lock = threading.Lock()
_last_nominatim_request = 0.0


def _cache_key(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"asar:{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _http_client():
    import httpx

    return httpx.Client(timeout=settings.ASAR_GEO_TIMEOUT_SECONDS, follow_redirects=True)


def _request_error_code(exc: Exception) -> str:
    name = exc.__class__.__name__
    if "Timeout" in name:
        return "geo_timeout"
    return "geo_unavailable"


def _wait_for_nominatim_slot() -> None:
    """Enforce the public Nominatim maximum of one request per second."""
    global _last_nominatim_request
    with _nominatim_lock:
        elapsed = time.monotonic() - _last_nominatim_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        _last_nominatim_request = time.monotonic()


@dataclass
class OSMClient:
    http_client: object | None = None

    def __post_init__(self):
        if not settings.ASAR_GEO_USER_AGENT.strip():
            raise GeoServiceError(
                "geo_not_configured",
                "ASAR_GEO_USER_AGENT is required for public geographic services.",
            )
        if self.http_client is None:
            self.http_client = _http_client()
        self.headers = {
            "User-Agent": settings.ASAR_GEO_USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "ar,en;q=0.8",
        }

    def close(self) -> None:
        close = getattr(self.http_client, "close", None)
        if callable(close):
            close()

    def geocode_place(self, name: str, country_code: str | None = None) -> list[dict]:
        normalized_name = " ".join(name.split())
        country_code = (country_code or "").strip().lower() or None
        key = _cache_key("geocode", [normalized_name.casefold(), country_code])
        cache = caches["geospatial"]
        cached = cache.get(key)
        if cached is not None:
            return cached

        params = {
            "q": normalized_name,
            "format": "jsonv2",
            "limit": 3,
            "addressdetails": 1,
            "accept-language": "ar,en",
        }
        if country_code:
            params["countrycodes"] = country_code

        _wait_for_nominatim_slot()
        try:
            response = self.http_client.get(
                f"{settings.ASAR_NOMINATIM_URL}/search",
                params=params,
                headers=self.headers,
            )
            if response.status_code == 429:
                raise GeoServiceError("geo_rate_limited", "Nominatim rate limit reached.")
            response.raise_for_status()
            payload = response.json()
        except GeoServiceError:
            raise
        except Exception as exc:
            raise GeoServiceError(_request_error_code(exc), "Nominatim request failed.") from exc

        if not isinstance(payload, list):
            raise GeoServiceError("geo_invalid_response", "Nominatim returned invalid data.")

        results = []
        for index, item in enumerate(payload):
            try:
                lat, lng = float(item["lat"]), float(item["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                continue
            osm_type = item.get("osm_type")
            osm_id = item.get("osm_id")
            results.append({
                "source_ref": f"nominatim:{osm_type}:{osm_id or index}",
                "name": str(item.get("display_name") or normalized_name),
                "lat": lat,
                "lng": lng,
                "osm_type": osm_type if osm_type in {"node", "way", "relation"} else None,
                "osm_id": str(osm_id) if osm_id is not None else None,
                "importance": self._safe_float(item.get("importance")),
                "address": item.get("address") if isinstance(item.get("address"), dict) else {},
            })
        cache.set(key, results, settings.ASAR_GEO_CACHE_SECONDS)
        return results

    def search_nearby_pois(
        self,
        lat: float,
        lng: float,
        categories: list[str],
        radius_m: int,
    ) -> dict:
        invalid = sorted(set(categories) - POI_FILTERS.keys())
        if invalid:
            raise GeoServiceError("unsupported_category", "Unsupported POI category.")
        radius_m = int(radius_m)
        if not 100 <= radius_m <= settings.ASAR_GEO_MAX_RADIUS_M:
            raise GeoServiceError("invalid_radius", "Radius is outside the supported range.")
        from .local_osm import search_local_pois

        try:
            local_result = search_local_pois(
                lat,
                lng,
                categories,
                radius_m,
            )

            if local_result["items"]:
                return local_result

        except GeoServiceError as exc:
            print("\n===== LOCAL OSM ERROR =====")
            print("CODE:", exc.code)
            print("MESSAGE:", str(exc))
            print("===== END LOCAL OSM ERROR =====\n")
        categories = sorted(set(categories))
        key = _cache_key("overpass", [round(lat, 5), round(lng, 5), categories, radius_m])
        cache = caches["geospatial"]
        cached = cache.get(key)
        if cached is not None:
            return cached

        filters = sorted({pair for category in categories for pair in POI_FILTERS[category]})
        fragments = []
        for key_name, value in filters:
            fragments.extend(
                f'{kind}(around:{radius_m},{lat:.7f},{lng:.7f})["{key_name}"="{value}"];'
                for kind in ("node", "way", "relation")
            )
        query = (
            f"[out:json][timeout:{settings.ASAR_OVERPASS_QUERY_TIMEOUT_SECONDS}];"
            f"({''.join(fragments)});out center tags;"
        )
        try:
            response = self.http_client.post(
                settings.ASAR_OVERPASS_URL,
                data={"data": query},
                headers=self.headers,
            )
            if response.status_code == 429:
                raise GeoServiceError("geo_rate_limited", "Overpass rate limit reached.")
            response.raise_for_status()
            payload = response.json()
        except GeoServiceError:
            raise
        except Exception as exc:
            print("\n===== OVERPASS REAL ERROR =====")
            print("TYPE:", type(exc).__name__)
            print("MESSAGE:", str(exc))

            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                print("STATUS:", response_obj.status_code)
                try:
                    print("BODY:", response_obj.text[:2000])
                except Exception:
                    pass

            print("QUERY:")
            print(query)
            print("===== END OVERPASS ERROR =====\n")

            raise GeoServiceError(
                _request_error_code(exc),
                "Overpass request failed."
            ) from exc

        elements = payload.get("elements") if isinstance(payload, dict) else None
        if not isinstance(elements, list):
            raise GeoServiceError("geo_invalid_response", "Overpass returned invalid data.")

        items = []
        seen = set()
        for element in elements:
            normalized = self._normalize_element(element, categories)
            if normalized is None:
                continue
            identity = (normalized["osm_type"], normalized["osm_id"])
            if identity in seen:
                continue
            seen.add(identity)
            items.append(normalized)
            if len(items) >= settings.ASAR_GEO_MAX_RESULTS:
                break

        osm3s = payload.get("osm3s") if isinstance(payload.get("osm3s"), dict) else {}
        result = {
            "items": items,
            "data_timestamp": osm3s.get("timestamp_osm_base"),
        }
        cache.set(key, result, settings.ASAR_GEO_CACHE_SECONDS)
        return result

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize_element(element: dict, requested_categories: list[str]) -> dict | None:
        if not isinstance(element, dict):
            return None
        center = element.get("center") or {}
        try:
            lat = float(element.get("lat", center.get("lat")))
            lng = float(element.get("lon", center.get("lon")))
        except (TypeError, ValueError):
            return None
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return None
        osm_type, osm_id = element.get("type"), element.get("id")
        if osm_type not in {"node", "way", "relation"} or osm_id is None:
            return None
        tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
        category = next(
            (
                requested
                for requested in requested_categories
                if any(tags.get(key) == value for key, value in POI_FILTERS[requested])
            ),
            None,
        )
        if category is None:
            return None
        return {
            "source_ref": f"osm:{osm_type}:{osm_id}",
            "name": str(tags.get("name:ar") or tags.get("name") or f"مرفق {osm_id}"),
            "lat": lat,
            "lng": lng,
            "category": category,
            "osm_type": osm_type,
            "osm_id": str(osm_id),
        }
    def google_geocode_place(self, name: str) -> list[dict]:
        api_key = settings.ASAR_GOOGLE_MAPS_API_KEY

        if not api_key:
            print("GOOGLE_MAPS_API_KEY is missing")
            return []

        try:
            response = self.http_client.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={
                    "address": name,
                    "key": api_key,
                    "language": "ar",
                    "region": "sa",
                },
            )

            response.raise_for_status()
            payload = response.json()

        except Exception as exc:
            print("\n===== GOOGLE GEOCODING ERROR =====")
            print("TYPE:", type(exc).__name__)
            print("MESSAGE:", str(exc))
            print("===== END GOOGLE ERROR =====\n")
            return []

        print("\n===== GOOGLE GEOCODING DEBUG =====")
        print("STATUS:", payload.get("status"))
        print("ERROR:", payload.get("error_message"))
        print("RESULTS:", len(payload.get("results", [])))
        print("===== END GOOGLE DEBUG =====\n")

        results = []

        for index, item in enumerate(payload.get("results", [])):
            location = item.get("geometry", {}).get("location", {})

            lat = location.get("lat")
            lng = location.get("lng")

            if lat is None or lng is None:
                continue

            results.append(
                {
                    "source_ref": f"google:{item.get('place_id', index)}",
                    "name": item.get("formatted_address") or name,
                    "lat": float(lat),
                    "lng": float(lng),
                    "osm_type": None,
                    "osm_id": None,
                    "importance": 1.0,
                    "address": {},
                }
            )

        return results