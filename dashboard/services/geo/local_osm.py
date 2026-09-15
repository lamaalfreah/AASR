from __future__ import annotations

import math
import os

from django.conf import settings
from pyrosm import OSM

from .client import GeoServiceError, POI_FILTERS


def _bbox_from_radius(lat: float, lng: float, radius_m: int) -> list[float]:
    """
    Return bounding box:
    [min_lon, min_lat, max_lon, max_lat]
    """

    radius_km = radius_m / 1000

    lat_delta = radius_km / 111.0

    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    lon_delta = radius_km / (111.0 * cos_lat)

    return [
        lng - lon_delta,
        lat - lat_delta,
        lng + lon_delta,
        lat + lat_delta,
    ]


def _build_filter(categories: list[str]) -> dict:
    """
    Convert ASAR categories into pyrosm custom_filter.
    """

    custom_filter: dict[str, list[str]] = {}

    for category in categories:
        if category not in POI_FILTERS:
            raise GeoServiceError(
                "unsupported_category",
                f"Unsupported POI category: {category}",
            )

        for key, value in POI_FILTERS[category]:
            custom_filter.setdefault(key, [])

            if value not in custom_filter[key]:
                custom_filter[key].append(value)

    return custom_filter


def _detect_category(row, requested_categories: list[str]) -> str | None:
    for category in requested_categories:
        filters = POI_FILTERS[category]

        for key, value in filters:
            try:
                current = row.get(key)
            except Exception:
                current = None

            if current == value:
                return category

        # pyrosm may preserve some tags inside the "tags" field
        raw_tags = row.get("tags") if hasattr(row, "get") else None

        if isinstance(raw_tags, dict):
            for key, value in filters:
                if raw_tags.get(key) == value:
                    return category

    return None


def search_local_pois(
    lat: float,
    lng: float,
    categories: list[str],
    radius_m: int,
) -> dict:
    pbf_path = settings.ASAR_OSM_PBF_PATH

    if not pbf_path or not os.path.exists(pbf_path):
        raise GeoServiceError(
            "local_osm_not_configured",
            "Local OSM PBF file was not found.",
        )

    bbox = _bbox_from_radius(lat, lng, radius_m)
    custom_filter = _build_filter(categories)

    try:
        osm = OSM(
            pbf_path,
            bounding_box=bbox,
        )

        pois = osm.get_pois(
            custom_filter=custom_filter,
        )

    except Exception as exc:
        raise GeoServiceError(
            "local_osm_failed",
            f"Local OSM search failed: {exc}",
        ) from exc

    if pois is None or len(pois) == 0:
        return {
            "items": [],
            "data_timestamp": None,
            "provider": "local_osm",
        }

    pois = pois.to_crs(epsg=4326)

    items = []
    seen = set()

    for _, row in pois.iterrows():
        geometry = row.get("geometry")

        if geometry is None or geometry.is_empty:
            continue

        try:
            if geometry.geom_type == "Point":
                point = geometry
            else:
                point = geometry.centroid

            poi_lat = float(point.y)
            poi_lng = float(point.x)

        except Exception:
            continue

        category = _detect_category(row, categories)

        if category is None:
            continue

        osm_id = row.get("id")
        osm_type = row.get("osm_type") or "local"

        identity = f"{osm_type}:{osm_id}"

        if identity in seen:
            continue

        seen.add(identity)

        name = (
            row.get("name:ar")
            or row.get("name")
            or f"مرفق {osm_id}"
        )

        items.append(
            {
                "source_ref": f"osm:{identity}",
                "name": str(name),
                "lat": poi_lat,
                "lng": poi_lng,
                "category": category,
                "osm_type": str(osm_type),
                "osm_id": str(osm_id) if osm_id is not None else None,
            }
        )

        if len(items) >= settings.ASAR_GEO_MAX_RESULTS:
            break

    return {
        "items": items,
        "data_timestamp": None,
        "provider": "local_osm",
    }