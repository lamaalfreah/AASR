"""Deterministic distance, coverage, and candidate calculations."""

from __future__ import annotations

import math

from django.conf import settings

from dashboard.services.agents.verifier import bearing_cardinal, haversine_km


def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    return haversine_km(lat1, lng1, lat2, lng2)


def calculate_direction(origin: dict, target: dict) -> str:
    return bearing_cardinal(origin["lat"], origin["lng"], target["lat"], target["lng"])


def distance_between(origin: dict, target: dict) -> float:
    return round(
        calculate_distance(origin["lat"], origin["lng"], target["lat"], target["lng"]),
        3,
    )


def is_within_radius(origin: dict, target: dict, radius_m: int) -> bool:
    return distance_between(origin, target) <= radius_m / 1000


def rank_by_distance(points: list[dict], origin: dict) -> list[dict]:
    ranked = [
        {**point, "distance_km": distance_between(origin, point)}
        for point in points
    ]
    return sorted(ranked, key=lambda point: (point["distance_km"], point.get("name", "")))


def nearest_for_each_category(
    points: list[dict], origin: dict, categories: list[str]
) -> list[dict]:
    nearest = []
    for category in categories:
        ranked = rank_by_distance(
            [point for point in points if point.get("category") == category], origin
        )
        if ranked:
            nearest.append(ranked[0])
    return nearest


def with_anchor_distances(points: list[dict], anchor: dict) -> list[dict]:
    return [
        {
            **point,
            "distance_km": round(
                calculate_distance(anchor["lat"], anchor["lng"], point["lat"], point["lng"]),
                3,
            ),
        }
        for point in points
    ]


def find_nearest(points: list[dict], origin: dict) -> dict | None:
    if not points:
        return None
    return min(
        points,
        key=lambda point: calculate_distance(
            origin["lat"], origin["lng"], point["lat"], point["lng"]
        ),
    )


def count_within_radius(points: list[dict], origin: dict, radius_m: int) -> int:
    return sum(
        calculate_distance(origin["lat"], origin["lng"], point["lat"], point["lng"])
        <= radius_m / 1000
        for point in points
    )


def _destination_point(lat: float, lng: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
    angular = distance_km / 6371.0088
    bearing = math.radians(bearing_deg)
    lat1, lng1 = math.radians(lat), math.radians(lng)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lng2 = lng1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), ((math.degrees(lng2) + 540) % 360) - 180


def _weights(has_supporting_pois: bool) -> tuple[float, float, float]:
    raw = [
        max(0.0, settings.ASAR_CANDIDATE_GAP_WEIGHT),
        max(0.0, settings.ASAR_CANDIDATE_ANCHOR_WEIGHT),
        max(0.0, settings.ASAR_CANDIDATE_SUPPORT_WEIGHT if has_supporting_pois else 0.0),
    ]
    total = sum(raw)
    if total <= 0:
        return 0.625, 0.375, 0.0
    return tuple(value / total for value in raw)


def generate_service_candidates(
    anchor: dict,
    facilities: list[dict],
    supporting_pois: list[dict],
    radius_m: int,
    candidate_count: int,
) -> list[dict]:
    """Rank reproducible ring samples as analytical service-coverage points."""
    radius_km = radius_m / 1000
    gap_weight, anchor_weight, support_weight = _weights(bool(supporting_pois))
    samples = []
    for ring_fraction in (0.35, 0.65):
        for bearing in range(0, 360, 45):
            lat, lng = _destination_point(
                anchor["lat"], anchor["lng"], radius_km * ring_fraction, bearing
            )
            point = {"lat": lat, "lng": lng}
            distance_to_anchor = calculate_distance(
                anchor["lat"], anchor["lng"], lat, lng
            )
            nearest_facility = find_nearest(facilities, point)
            nearest_facility_km = (
                calculate_distance(lat, lng, nearest_facility["lat"], nearest_facility["lng"])
                if nearest_facility
                else radius_km
            )
            nearest_support = find_nearest(supporting_pois, point)
            nearest_support_km = (
                calculate_distance(lat, lng, nearest_support["lat"], nearest_support["lng"])
                if nearest_support
                else None
            )
            gap_score = min(nearest_facility_km / radius_km, 1.0)
            anchor_score = max(0.0, 1.0 - distance_to_anchor / radius_km)
            support_score = (
                max(0.0, 1.0 - nearest_support_km / radius_km)
                if nearest_support_km is not None
                else 0.0
            )
            score = (
                gap_weight * gap_score
                + anchor_weight * anchor_score
                + support_weight * support_score
            )
            samples.append({
                "lat": round(lat, 7),
                "lng": round(lng, 7),
                "distance_km": round(distance_to_anchor, 3),
                "nearest_facility_km": round(nearest_facility_km, 3),
                "nearest_support_km": (
                    round(nearest_support_km, 3) if nearest_support_km is not None else None
                ),
                "score": round(score, 4),
                "bearing": bearing,
            })
    samples.sort(key=lambda item: (-item["score"], item["distance_km"], item["bearing"]))
    selected = samples[: max(1, min(int(candidate_count), 5))]
    return [
        {
            **point,
            "source_ref": f"candidate:{index}",
            "name": f"نقطة تغطية تحليلية {index}",
            "category": "service_coverage_candidate",
        }
        for index, point in enumerate(selected, 1)
    ]


def candidate_method() -> dict:
    return {
        "method": "deterministic_two_ring_distance_coverage",
        "weights": {
            "distance_from_existing_facilities": settings.ASAR_CANDIDATE_GAP_WEIGHT,
            "closeness_to_reference": settings.ASAR_CANDIDATE_ANCHOR_WEIGHT,
            "closeness_to_supporting_pois": settings.ASAR_CANDIDATE_SUPPORT_WEIGHT,
        },
        "weights_note": (
            "الأوزان خيارات تصميم أولية وليست متعلمة أو محسنة علميًا."
        ),
    }
