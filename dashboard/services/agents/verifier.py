
"""Verifier agent: cross-checks the model's spatial answer against real geometry.
 
Runs independently of the LLM. Uses the Haversine formula for distance and
bearing math for direction, then compares those ground-truth values against
whatever the model/builder claimed in `metrics` and `locations`.
"""
import math
 
EARTH_RADIUS_KM = 6371.0088
DISTANCE_TOLERANCE_KM = 0.15   # allowed drift before flagging a mismatch
DIRECTION_LABELS = ["شمال", "شمال شرق", "شرق", "جنوب شرق", "جنوب", "جنوب غرب", "غرب", "شمال غرب"]
 
 
def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
 
 
def bearing_cardinal(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    bearing_deg = (math.degrees(math.atan2(x, y)) + 360) % 360
    index = round(bearing_deg / 45) % 8
    return DIRECTION_LABELS[index]
 
 
def _find(locations, role):
    return next((p for p in locations if p.get("role") == role), None)


def _valid_point(point):
    if not isinstance(point, dict):
        return False
    lat, lng = point.get("lat"), point.get("lng")
    return (
        isinstance(lat, (int, float))
        and not isinstance(lat, bool)
        and math.isfinite(lat)
        and -90 <= lat <= 90
        and isinstance(lng, (int, float))
        and not isinstance(lng, bool)
        and math.isfinite(lng)
        and -180 <= lng <= 180
    )
 
 
def verify_result(result: dict) -> dict:
    """Adds a `verification` block to result. Never raises — verifier failures
    should never break the API response."""
    checks = []
    if result.get("status") != "answered":
        result["verification"] = {"passed": None, "checks": checks}
        return result

    anchor = result.get("anchor")
    locations = result.get("locations", [])
    answer_point = _find(locations, "answer")
    task_type = result.get("task_type")
 
    try:
        if (
            task_type != "two_hop_nearest"
            and _valid_point(anchor)
            and _valid_point(answer_point)
        ):
            real_km = haversine_km(anchor["lat"], anchor["lng"], answer_point["lat"], answer_point["lng"])
 
            claimed_km = result.get("metrics", {}).get("distance_km")
            if claimed_km is None:
                claimed_km = answer_point.get("distance_km")
            if claimed_km is not None:
                ok = abs(real_km - claimed_km) <= DISTANCE_TOLERANCE_KM
                checks.append({
                    "check": "distance_km",
                    "claimed": claimed_km,
                    "computed": round(real_km, 3),
                    "passed": ok,
                })
 
            claimed_dir = result.get("metrics", {}).get("direction")
            if claimed_dir:
                real_dir = bearing_cardinal(anchor["lat"], anchor["lng"], answer_point["lat"], answer_point["lng"])
                checks.append({
                    "check": "direction",
                    "claimed": claimed_dir,
                    "computed": real_dir,
                    "passed": claimed_dir == real_dir,
                })
 
        if task_type == "within_radius_yes_no" and _valid_point(anchor) and _valid_point(answer_point):
            radius = result.get("metrics", {}).get("radius_km")
            if radius is not None:
                computed = haversine_km(
                    anchor["lat"], anchor["lng"], answer_point["lat"], answer_point["lng"]
                ) <= radius
                claimed = result.get("answer", {}).get("value")
                checks.append({
                    "check": "within_radius",
                    "claimed": claimed,
                    "computed": computed,
                    "passed": claimed is computed,
                })

        if task_type in {"closer_of_two", "nearest_of_two_categories"} and _valid_point(anchor):
            compared = [
                point for point in locations
                if point.get("role") in {"answer", "alternative"} and _valid_point(point)
            ]
            if len(compared) == 2 and _valid_point(answer_point):
                computed_winner = min(
                    compared,
                    key=lambda point: haversine_km(
                        anchor["lat"], anchor["lng"], point["lat"], point["lng"]
                    ),
                )
                checks.append({
                    "check": "comparison_winner",
                    "claimed": answer_point.get("name"),
                    "computed": computed_winner.get("name"),
                    "passed": answer_point.get("source_ref") == computed_winner.get("source_ref"),
                })

        if task_type == "two_hop_nearest" and _valid_point(anchor):
            intermediate = _find(locations, "intermediate")
            if _valid_point(intermediate) and _valid_point(answer_point):
                hop_one = haversine_km(
                    anchor["lat"], anchor["lng"], intermediate["lat"], intermediate["lng"]
                )
                hop_two = haversine_km(
                    intermediate["lat"], intermediate["lng"],
                    answer_point["lat"], answer_point["lng"],
                )
                claimed_total = result.get("metrics", {}).get("distance_km")
                checks.append({
                    "check": "two_hop_distance",
                    "claimed": claimed_total,
                    "computed": round(hop_one + hop_two, 3),
                    "passed": (
                        claimed_total is not None
                        and abs(claimed_total - (hop_one + hop_two)) <= DISTANCE_TOLERANCE_KM * 2
                    ),
                })

        if task_type == "count_within_radius":
            radius = result.get("metrics", {}).get("radius_km")
            claimed_count = result.get("metrics", {}).get("count")
            if _valid_point(anchor) and radius is not None and claimed_count is not None:
                real_count = sum(
                    1 for p in locations
                    if p.get("role") == (
                        "facility" if result.get("data_mode") == "osm_assisted" else "answer"
                    )
                    and _valid_point(p)
                    and haversine_km(anchor["lat"], anchor["lng"], p["lat"], p["lng"]) <= radius
                )
                checks.append({
                    "check": "count_within_radius",
                    "claimed": claimed_count,
                    "computed": real_count,
                    "passed": claimed_count == real_count,
                })

        if result.get("data_mode") == "osm_assisted":
            grounded_points = [anchor, *locations]
            provenance_ok = all(
                not isinstance(point, dict)
                or (
                    point.get("source") in {"nominatim", "openstreetmap", "derived"}
                    and isinstance(point.get("source_ref"), str)
                    and bool(point["source_ref"])
                    and _valid_point(point)
                )
                for point in grounded_points
            )
            checks.append({"check": "geographic_provenance", "passed": provenance_ok})

            if result.get("visualization") == "service_coverage":
                claimed_count = result.get("metrics", {}).get("count")
                facility_count = sum(
                    point.get("role") == "facility" for point in locations
                    if isinstance(point, dict)
                )
                checks.append({
                    "check": "facility_count",
                    "claimed": claimed_count,
                    "computed": facility_count,
                    "passed": claimed_count == facility_count,
                })

                distance_checks = []
                if _valid_point(anchor):
                    for point in locations:
                        claimed = point.get("distance_km") if isinstance(point, dict) else None
                        if claimed is None or not _valid_point(point):
                            continue
                        computed = haversine_km(
                            anchor["lat"], anchor["lng"], point["lat"], point["lng"]
                        )
                        distance_checks.append(abs(computed - claimed) <= DISTANCE_TOLERANCE_KM)
                checks.append({
                    "check": "grounded_point_distances",
                    "passed": all(distance_checks) if distance_checks else True,
                })
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        checks.append({"check": "error", "passed": False, "detail": str(exc)})
 
    all_passed = all(c["passed"] for c in checks) if checks else None
    result["verification"] = {"passed": all_passed, "checks": checks}
    return result
 
