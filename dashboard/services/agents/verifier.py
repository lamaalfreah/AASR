
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
 
 
def verify_result(result: dict) -> dict:
    """Adds a `verification` block to result. Never raises — verifier failures
    should never break the API response."""
    checks = []
    anchor = result.get("anchor")
    locations = result.get("locations", [])
    answer_point = _find(locations, "answer")
 
    try:
        if anchor and answer_point:
            real_km = haversine_km(anchor["lat"], anchor["lng"], answer_point["lat"], answer_point["lng"])
 
            claimed_km = result.get("metrics", {}).get("distance_km") or answer_point.get("distance_km")
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
 
        if result.get("task_type") == "count_within_radius":
            radius = result.get("metrics", {}).get("radius_km")
            claimed_count = result.get("metrics", {}).get("count")
            if anchor and radius is not None and claimed_count is not None:
                real_count = sum(
                    1 for p in locations
                    if p.get("role") == "answer" and haversine_km(anchor["lat"], anchor["lng"], p["lat"], p["lng"]) <= radius
                )
                checks.append({
                    "check": "count_within_radius",
                    "claimed": claimed_count,
                    "computed": real_count,
                    "passed": claimed_count == real_count,
                })
    except (KeyError, TypeError, ValueError) as exc:
        checks.append({"check": "error", "passed": False, "detail": str(exc)})
 
    all_passed = all(c["passed"] for c in checks) if checks else None
    result["verification"] = {"passed": all_passed, "checks": checks}
    return result
 
