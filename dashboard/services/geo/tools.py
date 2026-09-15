"""One grounded geographic tool covering all supported ASAR task types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import quote

from django.conf import settings

from .analysis import (
    calculate_direction,
    candidate_method,
    distance_between,
    generate_service_candidates,
    is_within_radius,
    nearest_for_each_category,
    rank_by_distance,
)
from .client import GeoServiceError, OSMClient, POI_FILTERS


SUPPORTED_CATEGORIES = sorted(POI_FILTERS)
SUPPORTED_OPERATIONS = [
    "nearest_category",
    "cardinal_direction",
    "within_radius_yes_no",
    "closer_of_two",
    "count_within_radius",
    "nearest_of_two_categories",
    "two_hop_nearest",
    "spatial_multi_constraint",
]

ANALYZE_SPATIAL_QUERY_TOOL = {
    "type": "function",
    "name": "analyze_spatial_query",
    "description": (
        "Resolve real-world named places and deterministically answer one of the eight "
        "ASAR spatial operations. It performs all required Nominatim, Overpass, distance, "
        "direction, radius, comparison, two-hop, and service-gap work in one tool call."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": SUPPORTED_OPERATIONS},
            "reference_place": {"type": ["string", "null"], "maxLength": 300},
            "target_place": {"type": ["string", "null"], "maxLength": 300},
            "candidate_places": {
                "type": ["array", "null"],
                "items": {"type": "string", "minLength": 2, "maxLength": 300},
                "maxItems": 2,
            },
            "category": {
                "type": ["string", "null"],
                "enum": [*SUPPORTED_CATEGORIES, None],
            },
            "supporting_categories": {
                "type": ["array", "null"],
                "items": {"type": "string", "enum": SUPPORTED_CATEGORIES},
                "maxItems": 5,
            },
            "categories": {
                "type": ["array", "null"],
                "items": {
                    "type": "string",
                    "enum": SUPPORTED_CATEGORIES,
                },
                "maxItems": 5,
            },
            "radius_m": {
                "type": ["integer", "null"],
                "minimum": 100,
                "maximum": settings.ASAR_GEO_MAX_RADIUS_M,
            },
            "candidate_count": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
            "country_code": {"type": ["string", "null"], "minLength": 2, "maxLength": 2},
        },
        "required": [
            "operation",
            "reference_place",
            "target_place",
            "candidate_places",
            "category",
            "categories",
            "supporting_categories",
            "radius_m",
            "candidate_count",
            "country_code",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


class GroundingError(ValueError):
    pass


@dataclass
class GeoToolContext:
    called: bool = False
    result: dict | None = None
    registry: dict[str, dict] = field(default_factory=dict)


def _safe_failure(code: str, message: str) -> dict:
    return {
        "ok": False,
        "error": {"code": code, "message": message},
        "sources": [],
        "limitations": [
            "تعذر الحصول على بيانات جغرافية عامة موثوقة لهذه المحاولة؛ لا تُنشأ إحداثيات بديلة."
        ],
    }


def _clean_name(value) -> str | None:
    if value is None:
        return None
    name = " ".join(str(value).split())
    return name if 2 <= len(name) <= 300 else None


def _clean_list(value, *, maximum: int) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GeoServiceError("invalid_tool_arguments", "A list argument was invalid.")
    return list(dict.fromkeys(value))[:maximum]


def _validate_arguments(arguments: dict) -> dict:
    if not isinstance(arguments, dict):
        raise GeoServiceError("invalid_tool_arguments", "Tool arguments must be an object.")
    operation = arguments.get("operation")
    if operation not in SUPPORTED_OPERATIONS:
        raise GeoServiceError("invalid_tool_arguments", "Unsupported spatial operation.")

    reference = _clean_name(arguments.get("reference_place"))
    target = _clean_name(arguments.get("target_place"))
    candidates = [_clean_name(item) for item in _clean_list(arguments.get("candidate_places"), maximum=2)]
    categories = _clean_list(arguments.get("categories"), maximum=5)
    supporting = _clean_list(arguments.get("supporting_categories"), maximum=5)
    category = arguments.get("category")
    if not reference or any(item is None for item in candidates):
        raise GeoServiceError("invalid_tool_arguments", "A valid reference place is required.")
    if category is not None and category not in SUPPORTED_CATEGORIES:
        raise GeoServiceError("unsupported_category", "Unsupported POI category.")
    if set(categories + supporting) - set(SUPPORTED_CATEGORIES):
        raise GeoServiceError("unsupported_category", "Unsupported POI category.")
    supporting = [item for item in supporting if item not in categories]

    radius_value = arguments.get("radius_m")
    candidate_value = arguments.get("candidate_count")
    try:
        radius_m = (
            min(5000, settings.ASAR_GEO_MAX_RADIUS_M)
            if radius_value is None else int(radius_value)
        )
        candidate_count = 3 if candidate_value is None else int(candidate_value)
    except (TypeError, ValueError) as exc:
        raise GeoServiceError("invalid_tool_arguments", "Invalid numeric arguments.") from exc
    if not 100 <= radius_m <= settings.ASAR_GEO_MAX_RADIUS_M or not 1 <= candidate_count <= 5:
        raise GeoServiceError("invalid_tool_arguments", "Invalid radius or candidate count.")

    country_code = arguments.get("country_code")
    country_code = str(country_code).strip().lower() if country_code else None
    if country_code and (len(country_code) != 2 or not country_code.isalpha()):
        raise GeoServiceError("invalid_tool_arguments", "Invalid country code.")

    if operation in {"cardinal_direction", "within_radius_yes_no"} and not target:
        raise GeoServiceError("invalid_tool_arguments", "A target place is required.")
    if operation == "closer_of_two" and len(candidates) != 2:
        raise GeoServiceError("invalid_tool_arguments", "Exactly two candidate places are required.")
    
    if operation == "nearest_category":
        if not category and not categories:
            raise GeoServiceError(
                "invalid_tool_arguments",
                "A POI category is required."
            )

        if not category:
            category = categories[0]

    if operation == "count_within_radius":
        if not categories and category:
            categories = [category]

        if not categories:
            raise GeoServiceError(
                "invalid_tool_arguments",
                "At least one POI category is required."
            )

    if operation in {"nearest_of_two_categories", "two_hop_nearest"}:
        if len(categories) != 2:
            raise GeoServiceError(
                "invalid_tool_arguments",
                "Exactly two distinct categories are required."
            )

    if operation == "spatial_multi_constraint":
        if not categories:
            raise GeoServiceError(
                "invalid_tool_arguments",
                "At least one service category is required."
            )

    if operation in {"within_radius_yes_no", "count_within_radius"} and radius_value is None:
        raise GeoServiceError("invalid_tool_arguments", "This operation requires an explicit radius.")

    return {
        "operation": operation,
        "reference_place": reference,
        "target_place": target,
        "candidate_places": candidates,
        "category": category,
        "categories": categories,
        "supporting_categories": supporting,
        "radius_m": radius_m,
        "candidate_count": candidate_count,
        "country_code": country_code or settings.ASAR_GEO_DEFAULT_COUNTRY_CODE or None,
    }


def _geocode_is_ambiguous(matches: list[dict]) -> bool:
    if len(matches) < 2:
        return False
    first, second = matches[0], matches[1]
    importance_gap = first.get("importance", 0) - second.get("importance", 0)
    return distance_between(first, second) > 25 and importance_gap < 0.10


def _point(item: dict, *, role: str, source: str, category: str | None = None, step=None) -> dict:
    return {
        "name": item["name"],
        "lat": item["lat"],
        "lng": item["lng"],
        "role": role,
        "category": category if category is not None else item.get("category"),
        "distance_km": item.get("distance_km"),
        "step": step,
        "score": item.get("score"),
        "source_ref": item["source_ref"],
        "source": source,
        "osm_type": item.get("osm_type"),
        "osm_id": item.get("osm_id"),
    }

def _geocode(
    client: OSMClient,
    name: str,
    country_code: str | None,
    role: str,
) -> dict:

    queries = [
        name,
        f"{name}, الرياض",
        f"{name}, المملكة العربية السعودية",
    ]

    seen = set()

    for query in queries:
        if query in seen:
            continue

        seen.add(query)

        matches = client.geocode_place(
            query,
            country_code,
        )

        if matches:
            if _geocode_is_ambiguous(matches):
                continue

            return _point(
                matches[0],
                role=role,
                source="nominatim",
            )

    # Google fallback
    google_matches = client.google_geocode_place(name)

    if google_matches:
        return _point(
            google_matches[0],
            role=role,
            source="google",
        )

    raise GeoServiceError(
        "place_not_found",
        f"Named place was not found: {name}",
    )

def _poi_points(client: OSMClient, origin: dict, categories: list[str], radius_m: int) -> tuple[list[dict], str | None]:
    response = client.search_nearby_pois(origin["lat"], origin["lng"], categories, radius_m)
    ranked = rank_by_distance(response["items"], origin)
    points = [
        _point(item, role="facility", source="openstreetmap")
        for item in ranked
        if item["distance_km"] <= radius_m / 1000
    ]
    return points, response.get("data_timestamp")


def _metrics(**overrides) -> dict:
    metrics = {
        "distance_km": None,
        "radius_km": None,
        "nearest_distance_km": None,
        "count": None,
        "direction": None,
        "suitability": None,
    }
    metrics.update(overrides)
    return metrics


def _base_sources(args: dict, *, overpass_used: bool, data_timestamp: str | None) -> list[dict]:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    sources = [{
        "provider": "Nominatim / OpenStreetMap",
        "attribution": "© OpenStreetMap contributors",
        "url": f"{settings.ASAR_NOMINATIM_URL}/ui/search.html?q={quote(args['reference_place'])}",
        "retrieved_at": retrieved_at,
        "data_timestamp": None,
    }]
    if overpass_used:
        sources.append({
            "provider": "Overpass API / OpenStreetMap",
            "attribution": "© OpenStreetMap contributors",
            "url": "https://www.openstreetmap.org/copyright",
            "retrieved_at": retrieved_at,
            "data_timestamp": data_timestamp,
        })
    return sources


def _result(
    args: dict,
    anchor: dict,
    locations: list[dict],
    *,
    answer_value,
    visualization: str,
    metrics: dict,
    evidence: list[dict],
    comparison: list[dict] | None = None,
    constraints: list[dict] | None = None,
    overpass_used: bool = False,
    data_timestamp: str | None = None,
    limitations: list[str] | None = None,
) -> dict:
    return {
        "ok": True,
        "operation": args["operation"],
        "query": args,
        "answer_value": answer_value,
        "visualization": visualization,
        "anchor": anchor,
        "locations": locations,
        "metrics": metrics,
        "evidence": evidence,
        "comparison": comparison or [],
        "constraints": constraints or [],
        "sources": _base_sources(args, overpass_used=overpass_used, data_timestamp=data_timestamp),
        "limitations": limitations or [],
    }


def analyze_spatial_query(arguments: dict, client: OSMClient | None = None) -> dict:
    args = _validate_arguments(arguments)
    owns_client = client is None
    client = client or OSMClient()
    try:
        return _run_operation(args, client)
    finally:
        if owns_client:
            client.close()


def _run_operation(args: dict, client: OSMClient) -> dict:
    operation = args["operation"]
    anchor = _geocode(client, args["reference_place"], args["country_code"], "anchor")

    if operation == "cardinal_direction":
        target = _geocode(client, args["target_place"], args["country_code"], "answer")
        target["distance_km"] = distance_between(anchor, target)
        direction = calculate_direction(anchor, target)
        return _result(
            args, anchor, [anchor, target], answer_value=direction, visualization="direction",
            metrics=_metrics(distance_km=target["distance_km"], direction=direction),
            evidence=[
                {"label": "المسافة المحسوبة (كم)", "value": target["distance_km"]},
                {"label": "الاتجاه المحسوب", "value": direction},
            ],
        )

    if operation == "within_radius_yes_no":
        target = _geocode(client, args["target_place"], args["country_code"], "answer")
        target["distance_km"] = distance_between(anchor, target)
        within = is_within_radius(anchor, target, args["radius_m"])
        return _result(
            args, anchor, [anchor, target], answer_value=within, visualization="radius_yes_no",
            metrics=_metrics(distance_km=target["distance_km"], radius_km=args["radius_m"] / 1000),
            evidence=[{"label": "المسافة المحسوبة (كم)", "value": target["distance_km"]}],
            constraints=[{"label": "داخل نصف القطر المحدد", "passed": within}],
        )

    if operation == "closer_of_two":
        named = [
            _geocode(client, name, args["country_code"], "alternative")
            for name in args["candidate_places"]
        ]
        ranked = rank_by_distance(named, anchor)
        ranked[0]["role"] = "answer"
        comparison = [
            {"label": point["name"], "value": point["distance_km"], "score": None}
            for point in ranked
        ]
        return _result(
            args, anchor, [anchor, *ranked], answer_value=ranked[0]["name"], visualization="comparison",
            metrics=_metrics(distance_km=ranked[0]["distance_km"]),
            evidence=[{"label": "الأقرب بحساب المسافة", "value": ranked[0]["name"]}],
            comparison=comparison,
        )

    if operation in {"nearest_category", "count_within_radius"}:
        if operation == "nearest_category":
            search_categories = [args["category"]]
        else:
            search_categories = args["categories"]

        facilities, timestamp = _poi_points(
            client,
            anchor,
            search_categories,
            args["radius_m"],
        )

        if operation == "nearest_category":
            if not facilities:
                raise GeoServiceError(
                    "poi_not_found",
                    "No matching POI was found in the radius."
                )

            facilities[0]["role"] = "answer"

            return _result(
                args,
                anchor,
                [anchor, *facilities],
                answer_value=facilities[0]["name"],
                visualization="nearest",
                metrics=_metrics(
                    distance_km=facilities[0]["distance_km"],
                    radius_km=args["radius_m"] / 1000,
                    nearest_distance_km=facilities[0]["distance_km"],
                    count=len(facilities),
                ),
                evidence=[
                    {
                        "label": "عدد المرافق المطابقة",
                        "value": len(facilities),
                    },
                    {
                        "label": "مسافة أقرب مرفق (كم)",
                        "value": facilities[0]["distance_km"],
                    },
                ],
                overpass_used=True,
                data_timestamp=timestamp,
                limitations=[
                    "قد تكون بيانات OpenStreetMap غير مكتملة أو غير محدثة لجميع المرافق."
                ],
            )

        # هذا خاص بـ count_within_radius
        nearest = facilities[0] if facilities else None

        return _result(
            args,
            anchor,
            [anchor, *facilities],
            answer_value=len(facilities),
            visualization="count",
            metrics=_metrics(
                radius_km=args["radius_m"] / 1000,
                count=len(facilities),
                nearest_distance_km=nearest["distance_km"] if nearest else None,
            ),
            evidence=[
                {
                    "label": "عدد المرافق داخل النطاق",
                    "value": len(facilities),
                },
                {
                    "label": "أقرب مرفق",
                    "value": nearest["name"] if nearest else None,
                },
                {
                    "label": "مسافة أقرب مرفق (كم)",
                    "value": nearest["distance_km"] if nearest else None,
                },
            ],
            overpass_used=True,
            data_timestamp=timestamp,
            limitations=[
                "قد تكون بيانات OpenStreetMap غير مكتملة أو غير محدثة لجميع المرافق."
            ],
        )

    if operation == "nearest_of_two_categories":
        facilities, timestamp = _poi_points(client, anchor, args["categories"], args["radius_m"])
        nearest = nearest_for_each_category(facilities, anchor, args["categories"])
        if len(nearest) != 2:
            raise GeoServiceError("poi_not_found", "Both requested categories require a nearby result.")
        nearest.sort(key=lambda point: (point["distance_km"], point["name"]))
        nearest[0]["role"] = "answer"
        nearest[1]["role"] = "alternative"
        return _result(
            args, anchor, [anchor, *nearest], answer_value=nearest[0]["name"],
            visualization="category_comparison",
            metrics=_metrics(
                distance_km=nearest[0]["distance_km"],
                radius_km=args["radius_m"] / 1000,
                nearest_distance_km=nearest[0]["distance_km"],
            ),
            evidence=[{"label": "الفئة الأقرب", "value": nearest[0]["category"]}],
            comparison=[
                {"label": point["category"], "value": point["name"], "score": None}
                for point in nearest
            ], overpass_used=True, data_timestamp=timestamp,
            limitations=["قد تكون بيانات OpenStreetMap غير مكتملة أو غير محدثة لجميع المرافق."],
        )

    if operation == "two_hop_nearest":
        first_group, first_timestamp = _poi_points(client, anchor, [args["categories"][0]], args["radius_m"])
        if not first_group:
            raise GeoServiceError("poi_not_found", "No first-hop POI was found.")
        first = first_group[0]
        first["role"], first["step"] = "intermediate", 1
        second_group, second_timestamp = _poi_points(
            client, first, [args["categories"][1]], args["radius_m"]
        )
        if not second_group:
            raise GeoServiceError("poi_not_found", "No second-hop POI was found.")
        second = second_group[0]
        second["role"], second["step"] = "answer", 2
        total = round(first["distance_km"] + second["distance_km"], 3)
        return _result(
            args, anchor, [anchor, first, second], answer_value=second["name"], visualization="two_hop",
            metrics=_metrics(
                distance_km=total,
                radius_km=args["radius_m"] / 1000,
                nearest_distance_km=second["distance_km"],
            ),
            evidence=[
                {"label": "مسافة الخطوة الأولى (كم)", "value": first["distance_km"]},
                {"label": "مسافة الخطوة الثانية (كم)", "value": second["distance_km"]},
            ], overpass_used=True, data_timestamp=second_timestamp or first_timestamp,
            limitations=["قد تكون بيانات OpenStreetMap غير مكتملة أو غير محدثة لجميع المرافق."],
        )

    all_categories = list(dict.fromkeys(args["categories"] + args["supporting_categories"]))
    all_points, timestamp = _poi_points(client, anchor, all_categories, args["radius_m"])
    facilities = [point for point in all_points if point["category"] in args["categories"]]
    supporting = [
        point for point in all_points if point["category"] in args["supporting_categories"]
    ]
    raw_candidates = generate_service_candidates(
        anchor, facilities, supporting, args["radius_m"], args["candidate_count"]
    )
    candidate_points = [
        _point(
            item,
            role="answer" if index == 0 else "alternative",
            source="derived",
            step=index + 1,
        )
        for index, item in enumerate(raw_candidates)
    ]
    for point in supporting:
        point["role"] = "intermediate"
    top = candidate_points[0]
    limitations = [
        "النتيجة توصية تحليلية لتغطية الخدمة اعتمادًا على بيانات OSM المتاحة، وليست موقع بناء معتمدًا.",
        "لا يشمل التحليل ملكية الأرض أو استعمالاتها أو السكان أو المرور أو التنظيمات أو صلاحية البناء.",
        candidate_method()["weights_note"],
    ]
    return _result(
        args, anchor, [anchor, *facilities, *supporting, *candidate_points],
        answer_value=top["name"], visualization="service_coverage",
        metrics=_metrics(
            distance_km=top["distance_km"], radius_km=args["radius_m"] / 1000,
            nearest_distance_km=facilities[0]["distance_km"] if facilities else None,
            count=len(facilities), suitability=top["score"],
        ),
        evidence=[
            {"label": "مرافق OSM داخل النطاق", "value": len(facilities)},
            {"label": "نصف قطر التحليل (كم)", "value": args["radius_m"] / 1000},
        ],
        comparison=[
            {
                "label": point["name"],
                "value": f"يبعد {point['distance_km']} كم عن المرجع",
                "score": point["score"],
            }
            for point in candidate_points
        ], overpass_used=True, data_timestamp=timestamp,
        limitations=limitations,
    )


def execute_geo_tool(name: str, arguments: dict, context: GeoToolContext) -> dict:
    if context.called:
        result = _safe_failure("tool_limit_reached", "لا يُسمح بأكثر من تحليل جغرافي واحد.")
    else:
        context.called = True
        if name != "analyze_spatial_query":
            result = _safe_failure("unknown_tool", "الأداة الجغرافية المطلوبة غير متاحة.")
        else:
            try:
                result = analyze_spatial_query(arguments)
            except GeoServiceError as exc:
                print("\n===== GEO ERROR =====")
                print("CODE:", exc.code)
                print("MESSAGE:", str(exc))
                print("===== END GEO ERROR =====\n")

                result = _safe_failure(
                    exc.code,
                    "تعذر استرجاع بيانات جغرافية موثوقة."
                )
            except Exception:
                result = _safe_failure("geo_unavailable", "تعذر إكمال التحليل الجغرافي.")
    context.result = result
    if result.get("ok"):
        points = [result["anchor"], *result["locations"]]
        context.registry = {point["source_ref"]: point for point in points}
    return result


def _canonical_point(model_point: dict, context: GeoToolContext) -> dict:
    source_ref = model_point.get("source_ref") if isinstance(model_point, dict) else None
    canonical = context.registry.get(source_ref)
    if canonical is None:
        raise GroundingError("Model returned a point that was not present in tool output.")
    return dict(canonical)


def ground_geo_response(result: dict, context: GeoToolContext) -> dict:
    """Allow model prose, but replace every structured geographic fact."""
    if not context.called:
        if any(
            point.get("source") != "user"
            for point in [result.get("anchor"), *result.get("locations", [])]
            if isinstance(point, dict)
        ):
            raise GroundingError("A direct answer claimed external geographic provenance.")
        result["data_mode"] = "user_provided"
        result["sources"] = []
        result["limitations"] = result.get("limitations", [])
        return result

    tool_result = context.result or _safe_failure("geo_unavailable", "لا توجد نتيجة أداة.")
    if not tool_result.get("ok"):
        return build_insufficient_result(tool_result, result.get("answer", {}).get("text"))

    for point in [result.get("anchor"), *result.get("locations", [])]:
        if point is not None:
            _canonical_point(point, context)

    result.update({
        "status": "answered",
        "task_type": tool_result["operation"],
        "data_mode": "osm_assisted",
        "visualization": tool_result["visualization"],
        "anchor": dict(tool_result["anchor"]),
        "locations": [dict(point) for point in tool_result["locations"]],
        "metrics": dict(tool_result["metrics"]),
        "evidence": list(tool_result["evidence"]),
        "comparison": list(tool_result["comparison"]),
        "constraints": list(tool_result["constraints"]),
        "sources": list(tool_result["sources"]),
        "limitations": list(tool_result["limitations"]),
    })
    result["answer"]["value"] = tool_result["answer_value"]
    return result


def build_insufficient_result(tool_result: dict | None = None, answer_text: str | None = None) -> dict:
    tool_result = tool_result or {}
    error = tool_result.get("error") or {}
    message = answer_text or error.get("message") or "لا تتوفر معلومات مكانية كافية لإجابة موثوقة."
    return {
        "status": "insufficient_information",
        "task_type": "general_spatial",
        "reasoning_depth": "concise",
        "data_mode": "osm_assisted" if tool_result else "user_provided",
        "answer": {"value": None, "text": message},
        "reasoning": "لم تتوفر بيانات جغرافية عامة موثوقة وكافية لهذه المحاولة.",
        "visualization": "none",
        "anchor": None,
        "locations": [],
        "metrics": _metrics(),
        "evidence": [],
        "comparison": [],
        "constraints": [],
        "sources": tool_result.get("sources", []),
        "limitations": tool_result.get("limitations", []),
    }
