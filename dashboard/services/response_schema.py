import json


POINT_PROPERTIES = {
    "name": {"type": "string"},
    "lat": {"type": ["number", "null"]},
    "lng": {"type": ["number", "null"]},
    "role": {
        "type": "string",
        "enum": ["anchor", "answer", "alternative", "intermediate", "facility"],
    },
    "category": {"type": ["string", "null"]},
    "distance_km": {"type": ["number", "null"]},
    "step": {"type": ["integer", "null"]},
    "score": {"type": ["number", "null"]},
    "source_ref": {"type": ["string", "null"]},
    "source": {
        "type": "string",
        "enum": ["user", "nominatim", "openstreetmap", "derived"],
    },
    "osm_type": {
        "type": ["string", "null"],
        "enum": ["node", "way", "relation", None],
    },
    "osm_id": {"type": ["string", "null"]},
}

POINT_SCHEMA = {
    "type": "object",
    "properties": POINT_PROPERTIES,
    "required": list(POINT_PROPERTIES),
    "additionalProperties": False,
}

ASAR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "insufficient_information", "needs_clarification"],
        },
        "task_type": {
            "type": "string",
            "enum": [
                "nearest_category",
                "cardinal_direction",
                "within_radius_yes_no",
                "closer_of_two",
                "count_within_radius",
                "nearest_of_two_categories",
                "two_hop_nearest",
                "spatial_multi_constraint",
                "general_spatial",
            ],
        },
        "reasoning_depth": {"type": "string", "enum": ["concise", "deep"]},
        "data_mode": {"type": "string", "enum": ["user_provided", "osm_assisted"]},
        "answer": {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number", "boolean", "null"]},
                "text": {"type": "string"},
            },
            "required": ["value", "text"],
            "additionalProperties": False,
        },
        "explanation": {"type": "string"},
        "visualization": {
            "type": "string",
            "enum": [
                "none",
                "nearest",
                "direction",
                "radius_yes_no",
                "comparison",
                "count",
                "category_comparison",
                "two_hop",
                "multi_constraint",
                "service_coverage",
            ],
        },
        "anchor": {"anyOf": [POINT_SCHEMA, {"type": "null"}]},
        "locations": {"type": "array", "items": POINT_SCHEMA},
        "metrics": {
            "type": "object",
            "properties": {
                "distance_km": {"type": ["number", "null"]},
                "radius_km": {"type": ["number", "null"]},
                "nearest_distance_km": {"type": ["number", "null"]},
                "count": {"type": ["integer", "null"]},
                "direction": {"type": ["string", "null"]},
                "suitability": {"type": ["number", "null"]},
            },
            "required": [
                "distance_km",
                "radius_km",
                "nearest_distance_km",
                "count",
                "direction",
                "suitability",
            ],
            "additionalProperties": False,
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": ["string", "number", "boolean"]},
                },
                "required": ["label", "value"],
                "additionalProperties": False,
            },
        },
        "comparison": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": ["string", "number", "boolean"]},
                    "score": {"type": ["number", "null"]},
                },
                "required": ["label", "value", "score"],
                "additionalProperties": False,
            },
        },
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "passed": {"type": ["boolean", "null"]},
                },
                "required": ["label", "passed"],
                "additionalProperties": False,
            },
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string"},
                    "attribution": {"type": "string"},
                    "url": {"type": "string"},
                    "retrieved_at": {"type": "string"},
                    "data_timestamp": {"type": ["string", "null"]},
                },
                "required": ["provider", "attribution", "url", "retrieved_at", "data_timestamp"],
                "additionalProperties": False,
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "status",
        "task_type",
        "reasoning_depth",
        "data_mode",
        "answer",
        "explanation",
        "visualization",
        "anchor",
        "locations",
        "metrics",
        "evidence",
        "comparison",
        "constraints",
        "sources",
        "limitations",
    ],
    "additionalProperties": False,
}

ASAR_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "asar_spatial_response",
    "strict": True,
    "schema": ASAR_RESPONSE_SCHEMA,
}


class InvalidModelResponse(ValueError):
    pass


def parse_model_response(output_text: str) -> dict:
    try:
        result = json.loads(output_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise InvalidModelResponse("Model output was not valid JSON.") from exc

    if not isinstance(result, dict):
        raise InvalidModelResponse("Model output must be a JSON object.")

    required = set(ASAR_RESPONSE_SCHEMA["required"])
    if not required.issubset(result):
        raise InvalidModelResponse("Model output was missing required fields.")
    if result["status"] not in {
    "answered",
    "insufficient_information",
    "needs_clarification",
    }:
        raise InvalidModelResponse("Model output contained an invalid status.")        
    if result["reasoning_depth"] not in {"concise", "deep"}:
        raise InvalidModelResponse("Model output contained an invalid reasoning depth.")
    if not isinstance(result["answer"], dict) or not isinstance(result["answer"].get("text"), str):
        raise InvalidModelResponse("Model output contained an invalid answer.")

    # Keep the existing frontend contract while avoiding a model-facing field
    # that could be confused with hidden chain-of-thought.
    result["reasoning"] = result.pop("explanation")
    return result
