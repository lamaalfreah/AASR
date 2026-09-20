import json
import traceback
from django.conf import settings

from .agents.logger import log_result
from .agents.verifier import verify_result
from .geo.tools import (
    ANALYZE_SPATIAL_QUERY_TOOL,
    GeoToolContext,
    GroundingError,
    build_insufficient_result,
    execute_geo_tool,
    ground_geo_response,
)
from .prompts import ADAPTIVE_SPATIAL_PROMPT
from .response_schema import ASAR_RESPONSE_FORMAT, InvalidModelResponse, parse_model_response


class ModelServiceError(Exception):
    code = "model_unavailable"
    status_code = 502
    public_message = "تعذر إكمال التحليل حاليًا. يرجى المحاولة مرة أخرى."


class ModelTimeoutError(ModelServiceError):
    code = "model_timeout"
    status_code = 504
    public_message = "استغرق التحليل وقتًا أطول من المتوقع. يرجى المحاولة مرة أخرى."


class ModelConfigurationError(ModelServiceError):
    code = "model_not_configured"
    status_code = 503
    public_message = "خدمة التحليل غير مهيأة بعد."


def _create_client():
    # Lazy import keeps Django management commands usable before dependencies
    # are installed and makes the API boundary straightforward to mock in tests.
    from openai import OpenAI

    return OpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=settings.ASAR_LLM_TIMEOUT_SECONDS,
    )


def _is_timeout_error(exc: Exception) -> bool:
    return exc.__class__.__name__ in {
        "APITimeoutError",
        "ReadTimeout",
        "TimeoutError",
        "TimeoutException",
    }


def _output_items(response) -> list:
    output = getattr(response, "output", []) or []
    items = []
    for item in output:
        if isinstance(item, dict):
            items.append(item)
        elif hasattr(item, "model_dump"):
            items.append(item.model_dump(mode="json"))
        elif getattr(item, "type", None) == "function_call":
            items.append({
                "type": "function_call",
                "name": getattr(item, "name", ""),
                "arguments": getattr(item, "arguments", "{}"),
                "call_id": getattr(item, "call_id", ""),
            })
    return items


def _field(item, name, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _model_request(client, *, input_value, include_tool: bool):
    kwargs = {
        "model": settings.ASAR_LLM_MODEL,
        "instructions": ADAPTIVE_SPATIAL_PROMPT,
        "input": input_value,
        "reasoning": {"effort": "medium"},
        "text": {"format": ASAR_RESPONSE_FORMAT, "verbosity": "low"},
        "max_output_tokens": 2500,
        "store": False,
    }
    if include_tool:
        kwargs.update({
            "tools": [ANALYZE_SPATIAL_QUERY_TOOL],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        })
    return client.responses.create(**kwargs)


def analyze_spatial_question(question: str) -> dict:
    """Analyze with one fixed model and no more than two model calls."""
    if not settings.OPENAI_API_KEY:
        raise ModelConfigurationError("OPENAI_API_KEY is not configured.")

    try:
        client = _create_client()
        context = GeoToolContext()
        first = _model_request(client, input_value=question, include_tool=True)
        tool_calls = [
            item for item in (getattr(first, "output", []) or [])
            if _field(item, "type") == "function_call"
        ]

        if not tool_calls:
            result = parse_model_response(first.output_text)
            result = ground_geo_response(result, context)
        elif len(tool_calls) > 1:
            result = build_insufficient_result(
                answer_text="تعذر تنفيذ الطلب لأن النموذج طلب أكثر من تحليل جغرافي واحد."
            )
        else:
            tool_call = tool_calls[0]
            try:
                arguments = json.loads(_field(tool_call, "arguments", "{}"))
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            tool_result = execute_geo_tool(_field(tool_call, "name", ""), arguments, context)
            second_input = [
                {
                    "type": "function_call",
                    "call_id": _field(tool_call, "call_id", ""),
                    "name": _field(tool_call, "name", ""),
                    "arguments": _field(tool_call, "arguments", "{}"),
                },
                {
                    "type": "function_call_output",
                    "call_id": _field(tool_call, "call_id", ""),
                    "output": json.dumps(tool_result, ensure_ascii=False),
                },
            ]
            second = _model_request(client, input_value=second_input, include_tool=False)
            try:
                result = parse_model_response(second.output_text)
                result = ground_geo_response(result, context)
            except (InvalidModelResponse, GroundingError):
                result = build_insufficient_result(tool_result)
    except (InvalidModelResponse, GroundingError) as exc:
        raise ModelServiceError("The model returned an invalid structured response.") from exc
    except ModelServiceError:
        raise
    except Exception as exc:
        print("\n===== ASAR REAL ERROR =====")
        print("TYPE:", type(exc).__name__)
        print("MESSAGE:", str(exc))
        traceback.print_exc()
        print("===== END ASAR ERROR =====\n")

        if _is_timeout_error(exc):
            raise ModelTimeoutError("The model request timed out.") from exc

        raise ModelServiceError("The model request failed.") from exc
    result["question"] = question
    result = verify_result(result)
    log_result(result)
    return result
