from .agents.logger import log_result
from .agents.verifier import verify_result

from .adaptive_runtime import (
    AdaptiveConfigurationError,
    AdaptiveInputError,
    AdaptiveRuntimeError,
    analyze,
)


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


class ModelInputError(ModelServiceError):
    code = "spatial_input_incomplete"
    status_code = 422
    public_message = "تعذر تحديد العناصر المكانية في السؤال بوضوح."


def _is_timeout_error(exc: Exception) -> bool:
    return exc.__class__.__name__ in {
        "APITimeoutError",
        "ReadTimeout",
        "TimeoutError",
        "TimeoutException",
    }


def analyze_spatial_question(question: str) -> dict:
    """
    Main Django integration point.

    Flow:
    Question
        -> Context Builder
        -> BiGRU Short
        -> Router V2
        -> SHORT or Qwen3-4B LONG
        -> Structured Query
        -> Geo Engine
        -> Frontend response
    """

    try:
        result = analyze(question)

    except AdaptiveConfigurationError as exc:
        raise ModelConfigurationError(str(exc)) from exc

    except AdaptiveInputError as exc:
        raise ModelInputError(str(exc)) from exc

    except AdaptiveRuntimeError as exc:
        raise ModelServiceError(str(exc)) from exc

    except Exception as exc:
        if _is_timeout_error(exc):
            raise ModelTimeoutError(
                "Adaptive runtime timed out."
            ) from exc

        raise ModelServiceError(
            "Adaptive runtime failed."
        ) from exc

    result["question"] = question

    result = verify_result(result)

    log_result(result)

    return result