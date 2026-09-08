"""Logging agent: persists every question/answer/verification to the DB.

Runs after the verifier. Never raises — a logging failure must not break
the API response the user sees.
"""
import logging

logger = logging.getLogger(__name__)


def log_result(result: dict):
    from dashboard.models import QueryLog  # local import: avoids Django app-loading issues

    try:
        verification = result.get("verification") or {}
        QueryLog.objects.create(
            question=result.get("question", ""),
            task_type=result.get("task_type", ""),
            result_json=result,
            verification_passed=verification.get("passed"),
            verification_json=verification,
        )
    except Exception:
        logger.exception("QueryLog write failed")