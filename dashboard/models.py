from django.db import models


class QueryLog(models.Model):
    """One row per question sent through analyze_api.

    Kept as raw JSON for the AI-produced fields so the schema can evolve
    (new task types, new agents) without needing new migrations each time.
    """
    created_at = models.DateTimeField(auto_now_add=True)
    question = models.TextField()
    task_type = models.CharField(max_length=64, blank=True)

    result_json = models.JSONField()          # full response returned to the UI
    verification_passed = models.BooleanField(null=True)   # True/False/None (no checks ran)
    verification_json = models.JSONField(null=True, blank=True)

    user_feedback = models.CharField(
        max_length=16,
        choices=[("up", "up"), ("down", "down")],
        null=True, blank=True,
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.created_at:%Y-%m-%d %H:%M}] {self.task_type}: {self.question[:40]}"