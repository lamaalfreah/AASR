"""Deploy the unchanged cache-only Qwen class; no research entrypoint is run.

modal deploy -m dashboard.modal_production
"""
import modal
from language.modal_long_backend import QwenLong
app = modal.App('aasr-production-long')
from language.modal_long_backend import app as frozen_app
app.include(frozen_app)
