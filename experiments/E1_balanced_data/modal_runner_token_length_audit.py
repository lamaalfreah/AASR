"""
Modal runner for AASR Experiment E1 token-length audit.

Purpose
-------
Run token_length_audit.py inside Modal using:

- Qwen3 tokenizer
- private Hugging Face access
- shared asar-artifacts volume
- shared Hugging Face cache
- CPU only

No model training is performed here.
No GPU is required.

Why Jinja2 is required
----------------------
Qwen3 uses a chat template. The Transformers
apply_chat_template() function needs Jinja2 to render it.

Outputs are saved under:

    /workspace/artifacts/experiments/E1_balanced_data

Expected output files:
    token_length_audit.json
    token_length_audit.csv
"""

import modal


# ---------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------

app = modal.App(
    "aasr-e1-token-length-audit"
)


# ---------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------

image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .pip_install(
        "requests>=2.32",
        "transformers==4.57.1",
        "sentencepiece==0.2.1",
        "jinja2>=3.1",
    )
    .add_local_file(
        "experiments/E1_balanced_data/token_length_audit.py",
        remote_path="/workspace/token_length_audit.py",
    )
)


# ---------------------------------------------------------------------
# Shared Modal storage
# ---------------------------------------------------------------------

artifacts = modal.Volume.from_name(
    "asar-artifacts",
    create_if_missing=True,
)


cache = modal.Volume.from_name(
    "asar-hf-cache",
    create_if_missing=True,
)


# ---------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------

@app.function(
    image=image,

    # CPU only.
    # No gpu= parameter is intentionally used.
    timeout=60 * 60,

    volumes={
        "/workspace/artifacts":
            artifacts,

        "/root/.cache/huggingface":
            cache,
    },

    secrets=[
        modal.Secret.from_name(
            "huggingface-secret"
        )
    ],
)
def run_audit():
    """
    Execute the E1 token-length audit inside Modal.

    The script reads:
        balanced_short_train.jsonl
        balanced_long_train.jsonl

    from the shared asar-artifacts volume.

    It also loads the original private TRAIN dataset
    from Hugging Face.

    The audit uses the Qwen3 tokenizer and the same
    chat-template format used by the current AASR
    training pipeline.

    No model weights are loaded.
    No training takes place.
    """

    import os
    import subprocess

    # --------------------------------------------------
    # E1 output directory inside shared Modal volume
    # --------------------------------------------------

    os.environ["E1_OUTPUT_DIR"] = (
        "/workspace/artifacts/"
        "experiments/E1_balanced_data"
    )

    # --------------------------------------------------
    # Reuse shared Hugging Face cache
    # --------------------------------------------------

    os.environ["HF_HOME"] = (
        "/root/.cache/huggingface"
    )

    # --------------------------------------------------
    # Run token-length audit
    # --------------------------------------------------

    subprocess.run(
        [
            "python",
            "-u",
            "/workspace/token_length_audit.py",
        ],
        check=True,
        env=os.environ.copy(),
    )

    # --------------------------------------------------
    # Persist outputs and cache
    # --------------------------------------------------

    artifacts.commit()
    cache.commit()


# ---------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """
    Launch the E1 token-length audit on Modal.
    """

    run_audit.remote()



# results show that balancing substantially reduces Short training size but preserves the original token-length distribution. 
# The current Short prompt budget truncates approximately 60.5% of balanced Easy examples, while the Long prompt budget truncates approximately 19.5% of balanced Hard examples.