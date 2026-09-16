"""
Modal runner for AASR Experiment E1 balanced adapter training.

We train ONE adapter at a time to control GPU usage:

    short -> balanced Easy data
    long  -> balanced Hard data

The trained adapters are stored in the shared
asar-artifacts Modal volume.

GPU:
    NVIDIA L40S

The Hugging Face model cache is shared through
asar-hf-cache so Qwen3-4B does not need to be
downloaded again every run.
"""

import modal


# ---------------------------------------------------------
# Modal application
# ---------------------------------------------------------

app = modal.App(
    "aasr-e1-balanced-training"
)


# ---------------------------------------------------------
# Container image
# ---------------------------------------------------------

image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.1",
        "peft==0.17.1",
        "trl==0.23.1",
        "datasets>=3.0",
        "accelerate>=1.0",
        "sentencepiece==0.2.1",
        "jinja2>=3.1",
        "numpy>=1.26",
    )
    .add_local_file(
        "experiments/E1_balanced_data/train_balanced_adapters.py",
        remote_path="/workspace/train_balanced_adapters.py",
    )
)


# ---------------------------------------------------------
# Shared Modal volumes
# ---------------------------------------------------------

artifacts = modal.Volume.from_name(
    "asar-artifacts",
    create_if_missing=True,
)

cache = modal.Volume.from_name(
    "asar-hf-cache",
    create_if_missing=True,
)


# ---------------------------------------------------------
# Training function
# ---------------------------------------------------------

@app.function(
    image=image,

    # Same GPU family used in previous project experiments.
    gpu="L40S",

    # Enough time for either Short or Long training.
    timeout=60 * 60 * 12,

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
def train_adapter(
    adapter: str,
    speed_mode: str = "fast",
):
    """
    Train either the E1 Short or Long adapter.

    adapter:
        "short" or "long"

    speed_mode:
        "fast" -> batch 2 + grad accumulation 4
        "safe" -> batch 1 + grad accumulation 8

    Both modes keep effective batch size = 8.
    """

    import os
    import subprocess

    if adapter not in {
        "short",
        "long",
    }:
        raise ValueError(
            "adapter must be short or long"
        )

    if speed_mode not in {
        "fast",
        "safe",
    }:
        raise ValueError(
            "speed_mode must be fast or safe"
        )

    env = os.environ.copy()

    # Shared Hugging Face cache
    env["HF_HOME"] = (
        "/root/.cache/huggingface"
    )

    # Helps reduce CUDA memory fragmentation
    env[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = "expandable_segments:True"

    print(
        "\n"
        "========================================"
    )
    print(
        "AASR E1 BALANCED TRAINING"
    )
    print(
        "========================================"
    )
    print(
        f"Adapter: {adapter}"
    )
    print(
        f"Speed mode: {speed_mode}"
    )
    print(
        "GPU: L40S"
    )
    print(
        "========================================\n"
    )

    try:

        subprocess.run(
            [
                "python",
                "-u",
                "/workspace/train_balanced_adapters.py",

                "--adapter",
                adapter,

                "--e1-dir",
                (
                    "/workspace/artifacts/"
                    "experiments/"
                    "E1_balanced_data"
                ),

                "--seed",
                "42",

                "--speed-mode",
                speed_mode,
            ],
            check=True,
            env=env,
        )

    finally:

        # Save adapters/checkpoints even if training
        # fails after writing a checkpoint.
        artifacts.commit()

        # Persist downloaded HF files.
        cache.commit()


# ---------------------------------------------------------
# Local command
# ---------------------------------------------------------

@app.local_entrypoint()
def main(
    adapter: str = "short",
    speed_mode: str = "fast",
):
    """
    Launch one training job.

    Example:

        modal run ... --adapter short

    Training stays attached so logs are visible
    directly in the terminal.
    """

    train_adapter.remote(
        adapter=adapter,
        speed_mode=speed_mode,
    )