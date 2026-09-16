"""
Modal runner for AASR Experiment E1 evaluation.

Compares:
    E0 SFT adapters
    vs
    E1 balanced SFT adapters

Evaluation:
    - validation split
    - 400 examples by default
    - same examples for E0 and E1
    - balanced sampling by task_type
    - resume supported through saved JSONL outputs

GPU:
    NVIDIA L40S
"""

import modal


# =========================================================
# MODAL APP
# =========================================================

app = modal.App(
    "aasr-e1-evaluation"
)


# =========================================================
# IMAGE
# =========================================================

image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.1",
        "peft==0.17.1",
        "accelerate>=1.0",
        "requests>=2.32",
        "sentencepiece==0.2.1",
        "jinja2>=3.1",
        "numpy>=1.26",
    )
    .add_local_file(
        "experiments/E1_balanced_data/evaluate_e1_vs_e0.py",
        remote_path="/workspace/evaluate_e1_vs_e0.py",
    )
)


# =========================================================
# SHARED VOLUMES
# =========================================================

artifacts = modal.Volume.from_name(
    "asar-artifacts",
    create_if_missing=True,
)

cache = modal.Volume.from_name(
    "asar-hf-cache",
    create_if_missing=True,
)


# =========================================================
# REMOTE EVALUATION
# =========================================================

@app.function(
    image=image,

    gpu="L40S",

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
def evaluate(
    n: int = 400,
    seed: int = 42,
):
    """
    Run E0 vs E1 evaluation.

    Results are written to:

    /workspace/artifacts/
        experiments/
        E1_balanced_data/
        evaluation/

    If the run is interrupted, the evaluation script
    resumes from previously saved example outputs.
    """

    import os
    import subprocess

    env = os.environ.copy()

    env["HF_HOME"] = (
        "/root/.cache/huggingface"
    )

    env[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = "expandable_segments:True"

    print(
        "\n"
        "========================================"
    )

    print(
        "AASR E1 vs E0 EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"Validation samples: {n}"
    )

    print(
        f"Seed: {seed}"
    )

    print(
        "GPU: L40S"
    )

    print(
        "Comparison: E0 SFT vs E1 Balanced SFT"
    )

    print(
        "========================================\n"
    )

    try:

        subprocess.run(
            [
                "python",
                "-u",
                "/workspace/evaluate_e1_vs_e0.py",

                "--e0-root",
                "/workspace/artifacts/asar_sft_grpo",

                "--e1-root",
                (
                    "/workspace/artifacts/"
                    "experiments/"
                    "E1_balanced_data"
                ),

                "--n",
                str(n),

                "--seed",
                str(seed),

                "--checkpoint-every",
                "25",
            ],
            check=True,
            env=env,
        )

    finally:

        # Preserve partial evaluation outputs
        # even if the run is interrupted.
        artifacts.commit()

        # Preserve Hugging Face model cache.
        cache.commit()


# =========================================================
# LOCAL ENTRYPOINT
# =========================================================

@app.local_entrypoint()
def main(
    n: int = 400,
    seed: int = 42,
):
    """
    Example:

    modal run --detach \
        experiments/E1_balanced_data/modal_runner_evaluate_e1.py \
        --n 400 \
        --seed 42
    """

    evaluate.remote(
        n=n,
        seed=seed,
    )