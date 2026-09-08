import modal

app = modal.App("aasr-alpha-router-v3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.1",
        "scikit-learn==1.7.2",
        "requests>=2.32",
        "sentencepiece==0.2.1",
        "numpy>=1.26",
    )
    .add_local_file(
        "training/ASAR/train_alpha_router_v3.py",
        remote_path="/workspace/train_alpha_router_v3.py",
    )
)

artifacts = modal.Volume.from_name("asar-artifacts", create_if_missing=True)
cache = modal.Volume.from_name("asar-hf-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={
        "/workspace/artifacts": artifacts,
        "/root/.cache/huggingface": cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_router():
    import os, subprocess

    env = os.environ.copy()
    env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    env["HF_HOME"] = "/root/.cache/huggingface"

    try:
        subprocess.run(
            [
                "python", "-u", "/workspace/train_alpha_router_v3.py",
                "--output-root", "/workspace/artifacts/asar_sft_grpo",
                "--source-tag", "sft_validation_all_n120_seed42",
                "--seed", "42",
                "--max-length", "512",
                "--epochs", "6",
            ],
            check=True,
            env=env,
        )
    finally:
        artifacts.commit()
        cache.commit()
