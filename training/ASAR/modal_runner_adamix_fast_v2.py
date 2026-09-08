import modal

app = modal.App("aasr-adamix-fast-evaluation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.1",
        "peft==0.17.1",
        "requests>=2.32",
        "sentencepiece==0.2.1",
        "numpy>=1.26",
    )
    .add_local_file(
        "training/ASAR/evaluate_full_adamix_fast.py",
        remote_path="/workspace/evaluate_full_adamix_fast.py",
    )
)

artifacts = modal.Volume.from_name("asar-artifacts", create_if_missing=True)
cache = modal.Volume.from_name("asar-hf-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 12,
    volumes={
        "/workspace/artifacts": artifacts,
        "/root/.cache/huggingface": cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_eval(
    split: str = "validation",
    family: str = "sft",
    scope: str = "all",
    max_samples: int = 120,
    seed: int = 42,
    alphas: str = "0,0.25,0.5,0.75,1",
    beta: float = 0.2,
):
    import os, subprocess

    env = os.environ.copy()
    env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        subprocess.run(
            [
                "python", "-u", "/workspace/evaluate_full_adamix_fast.py",
                "--output-root", "/workspace/artifacts/asar_sft_grpo",
                "--split", split,
                "--family", family,
                "--scope", scope,
                "--max-samples", str(max_samples),
                "--seed", str(seed),
                "--alphas", alphas,
                "--beta", str(beta),
                "--checkpoint-every", "10",
            ],
            check=True,
            env=env,
        )
    finally:
        artifacts.commit()
        cache.commit()


@app.local_entrypoint()
def main(
    split: str = "validation",
    family: str = "sft",
    scope: str = "all",
    max_samples: int = 120,
    seed: int = 42,
    alphas: str = "0,0.25,0.5,0.75,1",
    beta: float = 0.2,
):
    call = run_eval.spawn(
        split=split,
        family=family,
        scope=scope,
        max_samples=max_samples,
        seed=seed,
        alphas=alphas,
        beta=beta,
    )
    print(f"Spawned remote call: {call.object_id}")
    print("Evaluation continues independently on Modal.")
