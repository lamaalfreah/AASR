import modal

app = modal.App("aasr-e2-english-diagnostic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.57.1",
        "peft==0.17.1",
        "datasets>=3.0",
        "requests>=2.32",
        "sentencepiece==0.2.1",
        "numpy>=1.26",
    )
    .add_local_file(
        "training/ASAR/evaluate_e2_english.py",
        remote_path="/workspace/evaluate_e2_english.py",
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
    family: str = "sft",
    max_samples: int = 250,
    seed: int = 42,
    alphas: str = "0,0.25,1",
    beta: float = 0.2,
    max_new_floor: int = 0,
    arabic_summary: str = "",
):
    import os, subprocess

    env = os.environ.copy()
    env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        subprocess.run(
            [
                "python", "-u", "/workspace/evaluate_e2_english.py",
                "--output-root", "/workspace/artifacts/asar_sft_grpo",
                "--family", family,
                "--max-samples", str(max_samples),
                "--seed", str(seed),
                "--alphas", alphas,
                "--beta", str(beta),
                "--max-new-floor", str(max_new_floor),
                "--checkpoint-every", "10",
            ] + (["--arabic-summary", arabic_summary] if arabic_summary else []),
            check=True,
            env=env,
        )
    finally:
        artifacts.commit()
        cache.commit()


@app.local_entrypoint()
def main(
    family: str = "sft",
    max_samples: int = 250,
    seed: int = 42,
    alphas: str = "0,0.25,1",
    beta: float = 0.2,
    max_new_floor: int = 0,
    arabic_summary: str = "",
):
    call = run_eval.spawn(
        family=family,
        max_samples=max_samples,
        seed=seed,
        alphas=alphas,
        beta=beta,
        max_new_floor=max_new_floor,
        arabic_summary=arabic_summary,
    )
    print(f"Spawned remote call: {call.object_id}")
    print("E2 diagnostic continues independently on Modal.")