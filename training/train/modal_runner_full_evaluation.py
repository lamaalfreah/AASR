import modal

app = modal.App("asar-full-evaluation")

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
        "training/ASAR/evaluate_full_optimized.py",
        remote_path="/workspace/evaluate_full_optimized.py",
    )
)

artifacts = modal.Volume.from_name("asar-artifacts", create_if_missing=True)
cache = modal.Volume.from_name("asar-hf-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu="L40S",
    timeout=60*60*12,
    volumes={
        "/workspace/artifacts": artifacts,
        "/root/.cache/huggingface": cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_eval(split: str = "validation", family: str = "grpo"):
    import os, subprocess
    env = os.environ.copy()
    env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        subprocess.run(
            [
                "python","-u","/workspace/evaluate_full_optimized.py",
                "--output-root","/workspace/artifacts/asar_sft_grpo",
                "--split",split,
                "--family",family,
                "--checkpoint-every","25",
            ],
            check=True,
            env=env,
        )
    finally:
        artifacts.commit()
        cache.commit()

@app.local_entrypoint()
def main(split: str = "validation", family: str = "grpo"):
    call = run_eval.spawn(split=split, family=family)
    print(f"Spawned remote call: {call.object_id}")
    print("Evaluation continues independently on Modal.")
