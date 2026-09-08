import modal
app=modal.App("asar-empirical-router-extend")
image=(modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch>=2.6","transformers>=4.51,<6","peft>=0.15","requests>=2.32","sentencepiece")
       .add_local_file("training/ASAR/extend_empirical_labels.py",remote_path="/workspace/extend_empirical_labels.py"))
artifacts=modal.Volume.from_name("asar-artifacts",create_if_missing=True)
cache=modal.Volume.from_name("asar-hf-cache",create_if_missing=True)

@app.function(image=image,gpu="L40S",timeout=60*60*8,
              volumes={"/workspace/artifacts":artifacts,"/root/.cache/huggingface":cache},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def extend_labels(target_n:int=2000,seed:int=42):
    import os,subprocess
    env=os.environ.copy()
    env["HF_TOKEN"]=os.environ["HF_TOKEN"]
    env["HF_HOME"]="/root/.cache/huggingface"
    env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    try:
        subprocess.run(["python","-u","/workspace/extend_empirical_labels.py",
                        "--output-root","/workspace/artifacts/asar_sft_grpo",
                        "--target-n",str(target_n),"--seed",str(seed),
                        "--checkpoint-every","25"],check=True,env=env)
    finally:
        artifacts.commit()
        cache.commit()

@app.local_entrypoint()
def main(target_n:int=2000,seed:int=42):
    extend_labels.remote(target_n=target_n,seed=seed)
