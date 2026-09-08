import modal
app=modal.App("asar-continuous-router")
image=(modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.6.0","transformers==4.57.1","scikit-learn==1.7.2","sentencepiece==0.2.1","numpy>=1.26")
       .add_local_file("training/ASAR/build_continuous_scores.py",remote_path="/workspace/build_continuous_scores.py")
       .add_local_file("training/ASAR/train_continuous_router.py",remote_path="/workspace/train_continuous_router.py"))
artifacts=modal.Volume.from_name("asar-artifacts",create_if_missing=True)
cache=modal.Volume.from_name("asar-hf-cache",create_if_missing=True)

@app.function(image=image,cpu=4.0,memory=8192,timeout=3600,volumes={"/workspace/artifacts":artifacts})
def build_scores():
    import subprocess
    try:
        subprocess.run(["python","-u","/workspace/build_continuous_scores.py","--output-root","/workspace/artifacts/asar_sft_grpo"],check=True)
    finally:
        artifacts.commit()

@app.function(image=image,gpu="L40S",timeout=60*60*3,
              volumes={"/workspace/artifacts":artifacts,"/root/.cache/huggingface":cache})
def train_router():
    import os,subprocess
    env=os.environ.copy(); env["HF_HOME"]="/root/.cache/huggingface"
    try:
        subprocess.run(["python","-u","/workspace/train_continuous_router.py",
                        "--output-root","/workspace/artifacts/asar_sft_grpo",
                        "--seed","42","--epochs","4","--max-length","512"],check=True,env=env)
    finally:
        artifacts.commit(); cache.commit()

@app.local_entrypoint()
def main(stage:str):
    if stage=="build-scores":
        build_scores.remote()
    elif stage=="train":
        call=train_router.spawn()
        print(f"Spawned remote call: {call.object_id}")
        print("The function continues independently on Modal.")
    else:
        raise ValueError("stage must be build-scores or train")
