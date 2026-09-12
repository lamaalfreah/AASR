"""Official Qwen3-4B only: one persistent L4, no adapters or spatial execution.

Run from the repository root: modal run -m language.modal_long_backend
No service is deployed; exiting the attached run releases its resources.
"""
from __future__ import annotations
import modal

app = modal.App('aasr-step2b-qwen3-4b-long')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.6.0', 'transformers==4.57.1',
                      'huggingface-hub==0.36.0', 'safetensors==0.6.2')
         .env({'HF_HOME': '/root/.cache/huggingface',
               'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
         .add_local_python_source('language'))
hf_cache = modal.Volume.from_name('asar-hf-cache')
checkpoints = modal.Volume.from_name('aasr-step2b-long-checkpoints', create_if_missing=True)


@app.function(image=image, volumes={'/root/.cache/huggingface': hf_cache},
              cpu=2, memory=4096, timeout=1200, retries=0, max_containers=1)
def prepare_weights(revision: str = ''):
    """Download on CPU, before any GPU is allocated; pin the official revision."""
    from huggingface_hub import HfApi, snapshot_download
    from language.modal_checkpoint import MODEL
    from time import perf_counter
    started = perf_counter()
    revision = revision or HfApi().model_info(MODEL).sha
    snapshot_download(MODEL, revision=revision,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
    hf_cache.commit()
    return dict(model=MODEL, revision=revision,
                weight_preparation_seconds=perf_counter()-started)


@app.cls(image=image, gpu='L4', cpu=2, memory=16384,
         volumes={'/root/.cache/huggingface': hf_cache, '/checkpoints': checkpoints},
         max_containers=1, min_containers=0, buffer_containers=0,
         scaledown_window=60, timeout=180, startup_timeout=600, retries=0)
class QwenLong:
    revision: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from language.modal_checkpoint import MODEL
        from time import perf_counter
        import uuid
        started = perf_counter()
        if torch.cuda.device_count() != 1 or 'L4' != torch.cuda.get_device_name(0).split()[-1]:
            raise RuntimeError('Exactly one NVIDIA L4 is required')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('bfloat16 is unavailable; stop rather than change precision')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=self.revision,
                                                       local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=self.revision, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            use_safetensors=True).to('cuda').eval()
        torch.cuda.synchronize()
        self.info = dict(model=MODEL, revision=self.revision,
                         gpu=torch.cuda.get_device_name(0), gpu_count=1,
                         precision=str(self.model.dtype),
                         model_load_seconds=perf_counter()-started,
                         container_session=uuid.uuid4().hex,
                         torch_version=torch.__version__, transformers_version=transformers.__version__)

    @modal.method()
    def metadata(self):
        return self.info

    @modal.method()
    def generate(self, messages: list, request_digest: str):
        import torch
        import json
        import re
        from pathlib import Path
        from time import perf_counter
        from language.modal_checkpoint import atomic_json, GENERATION
        if not re.fullmatch('[a-f0-9]{64}', request_digest):
            raise ValueError('Invalid checkpoint key')
        path = Path('/checkpoints') / (request_digest + '.json')
        if path.exists():
            return json.loads(path.read_text())
        began = perf_counter()
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt').to('cuda')
        length = inputs.input_ids.shape[-1]
        if length + GENERATION['max_new_tokens'] > self.model.config.max_position_embeddings:
            raise RuntimeError('Input exceeds context budget; no silent truncation')
        torch.cuda.synchronize()
        generation_start = perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False,
                       max_new_tokens=GENERATION['max_new_tokens'], use_cache=True,
                       pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        generated = output[0, length:]
        response = dict(self.info, request_digest=request_digest,
                        text=self.tokenizer.decode(generated, skip_special_tokens=True),
                        input_tokens=int(length), output_tokens=len(generated),
                        generation_ms=(perf_counter()-generation_start)*1000,
                        inference_ms=(perf_counter()-began)*1000,
                        reached_token_limit=len(generated) == GENERATION['max_new_tokens'])
        atomic_json(path, response)
        checkpoints.commit()  # Survives a lost response or interrupted local process.
        return response


@app.local_entrypoint()
def main():
    from evaluation.evaluate_modal_long import run
    run(prepare_weights, QwenLong, app.app_id)
