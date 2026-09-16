"""
AASR Experiment E1 - Balanced Adapter Training
==============================================

Purpose
-------
Train new Short and Long LoRA adapters using the balanced datasets
created in Experiment E1.

Experimental rule
-----------------
E1 changes ONLY the training-data distribution.

The following settings are intentionally kept identical to E0:

    Base model:
        Qwen/Qwen3-4B

    LoRA:
        r = 16
        alpha = 32
        dropout = 0.05
        target_modules = ["q_proj", "v_proj"]

    Training:
        learning_rate = 1e-4
        epochs = 1
        effective_batch_size = 8

    Context:
        Short max_length = 4096
        Long max_length = 8192

    Specialization:
        Balanced Easy -> Short Adapter
        Balanced Hard -> Long Adapter

This allows a controlled comparison:

    E0 = original imbalanced data
    E1 = balanced data

without changing the model configuration.

Inputs
------
balanced_short_train.jsonl
balanced_long_train.jsonl

Outputs
-------
adapters/short_sft_balanced/
adapters/long_sft_balanced/

Each adapter directory contains:
    adapter weights
    tokenizer files
    checkpoints
    run_info.json
    COMPLETE marker
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------
# Configuration - SAME AS E0
# ---------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-4B"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj",
    "v_proj",
]

SHORT_MAX_LENGTH = 4096
LONG_MAX_LENGTH = 8192

SEED = 42

SYSTEM_PROMPT = """أنت نموذج متخصص في الاستدلال المكاني باللغة العربية.
اعتمد فقط على السياق المعطى.
يجب أن تنتهي إجابتك دائمًا بالسطر:
FINAL: <الإجابة>
"""


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():
    """
    Parse training arguments.

    --adapter:
        short -> train balanced Easy data
        long  -> train balanced Hard data

    --speed-mode:
        fast -> batch 2, grad accumulation 4
        safe -> batch 1, grad accumulation 8
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter",
        choices=["short", "long"],
        required=True,
    )

    parser.add_argument(
        "--e1-dir",
        default=(
            "/workspace/artifacts/"
            "experiments/E1_balanced_data"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--speed-mode",
        choices=["fast", "safe"],
        default="fast",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def read_jsonl(path):
    """
    Read a JSONL dataset into memory.
    """

    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


# ---------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------

def make_lora():
    """
    Create the exact LoRA configuration used in E0.
    """

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------
# Prompt formatting - SAME AS E0
# ---------------------------------------------------------------------

def messages(row):
    """
    Build the system and user messages used in E0.
    """

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"السياق:\n"
                f"{row['context']}\n\n"
                f"السؤال:\n"
                f"{row['question']}\n\n"
                f"أجب اعتمادًا على السياق فقط."
            ),
        },
    ]


def render_prompt(
    tokenizer,
    row,
    thinking=False,
):
    """
    Render the Qwen chat template.

    Short:
        thinking=False

    Long:
        thinking=True
    """

    return tokenizer.apply_chat_template(
        messages(row),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def head_tail(
    tokenizer,
    text,
    max_tokens,
):
    """
    Apply the same head-tail truncation used in E0.

    If the prompt exceeds the allowed size:
        35% is kept from the beginning
        65% is kept from the end
    """

    ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    if len(ids) <= max_tokens:
        return text

    head = max(
        1,
        int(max_tokens * 0.35),
    )

    tail = (
        max_tokens
        - head
    )

    selected_ids = (
        ids[:head]
        + ids[-tail:]
    )

    return tokenizer.decode(
        selected_ids,
        skip_special_tokens=True,
    )


# ---------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------

def make_sft_dataset(
    rows,
    tokenizer,
    adapter_type,
    max_length,
):
    """
    Build the same prompt/completion structure used in E0.

    Short:
        completion = FINAL: answer

    Long:
        completion = verified rationale + FINAL: answer

    Completion space is reserved exactly as in E0:

        Short reserve = 256
        Long reserve  = 2048
    """

    reserve = (
        256
        if adapter_type == "short"
        else 2048
    )

    max_prompt = max(
        512,
        max_length - reserve,
    )

    data = []

    for row in rows:

        prompt = head_tail(
            tokenizer,
            render_prompt(
                tokenizer,
                row,
                thinking=(
                    adapter_type == "long"
                ),
            ),
            max_prompt,
        )

        if adapter_type == "short":

            completion = (
                f"FINAL: {row['answer']}"
            )

        else:

            rationale = str(
                row.get(
                    "verified_rationale"
                )
                or ""
            ).strip()

            completion = (
                (
                    rationale + "\n"
                    if rationale
                    else ""
                )
                + f"FINAL: {row['answer']}"
            )

        data.append(
            {
                "prompt": prompt,
                "completion": completion,
            }
        )

    return Dataset.from_list(data)


# ---------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------

def find_last_checkpoint(
    output_dir,
):
    """
    Find the most recent checkpoint so interrupted Modal
    training can resume instead of restarting.
    """

    output_dir = Path(
        output_dir
    )

    checkpoints = []

    for path in output_dir.glob(
        "checkpoint-*"
    ):

        try:
            step = int(
                path.name.split("-")[-1]
            )

            checkpoints.append(
                (step, path)
            )

        except ValueError:
            pass

    if not checkpoints:
        return None

    return str(
        max(
            checkpoints,
            key=lambda item:
                item[0],
        )[1]
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train(args):
    """
    Train one balanced E1 adapter.

    The function trains only the adapter selected by --adapter.

    Example:
        --adapter short

    trains:
        balanced_short_train.jsonl

    and saves:
        adapters/short_sft_balanced
    """

    set_seed(args.seed)

    root = Path(
        args.e1_dir
    )

    adapter_type = (
        args.adapter
    )

    if adapter_type == "short":

        data_path = (
            root
            / "balanced_short_train.jsonl"
        )

        output_dir = (
            root
            / "adapters"
            / "short_sft_balanced"
        )

        max_length = (
            SHORT_MAX_LENGTH
        )

    else:

        data_path = (
            root
            / "balanced_long_train.jsonl"
        )

        output_dir = (
            root
            / "adapters"
            / "long_sft_balanced"
        )

        max_length = (
            LONG_MAX_LENGTH
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Do not retrain a completed adapter.
    if (
        (
            output_dir
            / "COMPLETE"
        ).exists()
        and
        (
            output_dir
            / "adapter_config.json"
        ).exists()
    ):

        print(
            "Adapter already complete:",
            output_dir,
        )

        return

    if not data_path.exists():

        raise FileNotFoundError(
            f"Missing balanced dataset: "
            f"{data_path}"
        )

    rows = read_jsonl(
        data_path
    )

    print(
        f"\nTraining E1 "
        f"{adapter_type.upper()} adapter"
    )

    print(
        "Examples:",
        len(rows),
    )

    print(
        "Max length:",
        max_length,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    dataset = make_sft_dataset(
        rows,
        tokenizer,
        adapter_type,
        max_length,
    )

    # Same effective batch size as E0: 8
    if args.speed_mode == "fast":

        per_device_batch = 2
        gradient_accumulation = 4
        workers = 4

    else:

        per_device_batch = 1
        gradient_accumulation = 8
        workers = 2

    config = SFTConfig(
        output_dir=str(
            output_dir
        ),

        learning_rate=1e-4,

        num_train_epochs=1,

        per_device_train_batch_size=(
            per_device_batch
        ),

        gradient_accumulation_steps=(
            gradient_accumulation
        ),

        bf16=True,

        tf32=True,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        max_length=max_length,

        completion_only_loss=True,

        packing=False,

        logging_steps=10,

        save_strategy="steps",

        save_steps=25,

        save_total_limit=2,

        report_to="none",

        seed=args.seed,

        dataset_num_proc=4,

        dataloader_num_workers=(
            workers
        ),

        dataloader_pin_memory=True,

        optim="adamw_torch_fused",

        model_init_kwargs={
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    )

    trainer = SFTTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=make_lora(),
    )

    checkpoint = (
        find_last_checkpoint(
            output_dir
        )
    )

    if checkpoint:

        print(
            "Resuming from:",
            checkpoint,
        )

    else:

        print(
            "Starting from step 0."
        )

    print(
        f"speed_mode="
        f"{args.speed_mode}"
    )

    print(
        f"batch="
        f"{per_device_batch}"
    )

    print(
        f"gradient_accumulation="
        f"{gradient_accumulation}"
    )

    print(
        f"effective_batch="
        f"{per_device_batch * gradient_accumulation}"
    )

    try:

        trainer.train(
            resume_from_checkpoint=
                checkpoint
        )

    except torch.cuda.OutOfMemoryError:

        print(
            "\nCUDA OOM."
        )

        print(
            "Checkpoint preserved."
        )

        print(
            "Re-run using "
            "--speed-mode safe."
        )

        raise

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    (
        output_dir
        / "COMPLETE"
    ).write_text(
        "ok\n",
        encoding="utf-8",
    )

    run_info = {
        "experiment":
            "E1_balanced_data",

        "adapter":
            adapter_type,

        "method":
            "SFT+LoRA",

        "dataset":
            data_path.name,

        "examples":
            len(dataset),

        "base_model":
            MODEL_ID,

        "r":
            LORA_R,

        "alpha":
            LORA_ALPHA,

        "dropout":
            LORA_DROPOUT,

        "target_modules":
            LORA_TARGETS,

        "learning_rate":
            1e-4,

        "epochs":
            1,

        "max_length":
            max_length,

        "speed_mode":
            args.speed_mode,

        "batch":
            per_device_batch,

        "gradient_accumulation":
            gradient_accumulation,

        "effective_batch_size":
            (
                per_device_batch
                * gradient_accumulation
            ),

        "seed":
            args.seed,

        "change_vs_E0":
            "Training data distribution only",
    }

    with (
        output_dir
        / "run_info.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            run_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nTraining complete."
    )

    print(
        json.dumps(
            run_info,
            ensure_ascii=False,
            indent=2,
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    args = parse_args()

    train(args)