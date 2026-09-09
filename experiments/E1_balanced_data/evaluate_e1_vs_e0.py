"""
AASR Experiment E1 - Evaluate E1 vs E0
======================================

Goal
----
Measure the effect of DATA BALANCING ONLY.

Comparison:

    E0 Short adapter
        trained on original Easy data

    vs

    E1 Short adapter
        trained on balanced Easy data


    E0 Long adapter
        trained on original Hard data

    vs

    E1 Long adapter
        trained on balanced Hard data


Important
---------
This is NOT a Router experiment.

This is NOT an alpha-mixing experiment.

This experiment answers only:

    Did balancing the task-type distribution improve
    the Short and Long specialized adapters?


Evaluation design
-----------------
Validation split only.

Default:
    400 examples

Sampling:
    balanced / round-robin by task_type

Easy examples:
    E0 Short vs E1 Short

Hard examples:
    E0 Long vs E1 Long

Therefore every example is evaluated using the adapter
specialized for its difficulty.

Metrics:
    - overall accuracy
    - Easy accuracy
    - Hard accuracy
    - accuracy per task_type
    - average generated tokens
    - average latency
    - E1 - E0 accuracy delta

The evaluation uses the SAME generation and strict
answer-checking rules as the existing AASR evaluation.
"""

import argparse
import csv
import gzip
import io
import json
import os
import random
import re
import time

from pathlib import Path

import requests
import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_ID = "Qwen/Qwen3-4B"

DATA_BASE = (
    "https://huggingface.co/datasets/"
    "ArabicSpatialrReasoning/"
    "arabic-spatial-reasoning-v1/"
    "resolve/main"
)

SYSTEM_PROMPT = """أنت نموذج متخصص في الاستدلال المكاني باللغة العربية.
اعتمد فقط على السياق المعطى.
يجب أن تنتهي إجابتك دائمًا بالسطر:
FINAL: <الإجابة>
"""

DEFAULT_N = 400
DEFAULT_SEED = 42


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--e0-root",
        default="/workspace/artifacts/asar_sft_grpo",
    )

    parser.add_argument(
        "--e1-root",
        default=(
            "/workspace/artifacts/"
            "experiments/E1_balanced_data"
        ),
    )

    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
    )

    return parser.parse_args()


# =========================================================
# DATA
# =========================================================

def download_validation():
    """
    Download the private validation split from Hugging Face.
    """

    token = os.environ.get("HF_TOKEN")

    headers = (
        {
            "Authorization":
                f"Bearer {token}"
        }
        if token
        else {}
    )

    url = (
        f"{DATA_BASE}/validation.jsonl.gz"
    )

    response = requests.get(
        url,
        headers=headers,
        timeout=300,
    )

    response.raise_for_status()

    rows = []

    with gzip.GzipFile(
        fileobj=io.BytesIO(
            response.content
        )
    ) as gz:

        for raw in gz:

            line = (
                raw.decode("utf-8")
                .strip()
            )

            if line:

                rows.append(
                    json.loads(line)
                )

    return rows


def difficulty(row):

    return str(
        row.get(
            "difficulty",
            "",
        )
    ).strip().lower()


def make_balanced_sample(
    rows,
    n,
    seed,
):
    """
    Deterministic round-robin sampling by task_type.

    This prevents large task groups from dominating
    the evaluation accuracy.
    """

    eligible = [
        row
        for row in rows
        if difficulty(row)
        in {
            "easy",
            "hard",
        }
    ]

    buckets = {}

    for row in eligible:

        task = str(
            row.get(
                "task_type",
                "unknown",
            )
        )

        buckets.setdefault(
            task,
            [],
        ).append(row)

    rng = random.Random(seed)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    task_names = sorted(
        buckets.keys()
    )

    selected = []

    index = 0

    while (
        len(selected) < n
        and task_names
    ):

        task = task_names[
            index
            % len(task_names)
        ]

        if buckets[task]:

            selected.append(
                buckets[task].pop()
            )

        task_names = [
            name
            for name in task_names
            if buckets[name]
        ]

        index += 1

    return selected


# =========================================================
# STRICT ANSWER CHECKING
# SAME LOGIC AS EXISTING AASR EVALUATION
# =========================================================

def normalize_text(text):

    text = str(
        text or ""
    ).strip().lower()

    text = (
        text
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )

    text = re.sub(
        r"[^\w\s\u0600-\u06FF.-]",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def extract_final(text):

    matches = re.findall(
        r"FINAL\s*:\s*(.+)",
        str(text or ""),
        flags=re.I,
    )

    return (
        matches[-1].strip()
        if matches
        else ""
    )


def normalize_direction(text):

    value = normalize_text(text)

    mapping = {

        "شمال": "north",
        "الشمال": "north",
        "north": "north",

        "جنوب": "south",
        "الجنوب": "south",
        "south": "south",

        "شرق": "east",
        "الشرق": "east",
        "east": "east",

        "غرب": "west",
        "الغرب": "west",
        "west": "west",

        "شمال شرق": "northeast",
        "شمال شرقي": "northeast",
        "northeast": "northeast",

        "شمال غرب": "northwest",
        "شمال غربي": "northwest",
        "northwest": "northwest",

        "جنوب شرق": "southeast",
        "جنوب شرقي": "southeast",
        "southeast": "southeast",

        "جنوب غرب": "southwest",
        "جنوب غربي": "southwest",
        "southwest": "southwest",
    }

    return mapping.get(
        value,
        value,
    )


def normalize_yes_no(text):

    value = normalize_text(text)

    if value in {
        "نعم",
        "yes",
        "ايوه",
        "اي",
    }:
        return "yes"

    if value in {
        "لا",
        "no",
        "كلا",
    }:
        return "no"

    return value


def parse_integer(text):

    translation = (
        str.maketrans(
            "٠١٢٣٤٥٦٧٨٩",
            "0123456789",
        )
    )

    match = re.search(
        r"-?\d+",
        str(
            text or ""
        ).translate(
            translation
        ),
    )

    return (
        int(match.group())
        if match
        else None
    )


def is_correct(
    output,
    answer,
    task,
):

    prediction = extract_final(
        output
    )

    if not prediction:
        return False

    if task == "cardinal_direction":

        return (
            normalize_direction(
                prediction
            )
            ==
            normalize_direction(
                answer
            )
        )

    if task == "within_radius_yes_no":

        return (
            normalize_yes_no(
                prediction
            )
            ==
            normalize_yes_no(
                answer
            )
        )

    if task == "count_within_radius":

        return (
            parse_integer(
                prediction
            )
            ==
            parse_integer(
                answer
            )
        )

    return (
        normalize_text(
            prediction
        )
        ==
        normalize_text(
            answer
        )
    )


# =========================================================
# PROMPT
# =========================================================

def messages(row):

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
                "أجب اعتمادًا على السياق فقط."
            ),
        },
    ]


def truncate_prompt(
    tokenizer,
    prompt,
    max_input,
):
    """
    Same 35% head / 65% tail truncation
    used in the existing AASR pipeline.
    """

    ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    if len(ids) <= max_input:

        return prompt

    head = max(
        1,
        int(
            max_input
            * 0.35
        ),
    )

    tail = (
        max_input
        - head
    )

    return tokenizer.decode(
        ids[:head]
        + ids[-tail:],
        skip_special_tokens=True,
    )


# =========================================================
# GENERATION
# =========================================================

@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    row,
    adapter_name,
    adapter_type,
):
    """
    Short:
        thinking=False
        total budget=4096
        max_new_tokens=256

    Long:
        thinking=True
        total budget=8192
        max_new_tokens=1024
    """

    if adapter_type == "short":

        thinking = False
        total_budget = 4096
        max_new_tokens = 256

    else:

        thinking = True
        total_budget = 8192
        max_new_tokens = 1024

    model.set_adapter(
        adapter_name
    )

    prompt = (
        tokenizer.apply_chat_template(
            messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    )

    max_input = max(
        512,
        total_budget
        - max_new_tokens,
    )

    prompt = truncate_prompt(
        tokenizer,
        prompt,
        max_input,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(
        model.device
    )

    start = time.perf_counter()

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=(
            tokenizer.eos_token_id
        ),
    )

    latency = (
        time.perf_counter()
        - start
    )

    generated = output[
        0,
        inputs[
            "input_ids"
        ].shape[1]:
    ]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return (
        text,
        int(
            generated.numel()
        ),
        latency,
    )


# =========================================================
# ADAPTER PATH CHECK
# =========================================================

def require_adapter(path):

    path = Path(path)

    required = (
        path
        / "adapter_config.json"
    )

    if not required.exists():

        raise FileNotFoundError(
            f"Adapter not found: {path}"
        )

    return path


# =========================================================
# RESUME SUPPORT
# =========================================================

def load_jsonl(path):

    rows = []

    path = Path(path)

    if not path.exists():
        return rows

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


# =========================================================
# SUMMARY
# =========================================================

def summarize(records):
    """
    Produce:
        overall comparison
        difficulty comparison
        task-level comparison
    """

    groups = {
        "overall": records
    }

    difficulties = sorted(
        {
            row["difficulty"]
            for row in records
        }
    )

    for diff in difficulties:

        groups[
            f"difficulty:{diff}"
        ] = [
            row
            for row in records
            if row["difficulty"]
            == diff
        ]

    tasks = sorted(
        {
            row["task_type"]
            for row in records
        }
    )

    for task in tasks:

        groups[
            f"task:{task}"
        ] = [
            row
            for row in records
            if row["task_type"]
            == task
        ]

    summary = []

    for group_name, rows in groups.items():

        n = len(rows)

        if not n:
            continue

        e0_accuracy = (
            sum(
                int(
                    row[
                        "e0_correct"
                    ]
                )
                for row in rows
            )
            / n
        )

        e1_accuracy = (
            sum(
                int(
                    row[
                        "e1_correct"
                    ]
                )
                for row in rows
            )
            / n
        )

        summary.append(
            {
                "group":
                    group_name,

                "n":
                    n,

                "e0_accuracy":
                    round(
                        e0_accuracy,
                        6,
                    ),

                "e1_accuracy":
                    round(
                        e1_accuracy,
                        6,
                    ),

                "delta_e1_minus_e0":
                    round(
                        e1_accuracy
                        - e0_accuracy,
                        6,
                    ),

                "e0_avg_tokens":
                    round(
                        sum(
                            row[
                                "e0_tokens"
                            ]
                            for row in rows
                        )
                        / n,
                        4,
                    ),

                "e1_avg_tokens":
                    round(
                        sum(
                            row[
                                "e1_tokens"
                            ]
                            for row in rows
                        )
                        / n,
                        4,
                    ),

                "e0_avg_latency_sec":
                    round(
                        sum(
                            row[
                                "e0_latency"
                            ]
                            for row in rows
                        )
                        / n,
                        4,
                    ),

                "e1_avg_latency_sec":
                    round(
                        sum(
                            row[
                                "e1_latency"
                            ]
                            for row in rows
                        )
                        / n,
                        4,
                    ),
            }
        )

    return summary


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    e0_root = Path(
        args.e0_root
    )

    e1_root = Path(
        args.e1_root
    )

    # -----------------------------------------------------
    # Adapter paths
    # -----------------------------------------------------

    e0_short = require_adapter(
        e0_root
        / "adapters"
        / "short_sft"
    )

    e0_long = require_adapter(
        e0_root
        / "adapters"
        / "long_sft"
    )

    e1_short = require_adapter(
        e1_root
        / "adapters"
        / "short_sft_balanced"
    )

    e1_long = require_adapter(
        e1_root
        / "adapters"
        / "long_sft_balanced"
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output_dir = (
        e1_root
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"e1_vs_e0_"
            f"validation_n{args.n}_"
            f"seed{args.seed}_outputs.jsonl"
        )
    )

    summary_csv = (
        output_dir
        / (
            f"e1_vs_e0_"
            f"validation_n{args.n}_"
            f"seed{args.seed}_summary.csv"
        )
    )

    summary_json = (
        output_dir
        / (
            f"e1_vs_e0_"
            f"validation_n{args.n}_"
            f"seed{args.seed}_summary.json"
        )
    )

    # -----------------------------------------------------
    # Validation sample
    # -----------------------------------------------------

    print(
        "Loading validation...",
        flush=True,
    )

    validation = (
        download_validation()
    )

    selected = (
        make_balanced_sample(
            validation,
            args.n,
            args.seed,
        )
    )

    print(
        "Selected examples:",
        len(selected),
        flush=True,
    )

    distribution = {}

    for row in selected:

        task = str(
            row.get(
                "task_type",
                "unknown",
            )
        )

        distribution[task] = (
            distribution.get(
                task,
                0,
            )
            + 1
        )

    print(
        "Evaluation distribution:"
    )

    print(
        json.dumps(
            distribution,
            ensure_ascii=False,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------

    existing = load_jsonl(
        output_path
    )

    completed_ids = {
        str(
            row["id"]
        )
        for row in existing
    }

    remaining = [
        row
        for row in selected
        if str(
            row.get("id")
        )
        not in completed_ids
    ]

    print(
        f"Resume: "
        f"existing={len(existing)} "
        f"remaining={len(remaining)}",
        flush=True,
    )

    # -----------------------------------------------------
    # Load model once
    # -----------------------------------------------------

    print(
        "Loading Qwen3 tokenizer...",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    print(
        "Loading Qwen3 base model...",
        flush=True,
    )

    base = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    # -----------------------------------------------------
    # Load all four adapters into one base model
    # -----------------------------------------------------

    print(
        "Loading E0/E1 adapters...",
        flush=True,
    )

    model = (
        PeftModel.from_pretrained(
            base,
            str(e0_short),
            adapter_name="e0_short",
            is_trainable=False,
        )
    )

    model.load_adapter(
        str(e0_long),
        adapter_name="e0_long",
        is_trainable=False,
    )

    model.load_adapter(
        str(e1_short),
        adapter_name="e1_short",
        is_trainable=False,
    )

    model.load_adapter(
        str(e1_long),
        adapter_name="e1_long",
        is_trainable=False,
    )

    model.eval()

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as file:

        start_n = len(existing)

        for index, row in enumerate(
            remaining,
            1,
        ):

            diff = difficulty(
                row
            )

            task = row.get(
                "task_type"
            )

            # Easy -> Short
            # Hard -> Long

            if diff == "easy":

                adapter_type = "short"

                e0_adapter = (
                    "e0_short"
                )

                e1_adapter = (
                    "e1_short"
                )

            elif diff == "hard":

                adapter_type = "long"

                e0_adapter = (
                    "e0_long"
                )

                e1_adapter = (
                    "e1_long"
                )

            else:
                continue

            # -------------------------
            # E0
            # -------------------------

            (
                e0_output,
                e0_tokens,
                e0_latency,
            ) = generate_one(
                model,
                tokenizer,
                row,
                e0_adapter,
                adapter_type,
            )

            # -------------------------
            # E1
            # -------------------------

            (
                e1_output,
                e1_tokens,
                e1_latency,
            ) = generate_one(
                model,
                tokenizer,
                row,
                e1_adapter,
                adapter_type,
            )

            record = {

                "id":
                    str(
                        row.get("id")
                    ),

                "task_type":
                    task,

                "difficulty":
                    diff,

                "adapter_type":
                    adapter_type,

                "gold":
                    row.get(
                        "answer"
                    ),

                "e0_prediction":
                    extract_final(
                        e0_output
                    ),

                "e1_prediction":
                    extract_final(
                        e1_output
                    ),

                "e0_correct":
                    is_correct(
                        e0_output,
                        row["answer"],
                        task,
                    ),

                "e1_correct":
                    is_correct(
                        e1_output,
                        row["answer"],
                        task,
                    ),

                "e0_tokens":
                    e0_tokens,

                "e1_tokens":
                    e1_tokens,

                "e0_latency":
                    round(
                        e0_latency,
                        4,
                    ),

                "e1_latency":
                    round(
                        e1_latency,
                        4,
                    ),
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            done = (
                start_n
                + index
            )

            if (
                done
                % args.checkpoint_every
                == 0
                or
                index
                == len(remaining)
            ):

                file.flush()

                os.fsync(
                    file.fileno()
                )

                print(
                    f"{done}/"
                    f"{len(selected)} "
                    "completed",
                    flush=True,
                )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    records = load_jsonl(
        output_path
    )

    summary = summarize(
        records
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            summary
        )

    result_json = {

        "experiment":
            "E1_balanced_data",

        "comparison":
            "E0 SFT vs E1 balanced SFT",

        "evaluation_split":
            "validation",

        "sample_size":
            len(records),

        "sampling":
            (
                "deterministic balanced "
                "round-robin by task_type"
            ),

        "seed":
            args.seed,

        "policy":
            (
                "Easy -> Short, "
                "Hard -> Long"
            ),

        "summary":
            summary,
    }

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_json,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        "========================================"
    )

    print(
        "E1 VS E0 FINAL SUMMARY"
    )

    print(
        "========================================"
    )

    for result in summary:

        print(
            json.dumps(
                result,
                ensure_ascii=False,
            ),
            flush=True,
        )

    print(
        "\nResults saved to:",
        output_dir,
    )


if __name__ == "__main__":
    main()