"""
AASR Experiment E1 - Token Length Audit
=======================================

Goal
----
Measure the real token-length distribution of the AASR training prompts
using the same tokenizer and prompt format used by the current model.

Why this audit is needed
------------------------
The current AASR configuration uses:

    Short max_length = 4096
    Long  max_length = 8192

However, the SFT pipeline reserves part of this space for the completion:

    Short completion reserve = 256 tokens
    Long completion reserve  = 2048 tokens

Therefore, the current effective prompt limits are:

    Short prompt limit = 3840 tokens
    Long prompt limit  = 6144 tokens

This audit tells us how much of the dataset fits within those limits
before truncation.

Datasets analyzed
-----------------
1. Original Easy data      -> current Short training source
2. Balanced Easy data      -> E1 Short training source
3. Original Hard data      -> current Long training source
4. Balanced Hard data      -> E1 Long training source

Medium is not included because E1 preserves the same E0 specialization:
    Easy -> Short
    Hard -> Long

Metrics
-------
For every subset we calculate:

    Mean
    P50
    P90
    P95
    P99
    Maximum

We also calculate:
    - Number of examples above the current prompt limit
    - Percentage of examples above the current prompt limit

Important
---------
Token lengths are measured BEFORE truncation.

This allows us to determine whether the current Short/Long context
limits are appropriate for the actual dataset.

Outputs
-------
token_length_audit.json
token_length_audit.csv
"""

import csv
import gzip
import io
import json
import math
import os
from pathlib import Path

import requests
from transformers import AutoTokenizer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-4B"

DATA_URL = (
    "https://huggingface.co/datasets/"
    "ArabicSpatialrReasoning/arabic-spatial-reasoning-v1/"
    "resolve/main/train.jsonl.gz"
)

E1_DIR = Path(
    os.environ.get(
        "E1_OUTPUT_DIR",
        Path(__file__).resolve().parent,
    )
)

# Current E0 SFT configuration
SHORT_MAX_LENGTH = 4096
LONG_MAX_LENGTH = 8192

SHORT_COMPLETION_RESERVE = 256
LONG_COMPLETION_RESERVE = 2048

SHORT_PROMPT_LIMIT = (
    SHORT_MAX_LENGTH
    - SHORT_COMPLETION_RESERVE
)

LONG_PROMPT_LIMIT = (
    LONG_MAX_LENGTH
    - LONG_COMPLETION_RESERVE
)


SYSTEM_PROMPT = """أنت نموذج متخصص في الاستدلال المكاني باللغة العربية.
اعتمد فقط على السياق المعطى.
يجب أن تنتهي إجابتك دائمًا بالسطر:
FINAL: <الإجابة>
"""


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_original_train():
    """
    Load the original AASR TRAIN file from private Hugging Face.

    Returns
    -------
    list[dict]
        Original training records.

    Notes
    -----
    HF_TOKEN must exist in the environment because the dataset
    repository is private.
    """

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is required."
        )

    response = requests.get(
        DATA_URL,
        headers={
            "Authorization":
                f"Bearer {token}"
        },
        timeout=300,
    )

    response.raise_for_status()

    rows = []

    with gzip.GzipFile(
        fileobj=io.BytesIO(
            response.content
        )
    ) as gz:

        for raw_line in gz:

            line = raw_line.decode(
                "utf-8"
            ).strip()

            if line:

                rows.append(
                    json.loads(line)
                )

    return rows


def load_jsonl(path):
    """
    Load a local JSONL dataset.

    Parameters
    ----------
    path : Path
        Path to the JSONL file.

    Returns
    -------
    list[dict]
        Dataset records.
    """

    rows = []

    with path.open(
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
# Metadata helpers
# ---------------------------------------------------------------------

def difficulty(row):
    """
    Return normalized difficulty.
    """

    return str(
        row.get(
            "difficulty",
            "",
        )
    ).strip().lower()


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

def build_messages(row):
    """
    Build the same system/user messages used by the current AASR model.

    The task_type and difficulty metadata are NOT added to the prompt.
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


def count_prompt_tokens(
    tokenizer,
    row,
    reasoning_type,
):
    """
    Count tokens in the complete rendered prompt BEFORE truncation.

    Parameters
    ----------
    tokenizer :
        Qwen3 tokenizer.

    row : dict
        One dataset example.

    reasoning_type : str
        "short" or "long".

    Returns
    -------
    int
        Number of prompt tokens.

    Notes
    -----
    Short prompts use:
        enable_thinking=False

    Long prompts use:
        enable_thinking=True

    This matches the current AASR SFT pipeline.
    """

    thinking = (
        reasoning_type == "long"
    )

    prompt = (
        tokenizer.apply_chat_template(
            build_messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    )

    token_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    return len(token_ids)


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def percentile(values, percent):
    """
    Calculate a percentile using linear interpolation.

    Parameters
    ----------
    values : list[int]
        Token lengths.

    percent : float
        Percentile from 0 to 100.

    Returns
    -------
    float
        Percentile value.
    """

    values = sorted(values)

    if not values:
        return 0.0

    position = (
        (len(values) - 1)
        * percent
        / 100
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(
            values[lower]
        )

    weight = (
        position - lower
    )

    return (
        values[lower]
        * (1 - weight)
        + values[upper]
        * weight
    )


def summarize_lengths(
    lengths,
    prompt_limit,
):
    """
    Summarize one token-length distribution.

    Returns:
        number of examples,
        mean,
        percentiles,
        maximum,
        and current prompt-limit coverage.
    """

    if not lengths:
        raise ValueError(
            "No token lengths provided."
        )

    over_limit = sum(
        length > prompt_limit
        for length in lengths
    )

    return {
        "examples":
            len(lengths),

        "mean":
            round(
                sum(lengths)
                / len(lengths),
                2,
            ),

        "p50":
            round(
                percentile(
                    lengths,
                    50,
                ),
                2,
            ),

        "p90":
            round(
                percentile(
                    lengths,
                    90,
                ),
                2,
            ),

        "p95":
            round(
                percentile(
                    lengths,
                    95,
                ),
                2,
            ),

        "p99":
            round(
                percentile(
                    lengths,
                    99,
                ),
                2,
            ),

        "max":
            max(lengths),

        "current_prompt_limit":
            prompt_limit,

        "examples_over_limit":
            over_limit,

        "percent_over_limit":
            round(
                100
                * over_limit
                / len(lengths),
                2,
            ),

        "percent_within_limit":
            round(
                100
                * (
                    len(lengths)
                    - over_limit
                )
                / len(lengths),
                2,
            ),
    }


def audit_subset(
    name,
    rows,
    tokenizer,
    reasoning_type,
    prompt_limit,
):
    """
    Calculate token lengths for one AASR subset
    and print its statistical summary.
    """

    print(
        "\n" + "=" * 70
    )

    print(name)

    print(
        "=" * 70
    )

    print(
        "Examples:",
        len(rows),
    )

    lengths = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        length = count_prompt_tokens(
            tokenizer,
            row,
            reasoning_type,
        )

        lengths.append(length)

        if index % 1000 == 0:

            print(
                f"Processed "
                f"{index}/"
                f"{len(rows)}"
            )

    summary = summarize_lengths(
        lengths,
        prompt_limit,
    )

    print(
        "\nToken length summary:"
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    return summary


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_csv(results):
    """
    Save all subset statistics into one CSV file.
    """

    path = (
        E1_DIR
        / "token_length_audit.csv"
    )

    fields = [
        "subset",
        "examples",
        "mean",
        "p50",
        "p90",
        "p95",
        "p99",
        "max",
        "current_prompt_limit",
        "examples_over_limit",
        "percent_over_limit",
        "percent_within_limit",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for subset, metrics in (
            results.items()
        ):

            writer.writerow(
                {
                    "subset": subset,
                    **metrics,
                }
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Run the complete E1 token-length audit.

    The audit compares original and balanced
    Short/Long training subsets using the exact
    Qwen3 tokenizer.
    """

    print(
        "Loading Qwen3 tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    print(
        "Loading original TRAIN..."
    )

    original_train = (
        load_original_train()
    )

    original_easy = [
        row
        for row in original_train
        if difficulty(row) == "easy"
    ]

    original_hard = [
        row
        for row in original_train
        if difficulty(row) == "hard"
    ]

    balanced_easy = load_jsonl(
        E1_DIR
        / "balanced_short_train.jsonl"
    )

    balanced_hard = load_jsonl(
        E1_DIR
        / "balanced_long_train.jsonl"
    )

    print(
        "\nCurrent effective prompt limits:"
    )

    print(
        "Short:",
        SHORT_PROMPT_LIMIT,
    )

    print(
        "Long:",
        LONG_PROMPT_LIMIT,
    )

    results = {}

    results[
        "original_easy_short"
    ] = audit_subset(
        name=(
            "ORIGINAL EASY / SHORT"
        ),
        rows=original_easy,
        tokenizer=tokenizer,
        reasoning_type="short",
        prompt_limit=
            SHORT_PROMPT_LIMIT,
    )

    results[
        "balanced_easy_short"
    ] = audit_subset(
        name=(
            "BALANCED EASY / SHORT"
        ),
        rows=balanced_easy,
        tokenizer=tokenizer,
        reasoning_type="short",
        prompt_limit=
            SHORT_PROMPT_LIMIT,
    )

    results[
        "original_hard_long"
    ] = audit_subset(
        name=(
            "ORIGINAL HARD / LONG"
        ),
        rows=original_hard,
        tokenizer=tokenizer,
        reasoning_type="long",
        prompt_limit=
            LONG_PROMPT_LIMIT,
    )

    results[
        "balanced_hard_long"
    ] = audit_subset(
        name=(
            "BALANCED HARD / LONG"
        ),
        rows=balanced_hard,
        tokenizer=tokenizer,
        reasoning_type="long",
        prompt_limit=
            LONG_PROMPT_LIMIT,
    )

    output = {
        "model":
            MODEL_ID,

        "measurement":
            "Rendered prompt tokens before truncation",

        "short_max_length":
            SHORT_MAX_LENGTH,

        "short_completion_reserve":
            SHORT_COMPLETION_RESERVE,

        "short_prompt_limit":
            SHORT_PROMPT_LIMIT,

        "long_max_length":
            LONG_MAX_LENGTH,

        "long_completion_reserve":
            LONG_COMPLETION_RESERVE,

        "long_prompt_limit":
            LONG_PROMPT_LIMIT,

        "results":
            results,
    }

    with (
        E1_DIR
        / "token_length_audit.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    save_csv(results)

    print(
        "\n" + "=" * 70
    )

    print(
        "TOKEN LENGTH AUDIT COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "Results saved to:",
        E1_DIR,
    )


if __name__ == "__main__":
    main()