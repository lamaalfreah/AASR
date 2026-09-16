"""
AASR Experiment E1 - Balanced Training Data

Goal
----
Test whether task-type imbalance affects the current Short and Long adapters.

We preserve the same E0 specialization:

    Easy   -> Short Adapter
    Hard   -> Long Adapter
    Medium -> Audit only

Balancing method
----------------
Random undersampling is performed separately inside Easy and Hard.

For each difficulty:
1. Group examples by task_type.
2. Find the smallest task group.
3. Sample the same number from every task.
4. Shuffle the final subset.

Important
---------
- No oversampling.
- No duplicated examples.
- No synthetic data.
- Validation and test are NOT modified.
- Medium is NOT used for training in E1.

Outputs
-------
balanced_short_train.jsonl
balanced_long_train.jsonl
distribution_before.csv
distribution_after.csv
balancing_summary.json
"""

import csv
import gzip
import io
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import requests


DATA_URL = (
    "https://huggingface.co/datasets/"
    "ArabicSpatialrReasoning/arabic-spatial-reasoning-v1/"
    "resolve/main/train.jsonl.gz"
)

SEED = 42

OUTPUT_DIR = Path(
    os.environ.get(
        "E1_OUTPUT_DIR",
        Path(__file__).resolve().parent,
    )
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """
    Load the private AASR dataset from Hugging Face.

    HF_TOKEN must be available in the environment.
    """

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError("HF_TOKEN is required.")

    response = requests.get(
        DATA_URL,
        headers={
            "Authorization": f"Bearer {token}"
        },
        timeout=300,
    )

    response.raise_for_status()

    rows = []

    with gzip.GzipFile(
        fileobj=io.BytesIO(response.content)
    ) as gz:

        for raw_line in gz:
            line = raw_line.decode("utf-8").strip()

            if line:
                rows.append(json.loads(line))

    return rows


def split_name(row):
    """Return normalized train/validation/test label."""

    return str(
        row.get("split", "")
    ).strip().lower()


def difficulty(row):
    """Return normalized difficulty label."""

    return str(
        row.get("difficulty", "")
    ).strip().lower()


def task_type(row):
    """Return the example's spatial task type."""

    return str(
        row.get("task_type", "")
    ).strip()


def task_counts(rows):
    """Count examples for every task type."""

    return Counter(
        task_type(row)
        for row in rows
    )


def print_tasks(title, rows):
    """
    Print total examples and task distribution
    for one dataset subset.
    """

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    print("Total:", len(rows))

    for task, count in sorted(
        task_counts(rows).items()
    ):
        print(
            f"{task:35} {count}"
        )


def balance_by_task(rows, seed):
    """
    Balance task types using deterministic undersampling.

    Every task is reduced to the size of the
    smallest task group.

    Parameters
    ----------
    rows:
        Examples from one difficulty only.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    balanced_rows:
        Balanced subset.

    target:
        Number of examples retained per task.
    """

    rng = random.Random(seed)

    groups = defaultdict(list)

    for row in rows:
        groups[task_type(row)].append(row)

    if not groups:
        raise ValueError(
            "No task groups found."
        )

    target = min(
        len(group)
        for group in groups.values()
    )

    balanced_rows = []

    for task in sorted(groups):

        group = list(groups[task])

        rng.shuffle(group)

        balanced_rows.extend(
            group[:target]
        )

    rng.shuffle(balanced_rows)

    return balanced_rows, target


def save_jsonl(rows, path):
    """Save examples as UTF-8 JSONL."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_distribution_csv(
    datasets,
    path,
):
    """
    Save before/after task counts.

    datasets is a dictionary:
        subset_name -> rows
    """

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "subset",
                "difficulty",
                "task_type",
                "count",
            ]
        )

        for name, rows in datasets.items():

            for task, count in sorted(
                task_counts(rows).items()
            ):

                diff = (
                    difficulty(rows[0])
                    if rows
                    else ""
                )

                writer.writerow(
                    [
                        name,
                        diff,
                        task,
                        count,
                    ]
                )


def main():
    """
    Run the full E1 balancing workflow.

    Workflow:
    1. Load dataset.
    2. Preserve original splits.
    3. Audit Easy, Medium, and Hard.
    4. Balance Easy for Short Adapter.
    5. Balance Hard for Long Adapter.
    6. Save datasets and metrics.
    """

    print("Loading AASR dataset...")

    rows = load_dataset()

    train = [
        row
        for row in rows
        if split_name(row) == "train"
    ]

    validation = [
        row
        for row in rows
        if split_name(row) == "validation"
    ]

    test = [
        row
        for row in rows
        if split_name(row) == "test"
    ]

    easy = [
        row
        for row in train
        if difficulty(row) == "easy"
    ]

    medium = [
        row
        for row in train
        if difficulty(row) == "medium"
    ]

    hard = [
        row
        for row in train
        if difficulty(row) == "hard"
    ]

    print("\nOriginal split sizes")
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))

    print_tasks(
        "ORIGINAL EASY / SHORT DATA",
        easy,
    )

    print_tasks(
        "ORIGINAL MEDIUM DATA - AUDIT ONLY",
        medium,
    )

    print_tasks(
        "ORIGINAL HARD / LONG DATA",
        hard,
    )

    # Balance Easy tasks
    balanced_easy, easy_target = (
        balance_by_task(
            easy,
            seed=SEED,
        )
    )

    # Balance Hard tasks
    balanced_hard, hard_target = (
        balance_by_task(
            hard,
            seed=SEED + 1,
        )
    )

    print_tasks(
        "BALANCED EASY / SHORT DATA",
        balanced_easy,
    )

    print_tasks(
        "BALANCED HARD / LONG DATA",
        balanced_hard,
    )

    # Remove old incorrect output if it exists.
    old_file = (
        OUTPUT_DIR
        / "balanced_train.jsonl"
    )

    if old_file.exists():
        old_file.unlink()

    # Save balanced adapter datasets.
    save_jsonl(
        balanced_easy,
        OUTPUT_DIR
        / "balanced_short_train.jsonl",
    )

    save_jsonl(
        balanced_hard,
        OUTPUT_DIR
        / "balanced_long_train.jsonl",
    )

    save_distribution_csv(
        {
            "original_easy": easy,
            "original_medium": medium,
            "original_hard": hard,
        },
        OUTPUT_DIR
        / "distribution_before.csv",
    )

    save_distribution_csv(
        {
            "balanced_easy":
                balanced_easy,

            "balanced_hard":
                balanced_hard,
        },
        OUTPUT_DIR
        / "distribution_after.csv",
    )

    summary = {
        "experiment":
            "E1_balanced_data",

        "seed":
            SEED,

        "balancing_method":
            "Task-type undersampling "
            "within Easy and Hard separately",

        "original_train_size":
            len(train),

        "original_easy_size":
            len(easy),

        "original_medium_size":
            len(medium),

        "original_hard_size":
            len(hard),

        "balanced_short_size":
            len(balanced_easy),

        "balanced_long_size":
            len(balanced_hard),

        "short_examples_per_task":
            easy_target,

        "long_examples_per_task":
            hard_target,

        "medium_used_for_training":
            False,

        "validation_modified":
            False,

        "test_modified":
            False,

        "oversampling":
            False,

        "synthetic_data":
            False,
    }

    with (
        OUTPUT_DIR
        / "balancing_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("E1 DATA BALANCING COMPLETED")
    print("=" * 65)

    print(
        "Balanced Short:",
        len(balanced_easy),
    )

    print(
        "Balanced Long:",
        len(balanced_hard),
    )

    print(
        "Medium used for training: NO"
    )

    print(
        "Validation modified: NO"
    )

    print(
        "Test modified: NO"
    )

    print(
        "\nSaved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()