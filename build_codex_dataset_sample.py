"""
Create a small representative sample of the AASR dataset
for architecture review by Codex.

Output:
    dataset_sample_for_architecture_review.jsonl

Sampling:
    10 examples from each task_type
    = up to 80 examples total

Important:
    All original dataset fields are preserved so Codex
    can inspect the real schema.
"""

import gzip
import io
import json
import os
import random
from collections import defaultdict

import requests


DATA_URL = (
    "https://huggingface.co/datasets/"
    "ArabicSpatialrReasoning/"
    "arabic-spatial-reasoning-v1/"
    "resolve/main/train.jsonl.gz"
)

OUTPUT_FILE = "dataset_sample_for_architecture_review.jsonl"

SAMPLES_PER_TASK = 10
SEED = 42


# ---------------------------------------------------------
# Download private TRAIN dataset
# ---------------------------------------------------------

token = os.environ.get("HF_TOKEN")

headers = (
    {"Authorization": f"Bearer {token}"}
    if token
    else {}
)

print("Downloading dataset...")

response = requests.get(
    DATA_URL,
    headers=headers,
    timeout=300,
)

response.raise_for_status()


# ---------------------------------------------------------
# Read JSONL.GZ
# ---------------------------------------------------------

rows = []

with gzip.GzipFile(
    fileobj=io.BytesIO(response.content)
) as gz:

    for raw in gz:

        line = raw.decode("utf-8").strip()

        if line:
            rows.append(json.loads(line))


print("Loaded:", len(rows))


# ---------------------------------------------------------
# Group by task_type
# ---------------------------------------------------------

groups = defaultdict(list)

for row in rows:

    task = str(
        row.get("task_type", "unknown")
    )

    groups[task].append(row)


# ---------------------------------------------------------
# Sample 10 from each task
# ---------------------------------------------------------

rng = random.Random(SEED)

sample = []

for task in sorted(groups):

    task_rows = groups[task]

    rng.shuffle(task_rows)

    selected = task_rows[:SAMPLES_PER_TASK]

    sample.extend(selected)

    print(
        f"{task}: {len(selected)} samples"
    )


# Shuffle final sample so tasks are not grouped
rng.shuffle(sample)


# ---------------------------------------------------------
# Save ALL original fields
# ---------------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:

    for row in sample:

        file.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print("Done.")
print("Total sample:", len(sample))
print("Saved to:", OUTPUT_FILE)

if sample:
    print()
    print("Fields found:")
    for key in sample[0].keys():
        print(" -", key)
        