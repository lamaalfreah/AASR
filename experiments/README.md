# AASR Experiment Tracking

This directory contains the controlled experimental history of AASR.

## Experiment Index

| ID | Experiment             |
| -- | ---------------------- |
| E0 | Current Baseline       |
| E1 | Balanced Training Data |
| E2 | English Diagnostic     |
| E3 | Adapter Improvement    |
| E4 | Router Evaluation      |

## Standard Experiment Folder

Each experiment folder should contain only the files relevant to that run.

```text
README.md
command.txt
artifact_paths.txt
metrics.json
modal_log.txt
```

Not every file is mandatory.

## Recording Rules

For every new experiment:

1. Do not overwrite a previous experiment.
2. Record the exact command used to run the experiment.
3. Save the final metrics.
4. Record the Modal or Hugging Face artifact location.
5. Summarize the objective, changed variable, result, and conclusion in the experiment README.
6. Do not commit large model weights, checkpoints, or generated outputs to GitHub.

## Naming Convention

Use the experiment ID followed by a descriptive name.

Example:

```text
E3_adapter_improvement/
```

If multiple runs are needed under the same experiment, create subfolders instead of overwriting previous results.

Example:

```text
E3_adapter_improvement/
├── r32_alpha64/
└── r64_alpha128/
```

## Artifact Reuse

Before running any training experiment, check:

`../ARTIFACT_REGISTRY.md`

Completed artifacts should be reused whenever possible instead of retraining them.
