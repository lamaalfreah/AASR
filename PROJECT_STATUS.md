# AASR Project Status

## Project

Arabic Adaptive Spatial Reasoning (AASR)

## Current Stage

Diagnostic review and model improvement.

## Base Model

`Qwen/Qwen3-4B`

## Current Architecture

```text
Qwen3-4B
   |
   +-- Short LoRA Adapter
   |
   +-- Long LoRA Adapter
   |
   +-- Alpha Mixing
   |
   +-- Adaptive Router
```

## Current Adapter Configuration

* LoRA rank: 16
* LoRA alpha: 32
* LoRA dropout: 0.05
* Target modules: `q_proj`, `v_proj`
* Short max context: 4096
* Long max context: 8192

## Current Findings

* Fixed alpha = 0 is currently the strongest fixed strategy.
* Router V3 collapsed toward alpha = 0.
* Oracle routing shows that per-query alpha selection still has potential.
* Dataset imbalance may affect current behavior.
* Current AASR differs from original AdaMix in adapter capacity, long-context budget, specialization strategy, and router supervision.

## Planned Experiments

| ID | Experiment                     | Status   |
| -- | ------------------------------ | -------- |
| E0 | Current Baseline               | Complete |
| E1 | Balanced Training Dataset      | Pending  |
| E2 | English Diagnostic             | Pending  |
| E3 | Improved Adapter Configuration | Pending  |
| E4 | Router Sanity Check            | Pending  |

## Storage

Large model artifacts and checkpoints are stored in Modal.

See `ARTIFACT_REGISTRY.md` for exact artifact locations.
