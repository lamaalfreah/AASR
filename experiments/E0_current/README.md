# E0 — Current AASR Baseline

## Status

Complete

## Objective

Document the current AASR implementation and its baseline performance before applying dataset balancing, English diagnostics, adapter improvements, or router changes.

## Base Model

`Qwen/Qwen3-4B`

## Current Adapter Configuration

* Adapter type: LoRA
* LoRA rank: 16
* LoRA alpha: 32
* LoRA dropout: 0.05
* Target modules: `q_proj`, `v_proj`
* Short max context: 4096
* Long max context: 8192
* Short Adapter: trained primarily on Easy examples
* Long Adapter: trained primarily on Hard examples
* Medium examples: excluded from primary adapter specialization

## Current Routing Setup

* Router backbone: XLM-R Base
* Routing target: adaptive mixing coefficient `alpha`
* Main alpha values:

  * `0.0`
  * `0.25`
  * `1.0`

## Current Results

Results below refer to the Router V3 holdout evaluation.

| Method             | Strict Accuracy | Avg. Tokens | Avg. Latency |
| ------------------ | --------------: | ----------: | -----------: |
| Fixed alpha = 0.0  |           47.5% |        7.38 |       0.46 s |
| Fixed alpha = 0.25 |           37.5% |        6.95 |       0.48 s |
| Fixed alpha = 1.0  |           35.0% |      120.78 |       5.87 s |
| Router V3          |           47.5% |        7.38 |       0.46 s |
| 3-alpha Oracle     |          66.25% |       19.23 |       1.04 s |

## Router Finding

Router V3 collapsed toward `alpha = 0`.

* Alpha classification accuracy: 71.25%
* Balanced accuracy: 33.33%
* The Router predicted `alpha = 0` for all holdout samples.

This means the apparently high classification accuracy was mainly caused by class imbalance rather than successful adaptive routing.

## Key Observations

* The Short Adapter is currently the strongest fixed strategy.
* The Long Adapter is substantially more expensive in tokens and latency.
* The Oracle result is considerably higher than the best fixed strategy.
* This indicates that per-query alpha selection still has potential.
* The current Router is not able to exploit this potential.
* Dataset imbalance may contribute to the routing collapse.
* The current AASR configuration differs from original AdaMix in several important aspects.

## Identified Differences from Original AdaMix

* Lower LoRA rank.
* Shorter Long-context budget.
* Simplified Short/Long specialization.
* Smaller Router supervision set.
* Different multilingual Router backbone.
* Fewer alpha values in the main routing experiment.
* Fewer generations per alpha during empirical Router supervision.

## Conclusion

E0 represents the current AASR baseline.

All future experiments should be compared against this baseline before being considered an improvement.

The next experiment is:

**E1 — Balanced Training Dataset**

The purpose of E1 is to isolate the effect of balancing the original training dataset while keeping the current model configuration unchanged.
