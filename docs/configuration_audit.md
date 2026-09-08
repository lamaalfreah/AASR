# AASR Configuration Audit

## Purpose

Document the important configuration differences between the current AASR implementation and the original AdaMix framework.

The goal is **not** to reproduce AdaMix exactly. Configuration changes may be retained when they are better suited to:

* Arabic language processing
* Spatial reasoning
* Long-context characteristics of the AASR dataset
* Available computational resources
* Empirical performance

## Configuration Comparison

| Component            | Current AASR                         | Original AdaMix                               | Review / Decision                         |
| -------------------- | ------------------------------------ | --------------------------------------------- | ----------------------------------------- |
| Base Model           | Qwen3-4B                             | Multiple backbones, including Qwen3-4B        | Keep                                      |
| Adapter Type         | LoRA                                 | LoRA                                          | Keep                                      |
| LoRA Rank            | 16                                   | 64                                            | Review                                    |
| LoRA Alpha           | 32                                   | 128                                           | Review together with rank                 |
| LoRA Dropout         | 0.05                                 | 0.05                                          | Keep                                      |
| Target Modules       | `q_proj`, `v_proj`                   | `q_proj`, `v_proj`                            | Keep                                      |
| Short Context        | 4096                                 | 4096                                          | Keep                                      |
| Long Context         | 8192                                 | 24576                                         | Review using AASR token-length statistics |
| Short Specialization | Easy samples                         | Empirically selected concise-solvable samples | Review                                    |
| Long Specialization  | Hard samples                         | Empirical reasoning-frontier samples          | Review                                    |
| Medium Samples       | Excluded from primary specialization | No direct equivalent rule                     | Review                                    |
| Router Backbone      | XLM-R Base                           | BERT Base                                     | Multilingual adaptation                   |
| Router Supervision   | Limited                              | Larger empirical supervision set              | Review                                    |
| Alpha Grid           | Reduced                              | Dense 0–1 grid                                | Reduced for computational efficiency      |

## Main Issues Identified

### 1. Adapter Capacity

The current LoRA rank is lower than the original AdaMix configuration.

Current:

```text
r = 16
LoRA alpha = 32
```

Original AdaMix:

```text
r = 64
LoRA alpha = 128
```

The current setup preserves the same alpha-to-rank ratio, but the adapters have lower trainable capacity.

This should be reviewed empirically rather than changed automatically.

---

### 2. Long-Context Budget

The current Long Adapter uses:

```text
8192 tokens
```

while the original AdaMix configuration uses a substantially larger context budget.

AASR should not automatically copy the original value.

The final Long context length should instead be selected using the actual token-length distribution of the AASR dataset, such as:

* P50
* P90
* P95
* P99

---

### 3. Short / Long Specialization

The current specialization rule is primarily:

```text
Easy → Short
Hard → Long
Medium → Excluded
```

This is simpler than the empirical specialization strategy used in AdaMix.

Future review should determine whether Short and Long training samples can be selected based on actual model reasoning behavior instead of dataset difficulty metadata alone.

---

### 4. Router Supervision

The current Router was trained with substantially less empirical supervision than the original AdaMix routing setup.

The latest Router also collapsed toward the majority `alpha = 0` class.

Potential contributing factors include:

* Class imbalance
* Limited Router training examples
* Limited alpha sampling
* Limited generations per alpha
* Insufficient separation between Short and Long adapter behavior

Router retraining should therefore be delayed until the adapter configuration and training data are reviewed.

---

### 5. Router Backbone

Current Router:

```text
XLM-R Base
```

Original AdaMix Router:

```text
BERT Base
```

This difference is intentional because AASR operates primarily on Arabic data.

XLM-R is retained as the current multilingual baseline.

A different multilingual Router should only be tested if routing collapse continues after data and adapter improvements.

## Decision Principle

A configuration should **not** be adopted only because it matches the original AdaMix paper.

The final AASR configuration should be selected based on:

* Arabic-language suitability
* Spatial reasoning performance
* Long-context coverage
* Accuracy
* Routing quality
* Token efficiency
* Latency
* Computational feasibility

## Current Review Priorities

1. Balance the original training dataset.
2. Analyze token-length statistics before changing the Long context budget.
3. Run a low-cost English diagnostic.
4. Evaluate one improved adapter configuration rather than performing a large hyperparameter grid.
5. Revisit Router training only after stronger Short and Long adapters are established.
