# AASR Artifact Registry

This registry tracks trained models and large experiment artifacts stored outside GitHub.

Before starting any training or evaluation job, check this file first.

## Modal Storage

**Main Volume:** `asar-artifacts`
**Hugging Face Cache Volume:** `asar-hf-cache`

---

## Current Artifacts

| Artifact                        | Status               | Configuration         | Storage | Verified Path                                             | Notes                                                                 |
| ------------------------------- | -------------------- | --------------------- | ------- | --------------------------------------------------------- | --------------------------------------------------------------------- |
| Short SFT Adapter               | Complete             | LoRA r16 / alpha32    | Modal   | `/asar_sft_grpo/adapters/short_sft`                       | Current Short Adapter                                                 |
| Long SFT Adapter                | Complete             | LoRA r16 / alpha32    | Modal   | `/asar_sft_grpo/adapters/long_sft`                        | Current Long Adapter                                                  |
| Continuous Router               | Complete             | XLM-R                 | Modal   | `/asar_sft_grpo/empirical_router/continuous_router/model` | Earlier continuous routing model                                      |
| Alpha Router V2                 | Complete             | Earlier alpha router  | Modal   | `/asar_sft_grpo/alpha_router_v2`                          | Directory exists; internal model path should be verified before reuse |
| Alpha Router V3                 | Complete — Collapsed | XLM-R, 3-alpha router | Modal   | `/asar_sft_grpo/alpha_router_v3/model`                    | Final checkpoint exists; Router collapsed toward alpha = 0            |
| Final AdaMix Evaluation Outputs | Complete             | SFT, alpha mixing     | Modal   | `/asar_sft_grpo/final_evaluation_adamix_fast`             | Contains baseline validation outputs                                  |
| Balanced Dataset                | Pending              | TBD                   | —       | —                                                         | E1                                                                    |
| Improved Short Adapter          | Pending              | Planned r32 / alpha64 | —       | —                                                         | E3                                                                    |
| Improved Long Adapter           | Pending              | Planned r32 / alpha64 | —       | —                                                         | E3                                                                    |
| Final Router                    | Pending              | TBD                   | —       | —                                                         | E4                                                                    |

---

## Verified Model Files

### Alpha Router V3

```text
/asar_sft_grpo/alpha_router_v3/model/
├── tokenizer/
└── alpha_router_v3.pt
```

The model checkpoint is approximately 1 GB.

### Continuous Router

```text
/asar_sft_grpo/empirical_router/continuous_router/model/
├── tokenizer/
├── continuous_router.pt
└── continuous_router_metrics.json
```

---

## Reuse Policy

Do **not** retrain an artifact marked `Complete` unless:

1. The experiment intentionally changes the configuration.
2. The saved artifact is unavailable or corrupted.
3. The experiment specifically requires a new training dataset or training strategy.

Evaluation experiments should reuse existing trained adapters and routers whenever possible.

---

## GitHub vs Modal

### GitHub Stores

* Source code
* Modal runner scripts
* Experiment configurations
* Commands
* Metrics
* Important logs
* Experiment conclusions
* Artifact paths

### Modal Stores

* Adapter weights
* Router checkpoints
* Large model outputs
* Intermediate checkpoints
* Large generated evaluation files

---

## Artifact Naming Convention

New artifacts should use descriptive names that include the main configuration change.

Examples:

```text
short_balanced_r32
long_balanced_r32
router_balanced_xlmr
router_balanced_mdeberta
```
So, avoid overwriting previous artifacts.
