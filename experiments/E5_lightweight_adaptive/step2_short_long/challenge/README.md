# Hard Natural Arabic Challenge Set

The existing 320 CORE questions retain their exact text, IDs, source references, labels, and DEV/EVAL split. This designation does not regenerate or alter any dataset file.

- DEV: 192 questions, 24 per task.
- Held-out EVAL: 128 questions, 16 per task.
- Rule Baseline 0: frozen Step-1 template parser, 0/320 parsed/correct; this documents template dependence.
- Neural Short: the separately trained BiGRU intent model plus deterministic argument extraction.

See the [consolidated comparison](../../step2_neural_short/BASELINE_COMPARISON.md) and [designation metadata](designation.json). CORE EVAL was excluded from training and model selection; its completed results must not be used to tune this frozen run. Long and Router work remain paused.
