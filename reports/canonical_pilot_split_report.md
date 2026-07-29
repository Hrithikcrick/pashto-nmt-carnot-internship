# Canonical Pilot Dataset Splits

Fixed dataset splits were created for reproducible experiments.

- Random seed: `42`
- Training proportion: `0.9`
- Train rows: `9002`
- Validation rows: `987`
- Test rows: `100`
- Training duplicate pairs removed: `1`
- Test duplicate pairs removed: `0`
- Exact test pairs removed from the training pool: `8`
- Additional same-source rows removed: `2`

## Leakage verification

| Comparison | Exact-pair overlap | Pashto-source overlap | English-target overlap |
|---|---:|---:|---:|
| train_vs_validation | 0 | 0 | 28 |
| train_vs_test | 0 | 0 | 0 |
| validation_vs_test | 0 | 0 | 0 |

Exact-pair and Pashto-source overlaps must remain zero.
English-target overlap is reported but is not automatically treated as leakage because common target sentences can occur independently.