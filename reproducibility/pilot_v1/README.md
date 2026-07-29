# Pilot Study v1

This directory freezes the initial Pashto-to-English NMT study before
the extended A/A*-conference experiments.

## Systems

- Baseline NLLB
- Original LoRA
- Semantic-filtered LoRA

## Evaluation

- 100 shared test sentences
- BLEU
- chrF++
- TER
- COMET
- Bootstrap significance testing
- Sentence-level pairwise comparison
- Automatic diagnostic indicators

## Important interpretation

The semantic-filtered LoRA system achieved the highest BLEU and chrF++
point estimates. Its COMET result remained approximately equal to the
baseline. The pilot study does not establish statistically significant
superiority over the baseline.

## Snapshot files

- requirements-lock.txt
- runtime_environment.json
- file_checksums.csv
- git_status_before_snapshot.txt
- git_commit.txt
