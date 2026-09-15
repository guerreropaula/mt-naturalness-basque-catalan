# Addressing Naturalness in LLM Translation for Low-resource Languages

This repository contains the reproducibility package for a master's thesis on English-to-Basque (EN-EU) and English-to-Catalan (EN-CA) machine translation. The study asks whether prompting, supervised fine-tuning (SFT), and Group Relative Policy Optimization (GRPO) can reduce signs of machine translationese while retaining translation quality.

The release is intentionally smaller than the working research directory. It keeps the code and configuration that define the experiments, but omits plotting utilities, scheduler-specific job files, exploratory scripts, caches, checkpoints, and private or copyrighted text. Nothing in the original project was removed when this folder was created.

## Experimental design

| Phase | Condition | Method |
|---|---|---|
| Prompting | P0 | Minimal direct-translation baseline |
| Prompting | P1 | Naturalness-aware instruction |
| Prompting | P2 | Direct translation followed by self-polishing |
| Prompting | P3 | Research, drafting, refinement, and proofreading pipeline |
| Post-training | P4 | QLoRA supervised fine-tuning |
| Post-training | P5 | GRPO with translation-quality, diversity, length, and optional learned-naturalness rewards |

The exact prompts are in `configs/experiments.yaml`. P4 and P5 share the direct translation template in `configs/sft.yaml` and `configs/grpo.yaml`. P5 contains A0-A5 and the A3 estimator variants, including sentence classifiers, five-sentence classifiers, and contrastive language-model scorers.

## Repository layout

```text
configs/                 Experiment, model, data, SFT, GRPO, and estimator settings
results/                 In-domain, out-of-domain, and significance aggregates
src/data/                Corpus loading, cleaning, and fixed split construction
src/prompting/           P0-P3 prompting experiments
src/sft/                 P4 data preparation, training, and deterministic evaluation
src/grpo/                P5 rewards, training, and deterministic evaluation
src/classifiers/         HT-versus-MT discriminative naturalness estimators
src/contrastive_lm/      HT-versus-MT contrastive causal-LM estimators
src/evaluation/run.py    Complete translation-quality and naturalness evaluation
src/evaluation/metrics/  Automatic, lexical, morphological, and syntactic metrics
src/evaluation/statistics/ Paired significance tests and corpus bootstrap
src/evaluation/compare.py Build aggregate P0-P5 comparison tables
src/utils/               Shared configuration, I/O, authentication, and model loading
```

Plotting code, scheduler files, repair scripts, diagnostics, and one-off dataset-construction programs are not included. The repository focuses on the reusable path required to clean the source corpora, train systems, generate translations, evaluate them, and test paired differences. The shared `src/utils/` package is retained because every experimental stage uses the same configuration and model-loading rules.

## Models

The model registry in `configs/models.yaml` records the exact Hugging Face identifiers and model-specific prompt adapters for:

- `HiTZ/Latxa-Llama-3.1-8B-Instruct`
- `HiTZ/Latxa-Llama-3.1-70B-Instruct`
- `BSC-LT/salamandraTA-7b-instruct`
- `BSC-LT/salamandra-7b-instruct`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-32B`
- `google/gemma-3-12b-it`
- `google/gemma-3-27b-it`

The inference loader preserves each model's native chat template. SalamandraTA uses its task-specific translation format, Gemma 3 uses its processor interface, and Qwen 3 explicitly disables thinking for deterministic translation evaluation.

## Installation

Python 3.11 is required. Create an isolated environment and install only the extras needed for a task:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,inference,training,evaluation,dev]'
```

For GRPO, add the GRPO dependencies:

```bash
python -m pip install -e '.[grpo]'
```

ASTrED has older, sensitive dependencies and should be installed in a separate environment:

```bash
python -m pip install -e '.[astred]'
python -m src.evaluation.astred_compat --apply
```

Some models are gated. Put a Hugging Face token in the environment, never in the repository:

```bash
export HF_TOKEN=your_token
```

## Data access

The thesis corpora are not redistributed. Configure authorized local corpus paths in `configs/datasets.yaml` before running preprocessing. Input records must provide the source and target fields mapped in that configuration; Catalan records must also expose the domain metadata used to retain HRM and CUL.

For each language direction, the post-training pool contains 76,000 SFT training pairs, 2,000 SFT development pairs, 16,000 GRPO training pairs, and 2,000 GRPO development pairs. A separate 2,000-pair `test` split is used for all in-domain P0-P5 comparisons, and 70,000 disjoint target texts form the lexical-frequency background. There are no SFT or GRPO test partitions and no second global holdout in this release. The classifier and contrastive-LM development and test files are retained because they are used to select and evaluate the naturalness estimators.

## Quick reproduction

After obtaining the corpora and the FastText language-identification model, configure their local paths and preprocess both directions:

```bash
python -m src.data.preprocessing --dataset en_eu --force
python -m src.data.preprocessing --dataset en_ca --force
python -m src.data.build_training_splits --dataset all --force
```

Run a prompting condition by selecting a dataset and model key. The example below uses the in-domain test set:

```bash
python -m src.prompting.p0.run --dataset en_eu --model latxa_8b_instruct --split test
python -m src.prompting.p1.run --dataset en_eu --model latxa_8b_instruct --split test
python -m src.prompting.p2.run --dataset en_eu --model latxa_8b_instruct --split test
python -m src.prompting.p3.run --dataset en_eu --model latxa_8b_instruct --split test
```

Prepare, train, and evaluate P4:

```bash
python -m src.sft.prepare_data --dataset en_eu --force
python -m src.sft.train --dataset en_eu --model latxa_8b_instruct
python -m src.sft.evaluate --dataset en_eu --model latxa_8b_instruct --split test
```

Train and evaluate P5 A2 from the P4 adapter:

```bash
python -m src.grpo.train --dataset en_eu --model latxa_8b_instruct --ablation a2
python -m src.grpo.evaluate --dataset en_eu --model latxa_8b_instruct --ablation a2 --split test
```

P0-P4 evaluation is deterministic (`do_sample: false`, one beam). GRPO training samples eight completions per source at temperature 0.6 because group-relative rewards and Self-BLEU require variation within each group.

Evaluate a prediction file and build the aggregate comparison tables:

```bash
python -m src.evaluation.run --predictions path/to/predictions.jsonl --output-dir path/to/evaluation
python -m src.evaluation.compare --results-root results --datasets en_eu en_ca
```

The evaluation command accepts optional model arguments for COMET, XCOMET, COMETKiwi, and MetricX, together with language and resource arguments for Stanza, ASTrED, and SFA. Sentence-level paired tests and corpus-level lexical bootstrap use separate entry points because they estimate uncertainty for different statistical units:

```bash
python -m src.evaluation.statistics.paired --dataset en_eu --model latxa_8b_instruct --baseline P5_GRPO_A2 --candidates P5_GRPO_A5
python -m src.evaluation.statistics.corpus --dataset en_eu --model latxa_8b_instruct --baseline P5_GRPO_A2 --candidates P5_GRPO_A5 --iterations 10000
```


## Evaluation

The evaluation pipeline combines complementary evidence:

- Translation quality: BLEU, chrF++, TER, COMET, XCOMET, COMETKiwi, and MetricX-24 Hybrid XL v2p6.
- Lexical diversity: corpus-level MTLD, MATTR-50, TTR, Yule's I/K, and a corpus-relative lexical-frequency profile.
- Morphology and syntax: Stanza-based morphological entropy and inventory measures, SynTTR/SFA, and ASTrED word-, sequence-, SACr-crossing, and tree-edit-distance measures.
- Uncertainty: paired bootstrap/randomization tests for sentence-level metrics and paired corpus bootstrap for MTLD and MATTR.

Corpus-level diversity metrics are computed after concatenating all outputs for one system, language, and split. They are not averaged over individual sentences.

The contrastive-LM reward keeps one necessary calibration step: the mean HT and MT margins on its development split define the center and scale of a logistic mapping to [0,1]. This mapping is part of the A3v3 and A3v4 reward definition, not a separate repair procedure. Runtime checks are limited to conditions that protect experimental validity, such as disjoint data splits, trainable LoRA parameters, complete GRPO groups, and reward weights that sum to one.

MetricX follows Google's official `metricx24` inference implementation and runs in a separate environment. Set `METRICX_PYTHON` and `METRICX_REPOSITORY` as described in `.env.example`.

## Released results

The `results/` directory contains the aggregate comparison matrices for the in-domain, FLORES+, news, and literary evaluations, together with the A2-versus-A5 significance tests reported in the appendix. No source, reference, generated sentence, prompt trace, model checkpoint, or copyrighted text is included.

## License and citation

The code is released under the MIT License. Corpus, model, and benchmark licenses remain with their respective owners. Citation metadata is provided in `CITATION.cff`.
