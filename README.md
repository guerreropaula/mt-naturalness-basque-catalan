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
src/data/                Loading, cleaning, splitting, chunking, and test preparation
src/prompting/           P0-P3 prompting experiments
src/sft/                 P4 data preparation, training, and deterministic evaluation
src/grpo/                P5 rewards, training, health checks, and evaluation
src/classifiers/         HT-versus-MT discriminative naturalness estimators
src/contrastive_lm/      HT-versus-MT contrastive causal-LM estimators
src/evaluation/          MT quality, lexical, morphological, syntactic, and significance analyses
```

Plotting code is not included. The repository focuses on the steps required to construct data, train systems, generate translations, evaluate them, and test paired differences.

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
python -m src.evaluation.patch_astred_awesome_align --apply
```

Some models are gated. Put a Hugging Face token in the environment, never in the repository:

```bash
export HF_TOKEN=your_token
```

## Data access

The thesis corpora are not redistributed. Configure authorized local corpus paths in `configs/datasets.yaml` before running preprocessing. Input records must provide the source and target fields mapped in that configuration; Catalan records must also expose the domain metadata used to retain HRM and CUL.

## Quick reproduction

After obtaining the corpora and the FastText language-identification model, configure their local paths and preprocess both directions:

```bash
python -m src.data.preprocessing --dataset en_eu --force
python -m src.data.preprocessing --dataset en_ca --force
python -m src.data.build_training_splits --dataset all --force
```

Run a prompting condition by selecting a dataset and model key. The example below uses the in-domain processed split:

```bash
python -m src.prompting.p0.run --dataset en_eu --model latxa_8b_instruct --split dev --results-dir results/p0_p3/p0
python -m src.prompting.p1.run --dataset en_eu --model latxa_8b_instruct --split dev --results-dir results/p0_p3/p1
python -m src.prompting.p2.run --dataset en_eu --model latxa_8b_instruct --split dev --results-dir results/p0_p3/p2
python -m src.prompting.p3.run --dataset en_eu --model latxa_8b_instruct --split dev --results-dir results/p0_p3/p3
```

Prepare, train, and evaluate P4:

```bash
python -m src.sft.prepare_data --dataset en_eu --force
python -m src.sft.train --dataset en_eu --model latxa_8b_instruct
python -m src.sft.evaluate --dataset en_eu --model latxa_8b_instruct --split global_dev
```

Train and evaluate P5 A2 from the P4 adapter:

```bash
python -m src.grpo.train --dataset en_eu --model latxa_8b_instruct --ablation a2
python -m src.grpo.evaluate --dataset en_eu --model latxa_8b_instruct --ablation a2 --split global_dev
```

P0-P4 evaluation is deterministic (`do_sample: false`, one beam). GRPO training samples eight completions per source at temperature 0.6 because group-relative rewards and Self-BLEU require variation within each group.


## Evaluation

The evaluation pipeline combines complementary evidence:

- Translation quality: BLEU, chrF++, TER, COMET, XCOMET, COMETKiwi, and MetricX-24 Hybrid XL v2p6.
- Lexical diversity: corpus-level MTLD, MATTR-50, TTR, Yule's I/K, and a corpus-relative lexical-frequency profile.
- Morphology and syntax: Stanza-based morphological entropy and inventory measures, SynTTR/SFA, and ASTrED word-, sequence-, SACr-crossing, and tree-edit-distance measures.
- Uncertainty: paired bootstrap/randomization tests for sentence-level metrics and paired corpus bootstrap for MTLD and MATTR.

Corpus-level diversity metrics are computed after concatenating all outputs for one system, language, and split. They are not averaged over individual sentences.

MetricX follows Google's official `metricx24` inference implementation and runs in a separate environment. Set `METRICX_PYTHON` and `METRICX_REPOSITORY` as described in `.env.example`.

## Released results

The `results/` directory contains the aggregate comparison matrices for the in-domain, FLORES+, news, and literary evaluations, together with the A2-versus-A5 significance tests reported in the appendix. No source, reference, generated sentence, prompt trace, model checkpoint, or copyrighted text is included.

## License and citation

The code is released under the MIT License. Corpus, model, and benchmark licenses remain with their respective owners. Citation metadata is provided in `CITATION.cff`.
