# Addressing Naturalness in LLM Translation for Low-Resource Languages

This repository contains the code and experimental results associated with the Master's Thesis **Addressing Naturalness in LLM Translation for Low-Resource Languages**.

The thesis investigates naturalness in Large Language Model (LLM) translation from English into **Basque (EN→EU)** and **Catalan (EN→CA)**. Although current LLMs achieve strong machine translation performance, their output may still exhibit characteristics of *machine translationese*, including reduced lexical diversity, morphological narrowing, and excessive structural similarity to the source language. The study examines whether these patterns can be reduced through interventions of increasing complexity (**prompting**, **supervised fine-tuning**, and **reinforcement learning with GRPO**) without systematically degrading translation quality.

This thesis was carried out within the Master's Degree in **Language Analysis and Processing (HAP/LAP)** and supervised by **Dr. Nora Aranberri** (University of the Basque Country, UPV/EHU) and **Dr. Antonio Toral** (University of Alicante).

## Research Questions

1. **RQ1:** What level of naturalness do the open-weight LLMs evaluated in this study exhibit relative to the human-reference profile?
2. **RQ2:** To what extent can prompting improve naturalness without a systematic loss of translation quality?
3. **RQ3:** Do post-training methods (SFT and RL) yield greater and more consistent improvements in naturalness than prompting, and at what cost to translation quality?

Naturalness is treated as a **multidimensional property** rather than a single score, covering lexical diversity, lexical frequency, morphological richness, synonym choice, and syntactic structure.

## Experimental Design

| Condition | Method | Description |
|---|---|---|
| **P0** | Direct translation | Minimal zero-shot translation baseline |
| **P1** | Naturalness-aware prompting | Explicit instruction to produce natural target-language text |
| **P2** | Self-polishing | Direct translation followed by a revision step |
| **P3** | Step-by-step prompting | Research, drafting, refinement, and proofreading pipeline |
| **P4** | Supervised fine-tuning | QLoRA adaptation on parallel translation data |
| **P5** | GRPO | Reinforcement learning starting from the P4 adapter |

P0 through P3 modify only the inference procedure. P4 and P5 modify model parameters. P5 includes several reward ablations: the main configuration (**A2**) combines chrF++, COMETKiwi, within-group Self-BLEU diversity, and a length penalty. Additional experiments test alternative quality metrics and learned naturalness estimators (discriminative classifiers and contrastive language models).

Prompts and training configurations are available under `configs/`.

## Models

Eight open-weight LLMs are evaluated under P0:

| Model | Size | Role |
|---|---:|---|
| [Latxa Llama 3.1 Instruct](https://huggingface.co/HiTZ/Latxa-Llama-3.1-8B-Instruct) | 8B | Basque-specialized; used throughout P0–P5 |
| [Latxa Llama 3.1 Instruct](https://huggingface.co/HiTZ/Latxa-Llama-3.1-70B-Instruct) | 70B | Larger Latxa model; used for P0–P3 |
| [SalamandraTA Instruct](https://huggingface.co/BSC-LT/salamandraTA-7b-instruct) | 7B | Translation-specialized; used for P0, P4, and P5 |
| [Salamandra Instruct](https://huggingface.co/BSC-LT/salamandra-7b-instruct) | 7B | General multilingual baseline |
| [Gemma 3 IT](https://huggingface.co/google/gemma-3-12b-it) | 12B | General multilingual baseline |
| [Gemma 3 IT](https://huggingface.co/google/gemma-3-27b-it) | 27B | Retained for prompting experiments |
| [Qwen3](https://huggingface.co/Qwen/Qwen3-8B) | 8B | General multilingual baseline |
| [Qwen3](https://huggingface.co/Qwen/Qwen3-32B) | 32B | General multilingual baseline |

After P0 baseline analysis, **Latxa 8B, Latxa 70B, and Gemma 3 27B** continue to the prompting experiments, while **Latxa 8B and SalamandraTA 7B** continue to SFT and GRPO. Model identifiers and inference settings are defined in `configs/models.yaml`.

## Data

### In-Domain Corpora

- **English→Basque:** [EHU-HAC](https://www.ehu.eus/ehg/hac/), a multilingual parallel corpus constructed from published books.
- **English→Catalan:** [AINA CA-EN Parallel Corpus](https://huggingface.co/datasets/projecte-aina/CA-EN_Parallel_Corpus), using material from the `HRM` and `CUL` domains.

### Out-of-Domain Evaluation

Selected systems are also evaluated on three external test sets:

- **FLORES+** (`devtest` partition) for general-domain translation.
- **News**, using English-Basque articles from [Berria](https://www.berria.eus/) and English-Catalan material from MaCoCu-ca-en.
- **Literary translation**, using Virginia Woolf's *To the Lighthouse* and its published Basque ([*Farorantz*](https://armiarma.eus/liburu-e/), translated by Anton Garikano) and Catalan ([*Cap al far*](https://pocketbook.es/cap-al-far-9788498594799), translated by Xavier Pàmies) translations.

Copyrighted literary text is not included in this repository.

## Evaluation

The evaluation framework separates **translation quality** from **naturalness**.

### Translation Quality
BLEU, chrF++, TER, COMET, XCOMET, COMETKiwi, and MetricX-24.

### Naturalness
- **Lexical diversity:** MTLD, MATTR-50, TTR, Yule's I/K
- **Lexical frequency:** Lexical Frequency Profile (LFP)
- **Morphological richness:** Shannon entropy, inverse Simpson diversity
- **Synonym variation:** SynTTR, PTF, CDU (using Apertium bilingual dictionaries for [Catalan-English](https://github.com/apertium/apertium-eng-cat) and [Basque-English](https://github.com/apertium/apertium-eu-en))
- **Syntactic structure:** ASTrED tree-edit distance, word crossing, sequence crossing, SACr crossing

## Repository Structure

```text
configs/                   Experiment, dataset, model, SFT, GRPO, and naturalness estimators settings
results/                   In-domain, out-of-domain, and statistical results
src/data/                  Corpus preprocessing and split construction
src/prompting/             P0–P3 prompting experiments
src/sft/                   P4 data preparation, training, and evaluation
src/grpo/                  P5 rewards, training, ablations, and evaluation
src/classifiers/           Binary HT-vs-MT naturalness classifiers
src/contrastive_lm/        Contrastive language-model naturalness estimators
src/evaluation/run.py      Translation quality and naturalness evaluation
src/evaluation/metrics/    Automatic, lexical, morphological, and syntactic metrics
src/evaluation/statistics/ Paired significance tests and corpus-level bootstrap
src/evaluation/compare.py  Aggregate P0–P5 comparison tables
src/utils/                 Shared configuration  and model-loading utilities
```

## Installation

Requires **Python 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,inference,training,evaluation,dev]'
```

For GRPO training:

```bash
python -m pip install -e '.[grpo]'
```

ASTrED has older dependencies and should be installed in a separate environment:

```bash
python -m pip install -e '.[astred]'
python -m src.evaluation.astred_compat --apply
```

MetricX follows Google's official `metricx24` implementation and runs in a separate environment. Set `METRICX_PYTHON` and `METRICX_REPOSITORY` as described in `.env.example`.

Some Hugging Face models require authentication:

```bash
export HF_TOKEN=your_token
```

## Running the Experiments

### Data Preprocessing

```bash
python -m src.data.preprocessing --dataset en_eu --force
python -m src.data.preprocessing --dataset en_ca --force
python -m src.data.build_training_splits --dataset all --force
```

### Prompting (P0–P3)

```bash
python -m src.prompting.p0.run --dataset en_eu --model latxa_8b_instruct --split test
python -m src.prompting.p1.run --dataset en_eu --model latxa_8b_instruct --split test
python -m src.prompting.p2.run --dataset en_eu --model latxa_8b_instruct --split test
python -m src.prompting.p3.run --dataset en_eu --model latxa_8b_instruct --split test
```

P0–P3 evaluation is deterministic (`do_sample: false`, single beam).

### Supervised Fine-Tuning (P4)

```bash
python -m src.sft.prepare_data --dataset en_eu --force
python -m src.sft.train --dataset en_eu --model latxa_8b_instruct
python -m src.sft.evaluate --dataset en_eu --model latxa_8b_instruct --split test
```

### GRPO (P5)

```bash
python -m src.grpo.train --dataset en_eu --model latxa_8b_instruct --ablation a2
python -m src.grpo.evaluate --dataset en_eu --model latxa_8b_instruct --ablation a2 --split test
```

Other reward configurations can be selected with `--ablation`.

### Evaluation and Statistical Analysis

```bash
python -m src.evaluation.run \
    --predictions path/to/predictions.jsonl \
    --output-dir path/to/evaluation

python -m src.evaluation.compare \
    --results-root results \
    --datasets en_eu en_ca

python -m src.evaluation.statistics.paired \
    --dataset en_eu --model latxa_8b_instruct \
    --baseline P5_GRPO_A2 --candidates P5_GRPO_A5

python -m src.evaluation.statistics.corpus \
    --dataset en_eu --model latxa_8b_instruct \
    --baseline P5_GRPO_A2 --candidates P5_GRPO_A5 \
    --iterations 10000
```

## Released Results

The `results/` directory contains aggregate comparison matrices for the in-domain, FLORES+, news, and literary evaluations, together with the A2-versus-A5 significance tests.

## Author

**Paula Guerrero Castelló**
Master's Degree in Language Analysis and Processing (HAP/LAP)

Supervised by:
- **Dr. Nora Aranberri**, University of the Basque Country (UPV/EHU)
- **Dr. Antonio Toral**, University of Alicante

## License and Citation

The code is released under the MIT License. Models, corpora, benchmarks, and other external resources remain subject to their respective licenses.