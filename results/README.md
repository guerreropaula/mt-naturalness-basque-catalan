# Released Results

This directory contains the aggregate comparison and significance outputs reported in the thesis.

- `in-domain/` contains the EN-EU and EN-CA P0-P5 comparison matrices for the main test set.
- `out-of-domain/flores/` contains the FLORES+ comparison matrices.
- `out-of-domain/news/` contains the Berria EN-EU and MaCoCu EN-CA comparison matrices.
- `out-of-domain/literary/` contains the full sentence-level Woolf comparison matrices and the corrected boundary-aware five-sentence diagnostic.
- `significance/` contains the A2-versus-A5 tests reported in the appendix. `paired_significance_a2_vs_a5_all` stores sentence-level paired tests, while `paired_corpus_bootstrap_a2_vs_a5` stores corpus-level MTLD and MATTR-50 intervals.

JSON and TSV versions of each significance analysis contain the same results.

No source text, human reference, model output, prompt trace, checkpoint, or copyrighted material is included.
