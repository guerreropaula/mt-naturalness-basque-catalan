#!/usr/bin/env python3
"""Check or apply the ASTrED 0.9.7 / awesome-align compatibility patch."""

from __future__ import annotations

import argparse
import sysconfig
from pathlib import Path

OLD = "        self.model = BertForMaskedLM.from_pretrained(\n            self.model_name_or_path, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, config=self.config\n        )\n"
NEW = "        self.model = BertForMaskedLM.from_pretrained(\n            self.model_name_or_path, config=self.config\n        )\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Apply the compatibility patch when needed."
    )
    args = parser.parse_args()

    site_packages = Path(sysconfig.get_paths()["purelib"])
    aligner_path = site_packages / "astred" / "aligner.py"
    if not aligner_path.exists():
        raise SystemExit(f"ASTrED aligner not found: {aligner_path}")

    text = aligner_path.read_text(encoding="utf-8")
    if NEW in text:
        print(f"ASTrED awesome-align patch already present: {aligner_path}")
        return
    if OLD not in text:
        raise SystemExit(
            "ASTrED aligner has an unexpected initialization block; do not patch automatically. "
            f"Inspect {aligner_path}."
        )
    if not args.apply:
        raise SystemExit(
            "ASTrED needs the awesome-align compatibility patch. Run: "
            "python -m src.evaluation.astred_compat --apply"
        )
    aligner_path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"Patched {aligner_path}")


if __name__ == "__main__":
    main()
