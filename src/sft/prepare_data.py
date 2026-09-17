"""Convert ordered SFT splits to chat-format JSONL."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.sft.data import build_chat_record, load_jsonl_records
from src.utils.config import load_sft_config
from src.utils.io import save_json, save_jsonl

logger = logging.getLogger(__name__)


def prepare_sft_chat_data(
    dataset_key: str,
    config_path: str | Path = "configs/sft.yaml",
    data_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, str]:
    config = load_sft_config(config_path)["sft"]
    if dataset_key not in config["target_languages"]:
        raise ValueError(f"Unsupported SFT dataset: {dataset_key}")
    root = Path(data_dir or config["data_dir"]) / dataset_key / "sft"
    chat_root = root / str(config["chat_format"]["directory"])
    target_lang = str(config["target_languages"][dataset_key])
    outputs: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "dev"):
        output = chat_root / f"{split}.jsonl"
        if output.exists() and not force:
            raise FileExistsError(f"Chat SFT output already exists: {output}")
        records = load_jsonl_records(root / f"{split}.jsonl")
        chats = [build_chat_record(record, target_lang) for record in records]
        save_jsonl(chats, output)
        outputs[split] = str(output)
        counts[split] = len(chats)
    metadata = {
        "dataset_key": dataset_key,
        "target_language": target_lang,
        "source": "ordered data/training/<dataset>/sft/{train,dev}.jsonl",
        "selection_order": "preserved exactly; no shuffle",
        "format": "OpenAI-style messages ending in the reference assistant response",
        "counts": counts,
        "outputs": outputs,
    }
    metadata_path = chat_root / "metadata.json"
    save_json(metadata, metadata_path)
    outputs["metadata"] = str(metadata_path)
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SFT parallel JSONL to chat-format JSONL.")
    parser.add_argument("--dataset", required=True, choices=("en_eu", "en_ca", "all"))
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    datasets = ("en_eu", "en_ca") if args.dataset == "all" else (args.dataset,)
    for dataset_key in datasets:
        paths = prepare_sft_chat_data(dataset_key, args.config, args.data_dir, args.force)
        logger.info("Chat SFT data ready for %s: %s", dataset_key, paths)


if __name__ == "__main__":
    main()
