#!/usr/bin/env python3
"""Build turn-level training files from the PersonaConflicts corpus.

The output is JSONL because it is easy to inspect and can be loaded directly
with Hugging Face Datasets.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VC_CANONICAL = {
    "Comparison",
    "Demand",
    "Denial of Responsibility",
    "Deserve Thinking",
    "Moralistic Judgment",
}

NVC_CANONICAL = {
    "Empathic/Understanding",
    "Feeling Statement",
    "Need Statement",
    "Neutral Observation",
    "Request (No-Pressure Ask)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/persona-conflicts-corpus-emnlp-2025/mturk_aggregate.csv",
        help="Path to mturk_aggregate.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/persona_conflicts",
        help="Directory for processed JSONL files.",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=4,
        help="Number of previous turns to include as context.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/validation/test split.",
    )
    parser.add_argument(
        "--relationship-subtype",
        default=None,
        help="Optional filter, for example: couple.",
    )
    return parser.parse_args()


def parse_list(value: str, fallback: list[Any] | None = None) -> list[Any]:
    if not value:
        return [] if fallback is None else fallback
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [] if fallback is None else fallback


def role_for_speaker(speaker: str, mapping: dict[str, str]) -> str:
    clean = speaker.strip()
    if clean not in mapping:
        mapping[clean] = f"speaker_{chr(ord('a') + len(mapping))}"
    return mapping[clean]


def risk_class(score: float) -> str:
    if score >= 3.0:
        return "high"
    if score >= 2.0:
        return "medium"
    return "low"


def format_turn(turn: dict[str, Any], speaker_mode: str, mapping: dict[str, str]) -> str:
    text = " ".join(str(turn.get("text", "")).split())
    if speaker_mode == "none":
        return text
    if speaker_mode == "role":
        return f"{role_for_speaker(str(turn.get('speaker', '')), mapping)}: {text}"
    if speaker_mode == "name":
        speaker = str(turn.get("speaker", "")).strip()
        return f"{speaker}: {text}"
    raise ValueError(f"Unknown speaker mode: {speaker_mode}")


def stable_dialogue_id(row: dict[str, str]) -> str:
    raw_id = row.get("HITId") or row.get("") or row.get("id") or row.get("conversation", "")
    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:10]
    return f"dialogue_{digest}"


def build_examples(row: dict[str, str], history_turns: int, speaker_mode: str) -> list[dict[str, Any]]:
    conversation = json.loads(row["transformed_conversation"])
    problematic_scores = parse_list(row.get("turn_problematic_avg", ""))
    feeling_scores = parse_list(row.get("turn_feeling_avg", ""))
    vc_labels = parse_list(row.get("turn_vc_union", ""))
    nvc_labels = parse_list(row.get("turn_nvc_union", ""))

    dialogue_id = stable_dialogue_id(row)
    speaker_mapping: dict[str, str] = {}
    formatted_turns = [format_turn(turn, speaker_mode, speaker_mapping) for turn in conversation]
    examples: list[dict[str, Any]] = []

    for index, turn in enumerate(conversation):
        if index >= len(problematic_scores):
            continue

        context_start = max(0, index - history_turns)
        history = formatted_turns[context_start:index]
        current = formatted_turns[index]
        score = float(problematic_scores[index])
        feeling_score = float(feeling_scores[index]) if index < len(feeling_scores) else None
        raw_vc = vc_labels[index] if index < len(vc_labels) else []
        raw_nvc = nvc_labels[index] if index < len(nvc_labels) else []
        canonical_vc = [label for label in raw_vc if label in VC_CANONICAL]
        canonical_nvc = [label for label in raw_nvc if label in NVC_CANONICAL]

        examples.append(
            {
                "example_id": f"{dialogue_id}_turn_{index + 1:02d}",
                "dialogue_id": dialogue_id,
                "turn_index": index + 1,
                "speaker_mode": speaker_mode,
                "input_text": "\n".join(
                    [
                        f"relationship: {row.get('relationship_subtype', '')}",
                        f"scenario: {row.get('rewritten_scenario', '')}",
                        "history:",
                        *history,
                        "current_message:",
                        current,
                    ]
                ).strip(),
                "current_message": " ".join(str(turn.get("text", "")).split()),
                "history": history,
                "risk_score": score,
                "risk_score_0_1": round((score - 1.0) / 3.0, 4),
                "risk_class": risk_class(score),
                "feeling_score": feeling_score,
                "vc_labels": canonical_vc,
                "nvc_labels": canonical_nvc,
                "raw_vc_labels": raw_vc,
                "raw_nvc_labels": raw_nvc,
                "is_conflict_dialogue": row.get("convo_type") == "conflict",
                "backstory_condition": row.get("condition"),
                "relationship_subtype": row.get("relationship_subtype"),
                "high_level_category": row.get("High Level Category"),
                "subcategory": row.get("Subcategory"),
            }
        )

    return examples


def split_dialogues(rows: list[dict[str, str]], seed: int) -> dict[str, set[str]]:
    by_key: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = "|".join(
            [
                row.get("convo_type", ""),
                row.get("condition", ""),
                row.get("relationship_subtype", ""),
            ]
        )
        by_key[key].append(stable_dialogue_id(row))

    rng = random.Random(seed)
    splits = {"train": set(), "val": set(), "test": set()}
    for ids in by_key.values():
        ids = sorted(set(ids))
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        n_val = max(1, round(n * 0.1)) if n >= 3 else 0
        splits["test"].update(ids[:n_test])
        splits["val"].update(ids[n_test : n_test + n_val])
        splits["train"].update(ids[n_test + n_val :])
    return splits


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def summarize(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "turn_examples": len(examples),
        "dialogues": len({example["dialogue_id"] for example in examples}),
        "risk_class_counts": Counter(example["risk_class"] for example in examples),
        "relationship_counts": Counter(example["relationship_subtype"] for example in examples),
        "conflict_dialogue_counts": Counter(str(example["is_conflict_dialogue"]) for example in examples),
        "backstory_condition_counts": Counter(example["backstory_condition"] for example in examples),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_dir)

    with input_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.relationship_subtype:
        rows = [row for row in rows if row.get("relationship_subtype") == args.relationship_subtype]

    splits = split_dialogues(rows, args.seed)
    full_summary: dict[str, Any] = {
        "source": str(input_path),
        "relationship_subtype_filter": args.relationship_subtype,
        "history_turns": args.history_turns,
        "seed": args.seed,
        "variants": {},
    }

    for speaker_mode, variant_dir in [("none", "no_speakers"), ("role", "role_speakers")]:
        examples: list[dict[str, Any]] = []
        for row in rows:
            examples.extend(build_examples(row, args.history_turns, speaker_mode))

        by_split = {
            split_name: [
                example for example in examples if example["dialogue_id"] in dialogue_ids
            ]
            for split_name, dialogue_ids in splits.items()
        }

        for split_name, split_examples in by_split.items():
            write_jsonl(output_root / variant_dir / f"{split_name}.jsonl", split_examples)

        full_summary["variants"][variant_dir] = {
            "all": summarize(examples),
            "splits": {
                split_name: summarize(split_examples)
                for split_name, split_examples in by_split.items()
            },
        }

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(full_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(full_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
