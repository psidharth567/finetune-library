#!/usr/bin/env python3
"""Build DPO preference pairs from `inference batch` outputs. Standard library only.

Two modes, both writing rows `finetune-lib` reads with data.format=preference
({"prompt": str | messages, "chosen": str, "rejected": str, ...}):

  reward  One samples file with --n > 1 per prompt. Every sample is scored with
          a reward file loaded by path (the same file grpo-library takes as
          reward.path); the best and worst sample of each prompt become the pair.

      python scripts/build_pairs.py reward --samples samples.jsonl \\
          --reward my_reward.py --gold-field gold --out pairs.jsonl

  delta   Two files over the same prompts, e.g. a strong and a weak model
          (the Olmo 3 "delta learning" heuristic): sample 0 of --chosen vs
          sample 0 of --rejected, joined on `_idx`.

      python scripts/build_pairs.py delta --chosen big.jsonl --rejected small.jsonl \\
          --out pairs.jsonl

Rows with an `error` or a finish_reason other than "stop" (truncated
generations) are never used. Reasoning returned in a separate `reasoning`
field is re-inserted as <think>...</think> before the response unless
--drop-reasoning is given, so thinking models are trained on what they emitted.
A JSON summary of kept and dropped counts is printed and written next to --out.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number} is not a JSON object")
                rows.append(row)
    return rows


def prompt_of(row: dict[str, Any]) -> str | list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        return messages
    prompt = row.get("prompt", row.get("prompts"))
    if not isinstance(prompt, str):
        raise ValueError(f"row _idx={row.get('_idx')} has no prompt/prompts/messages")
    system = row.get("system")
    if system:
        return [{"role": "system", "content": str(system)}, {"role": "user", "content": prompt}]
    return prompt


def usable(row: dict[str, Any]) -> bool:
    return (
        not row.get("error")
        and row.get("finish_reason", "stop") == "stop"
        and isinstance(row.get("responses"), str)
        and bool(row["responses"].strip())
    )


def response_of(row: dict[str, Any], *, drop_reasoning: bool) -> str:
    response = row["responses"]
    reasoning = row.get("reasoning")
    if reasoning and not drop_reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{response}"
    return response


def load_reward(path: Path, name: str) -> Callable[..., float]:
    spec = importlib.util.spec_from_file_location("dpo_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, name)
    parameters = inspect.signature(function).parameters
    if "solution_str" not in parameters:
        raise TypeError(
            f"{path}:{name} must take solution_str (grpo-library classic reward signature); "
            "transcript rewards are not supported here"
        )
    return function


def reward_pairs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    reward = load_reward(args.reward, args.reward_name)
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    counts = {"samples": 0, "unusable_samples": 0}
    for row in read_jsonl(args.samples):
        counts["samples"] += 1
        if not usable(row):
            counts["unusable_samples"] += 1
            continue
        groups[row["_idx"]].append(row)
    counts.update(prompts=len(groups), too_few_samples=0, gap_below_min=0)
    pairs = []
    for index in sorted(groups):
        rows = groups[index]
        if len(rows) < 2:
            counts["too_few_samples"] += 1
            continue
        scored = []
        for row in rows:
            response = response_of(row, drop_reasoning=args.drop_reasoning)
            # Score the visible answer; the reasoning is not graded.
            gold = row.get(args.gold_field)
            score = float(reward(solution_str=row["responses"], ground_truth=gold))
            scored.append((score, row.get("_sample", 0), response))
        # Deterministic: ties broken by sample index.
        scored.sort(key=lambda item: (-item[0], item[1]))
        best, worst = scored[0], scored[-1]
        if best[0] - worst[0] <= args.min_gap:
            counts["gap_below_min"] += 1
            continue
        pairs.append(
            {
                "prompt": prompt_of(rows[0]),
                "chosen": best[2],
                "rejected": worst[2],
                "chosen_score": best[0],
                "rejected_score": worst[0],
                "source_idx": index,
                "chosen_sample": best[1],
                "rejected_sample": worst[1],
            }
        )
    return pairs, counts


def delta_pairs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    def first_samples(path: Path) -> tuple[dict[Any, dict[str, Any]], int]:
        kept: dict[Any, dict[str, Any]] = {}
        unusable = 0
        for row in read_jsonl(path):
            if row.get("_sample", 0) != 0:
                continue
            if not usable(row):
                unusable += 1
                continue
            kept[row["_idx"]] = row
        return kept, unusable

    chosen, chosen_unusable = first_samples(args.chosen)
    rejected, rejected_unusable = first_samples(args.rejected)
    counts = {
        "chosen_rows": len(chosen),
        "rejected_rows": len(rejected),
        "chosen_unusable": chosen_unusable,
        "rejected_unusable": rejected_unusable,
        "unmatched": len(set(chosen) ^ set(rejected)),
        "prompt_mismatch": 0,
    }
    pairs = []
    for index in sorted(set(chosen) & set(rejected)):
        prompt = prompt_of(chosen[index])
        if prompt != prompt_of(rejected[index]):
            counts["prompt_mismatch"] += 1
            continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": response_of(chosen[index], drop_reasoning=args.drop_reasoning),
                "rejected": response_of(rejected[index], drop_reasoning=args.drop_reasoning),
                "source_idx": index,
                "chosen_model": chosen[index].get("model", str(args.chosen)),
                "rejected_model": rejected[index].get("model", str(args.rejected)),
            }
        )
    return pairs, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    reward = modes.add_parser("reward", help="best vs worst of N samples under a reward file")
    reward.add_argument("--samples", type=Path, required=True)
    reward.add_argument(
        "--reward", type=Path, required=True, help="python file exporting the reward"
    )
    reward.add_argument("--reward-name", default="compute_score")
    reward.add_argument("--gold-field", default="gold", help="row field passed as ground_truth")
    reward.add_argument(
        "--min-gap", type=float, default=0.0, help="keep pairs whose score gap exceeds this"
    )
    delta = modes.add_parser("delta", help="strong-model vs weak-model responses")
    delta.add_argument("--chosen", type=Path, required=True)
    delta.add_argument("--rejected", type=Path, required=True)
    for mode in (reward, delta):
        mode.add_argument("--out", type=Path, required=True)
        mode.add_argument("--drop-reasoning", action="store_true")
    args = parser.parse_args(argv)

    pairs, counts = reward_pairs(args) if args.mode == "reward" else delta_pairs(args)
    counts["pairs"] = len(pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    summary = {"mode": args.mode, **counts}
    # Written last: its presence means the pairs file is complete.
    args.out.with_suffix(args.out.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())
