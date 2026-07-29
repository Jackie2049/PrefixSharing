"""Standalone CLI for synthetic prefix data generation and save.

Usage:
    # 1) Generate and save to JSON (for later USE_FIXED_ROLLOUT)
    python standalone_synthetic.py  \
        --json /path/to/any_rollout.json  \
        --batch-size 32  \
        --max-prompt 97908  \
        --max-response 2744  \
        --save /tmp/synthetic_prefix.json

    # 2) Just print stats (no save)
    python standalone_synthetic.py  \
        --json /path/to/any_rollout.json  \
        --batch-size 32  \
        --max-prompt 97908  \
        --max-response 2744
"""

import argparse
import json
import os
import sys

import torch


# --------------- core logic (same as inject_synthetic_prefix.py) ---------------

def _load_base_tokens(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    outputs = raw.get("outputs", raw)

    def _ensure_list(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return v
        return v

    for k in ("input_ids", "position_ids"):
        if k in outputs:
            outputs[k] = _ensure_list(outputs[k])

    ids = outputs.get("input_ids")
    if not ids:
        raise RuntimeError("[SyntheticPrefix] No 'input_ids' in JSON")
    pos = outputs.get("position_ids")
    if pos is None:
        raise RuntimeError("[SyntheticPrefix] No 'position_ids' in JSON")

    def _valid_len(p):
        return max(p) + 1 if p else 0

    best = max(range(len(ids)), key=lambda i: _valid_len(pos[i]))
    best_pos = pos[best]

    try:
        first_one = next(i for i, p in enumerate(best_pos) if p == 1)
    except StopIteration:
        raise RuntimeError("[SyntheticPrefix] Sample has no position_id=1")

    start = first_one - 1
    end = max(i for i, p in enumerate(best_pos) if p > 0) + 1
    return ids[best][start:end]


def _build_synthetic_batch(
    base_tokens,
    batch_size,
    max_prompt_length,
    max_response_length,
    pad_id=151643,
):
    n_seg = 2 * batch_size
    S = min(max_response_length, max_prompt_length // (n_seg - 1))

    total_needed = n_seg * S
    if len(base_tokens) < total_needed:
        raise RuntimeError(
            "[SyntheticPrefix] Need %d tokens, have %d. "
            "Reduce batch_size or increase max lengths." % (total_needed, len(base_tokens))
        )

    print("[SyntheticPrefix] bs=%d seg_size=%d total=%d" % (batch_size, S, total_needed))
    tokens = base_tokens[:total_needed]
    segs = [tokens[i * S:(i + 1) * S] for i in range(n_seg)]

    prompt_lists = []
    response_lists = []
    for i in range(batch_size):
        prompt_lists.append(sum(segs[:2 * i + 1], []))
        response_lists.append(segs[2 * i + 1])

    P, R, bs = max_prompt_length, max_response_length, batch_size

    prompts = torch.full((bs, P), pad_id, dtype=torch.long)
    for i in range(bs):
        pl = len(prompt_lists[i])
        prompts[i, -pl:] = torch.tensor(prompt_lists[i], dtype=torch.long)

    responses = torch.full((bs, R), pad_id, dtype=torch.long)
    for i in range(bs):
        rl = len(response_lists[i])
        responses[i, :rl] = torch.tensor(response_lists[i], dtype=torch.long)

    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = (input_ids != pad_id).long()

    position_ids = torch.zeros(bs, P + R, dtype=torch.long)
    for i in range(bs):
        pl = len(prompt_lists[i])
        rl = len(response_lists[i])
        if pl > 0:
            position_ids[i, P - pl:P] = torch.arange(pl)
        if rl > 0:
            position_ids[i, P:P + rl] = torch.arange(rl) + pl

    response_mask = torch.zeros(bs, R, dtype=torch.float32)
    for i in range(bs):
        response_mask[i, :len(response_lists[i])] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "token_level_rewards": response_mask.clone(),
        "rollout_log_probs": torch.zeros(bs, R, dtype=torch.float32),
        "rm_scores": torch.ones(bs, 1, dtype=torch.float32),
    }


# --------------- main ---------------

def main():
    parser = argparse.ArgumentParser(description="Standalone synthetic prefix data generator")
    parser.add_argument("--json", required=True, help="Path to JSON with input_ids + position_ids")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-prompt", type=int, required=True)
    parser.add_argument("--max-response", type=int, required=True)
    parser.add_argument("--save", default=None, help="Save to this JSON file (fixed_rollout format)")
    parser.add_argument("--num-workers", type=int, default=8, help="Pad to be divisible by this (default 8)")
    args = parser.parse_args()

    base_tokens = _load_base_tokens(args.json)
    print("Loaded %d valid tokens from best sample in %s" % (len(base_tokens), args.json))

    batch = _build_synthetic_batch(
        base_tokens,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt,
        max_response_length=args.max_response,
    )

    n = batch["input_ids"].shape[0]
    print("Generated batch: %d samples, input_ids shape=%s" % (n, list(batch["input_ids"].shape)))

    # Optional: pad to num_workers
    rem = n % args.num_workers
    if rem:
        pad_size = args.num_workers - rem
        for k, v in batch.items():
            if k == "rm_scores":
                batch[k] = torch.cat([v, v.new_zeros(pad_size, v.shape[1])], dim=0)
            elif v.ndim >= 2 and v.shape[0] == n:
                batch[k] = torch.cat([v, v.new_zeros(pad_size, *v.shape[1:])], dim=0)
        print("Padded %d -> %d (divisible by %d)" % (n, n + pad_size, args.num_workers))

    if args.save:
        save_dict = {
            "outputs": {k: v.tolist() for k, v in batch.items()},
        }
        save_dir = os.path.dirname(os.path.abspath(args.save)) or "."
        os.makedirs(save_dir, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(save_dict, f)
        print("Saved to %s" % args.save)
        print("Use with: USE_FIXED_ROLLOUT=%s python train_script.py" % args.save)
    else:
        print("No --save specified, data not persisted.")


if __name__ == "__main__":
    main()
