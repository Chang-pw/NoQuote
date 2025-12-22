import argparse
import json
import math
import os
import queue
import time
from multiprocessing import Manager

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from code import *
except Exception:
    from .code import *


def novelty_score(
    quotes,
    context,
    tokenizer,
    model,
    penalty_params_lst,
    mode_lst,
    device,
    agg_mode="final",
):
    left = context + "正如那句经典所言："
    result = []
    for quote in quotes:
        right = quote
        penalty_param = penalty_params_lst[right]
        res = compute_token_level_log_ratios_cached(
            left,
            right,
            tokenizer,
            model,
            device=device,
            agg_mode=agg_mode,
            penalty_param=penalty_param,
            mode=mode_lst[quote],
        )
        result.append({"quote": right, "novelty": round(res["novelty_log_ratio"], 2)})
    return result


def popularity_score(quotes, popularity_lst):
    for quote in quotes:
        popularity_score = popularity_lst[quote["quote"]]
        quote["popularity"] = round(popularity_score, 2)
    return quotes

def reranking(quotes, n_alpha=1.0, p_alpha=1.0, r_alpha=1.0):
    for quote in quotes:
        novelty_score = quote["novelty"]
        popularity_score = quote["popularity"]
        relevance_score = quote["relevance"] 

        final_score = round(
            novelty_score * n_alpha
            + popularity_score * p_alpha
            + relevance_score * r_alpha,
            2,
        )
        quote["final_score"] = final_score

    return sorted(quotes, key=lambda x: x["final_score"], reverse=True)

def run_pipeline(
    context,
    tokenizer,
    model,
    penalty_params_lst,
    mode_lst,
    popularity_lst,
    top_n=50,
    agg_mode="final",
    n_alpha=1.0,
    p_alpha=1.0,
    r_alpha=1.0,
    threshold=0.7,
):
    hd_quotes, summary_context, row_quotes, dm_context, label_list = reasonable_retrieval(
        context, threshold=threshold, top_n=top_n, mode=detect_language(context)
    )

    relevance_dict = compute_relevance_dict(summary_context, hd_quotes)

    quotes = novelty_score(
        hd_quotes,
        summary_context,
        tokenizer,
        model,
        penalty_params_lst,
        mode_lst,
        device=model.device,
        agg_mode=agg_mode,
    )

    for q in quotes:
        q["relevance"] = round(relevance_dict.get(q["quote"], 0.0), 2)

    quotes = popularity_score(quotes, popularity_lst)
    final_quotes = reranking(quotes, n_alpha=n_alpha, p_alpha=p_alpha, r_alpha=r_alpha)

    return {
        "context": context,
        "summary_context": summary_context,
        "dm_context": dm_context,
        "label_list": label_list,
        "final_quote": final_quotes[0]["quote"],
        "final_quotes": final_quotes,
        "row_quotes": row_quotes,
        "hard_quotes": hd_quotes,
    }

def worker(
    rank,
    world_size,
    contexts,
    args,
    shared_queue,
    penalty_params_lst,
    mode_lst,
    popularity_lst,
):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    config = load_config()
    model_path = config["BASE_MODEL"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map={"": device}
    )

    for idx in range(rank, len(contexts), world_size):
        item = contexts[idx]
        context = item[args.context_key]
        try:
            result = run_pipeline(
                context,
                tokenizer,
                model,
                penalty_params_lst,
                mode_lst,
                popularity_lst,
                top_n=args.top_n,
                agg_mode=args.agg_mode,
                n_alpha=args.n_alpha,
                p_alpha=args.p_alpha,
                r_alpha=args.r_alpha,
                threshold=args.threshold,
            )
            item["quote"] = result["final_quote"]
            item["result"] = result
        except Exception as exc:
            item["error"] = str(exc)
        shared_queue.put((idx, item))


def load_quotes_metadata(quotes_path):
    with open(quotes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    penalty_params_lst = {i["quote"]: i["penalty_weights"] for i in data}
    mode_lst = {i["quote"]: i["mode"] for i in data}
    popularity_lst = {i["quote"]: i["Search_score"] for i in data}
    return penalty_params_lst, mode_lst, popularity_lst


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Input JSON file path")
    parser.add_argument("--output_path", required=True, help="Output JSON file path")
    parser.add_argument("--context_key", default="context", help="Context key in the input JSON file")
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--agg_mode", default="final")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--n_alpha", type=float, default=1.0)
    parser.add_argument("--p_alpha", type=float, default=0.0)
    parser.add_argument("--r_alpha", type=float, default=1.0)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument(
        "--quotes_path",
        default="./data/quote/quotes_zh_label.json",
        help="Quotes JSON file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)

    world_size = min(args.world_size, torch.cuda.device_count())
    if world_size < 1:
        raise RuntimeError("No available GPU detected.")

    penalty_params_lst, mode_lst, popularity_lst = load_quotes_metadata(
        args.quotes_path
    )

    manager = Manager()
    shared_queue = manager.Queue()

    procs = []
    for rank in range(world_size):
        p = torch.multiprocessing.Process(
            target=worker,
            args=(
                rank,
                world_size,
                contexts,
                args,
                shared_queue,
                penalty_params_lst,
                mode_lst,
                popularity_lst,
            ),
        )
        p.start()
        procs.append(p)

    updated = [None] * len(contexts)
    finished = 0
    while finished < len(contexts):
        idx, item = shared_queue.get()
        updated[idx] = item
        finished += 1

    for p in procs:
        p.join()

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)


    print(f"Completed, results saved in: {args.output_path}")


if __name__ == "__main__":
    main()

