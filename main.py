import argparse
import json
import math
import os
import queue
import time
from multiprocessing import Manager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .code import *

def compute_token_level_log_ratios_cached(
    left_text: str,
    right_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str = "cuda",
    k: int = 3,
    agg_mode: str = "topk_mean",
    penalty_param=None,
    mode: str = "ch",
    length_penalty_alpha: float = 1.2,
    avg_penalty_alpha: float = 0.8,
):
    """
    KV-Cache for Computing Token-Level Log Ratios
    """
    model.eval()
    with torch.no_grad():
        left_ids = tokenizer.encode(left_text, return_tensors="pt").to(device)
        right_ids = (
            tokenizer.encode(right_text, return_tensors="pt").to(device)[0, 1:]
        )

        # 1) prior
        prior_logp = []
        prior_past = None
        for tok in right_ids:
            inp = tok.view(1, 1)
            out = model(inp, past_key_values=prior_past, use_cache=True)
            logits = out.logits[0, -1]
            prior_logp.append(torch.log_softmax(logits, dim=-1)[tok].item())
            prior_past = out.past_key_values

        # 2) conditional
        cond_logp = []
        left_out = model(left_ids, use_cache=True)
        cond_past = left_out.past_key_values

        for tok in right_ids:
            inp = tok.view(1, 1)
            out = model(inp, past_key_values=cond_past, use_cache=True)
            logits = out.logits[0, -1]
            cond_logp.append(torch.log_softmax(logits, dim=-1)[tok].item())
            cond_past = out.past_key_values

    # 3) aggregate
    logp_prior = prior_logp
    logp_cond = cond_logp

    if penalty_param is None:
        penalty_param = [1.0] * len(right_ids)
    elif len(penalty_param) < len(right_ids):
        last_val = penalty_param[-1] if penalty_param else 1.0
        penalty_param += [last_val] * (len(right_ids) - len(penalty_param))
    elif len(penalty_param) > len(right_ids):
        penalty_param = penalty_param[: len(right_ids)]

    if penalty_param:
        penalty_param[0] = 0  # ignore the first token
    avg_penalty_param = sum(penalty_param) / max(len(penalty_param), 1)

    token_infos = []
    for idx, (tok_id, lp, lc) in enumerate(zip(right_ids, logp_prior, logp_cond)):
        tok_str = tokenizer.convert_ids_to_tokens(tok_id.item(), skip_special_tokens=True)
        log_ratio = lp - lc
        weighted_log_ratio = log_ratio * penalty_param[idx]
        token_infos.append(
            {
                "token": tok_str,
                "idx": idx,
                "logp_prior": lp,
                "logp_cond": lc,
                "log_ratio": log_ratio,
                "penalty_weight": penalty_param[idx],
                "weighted_log_ratio": weighted_log_ratio,
            }
        )

    sorted_by_log_ratio = sorted(token_infos, key=lambda x: -x["weighted_log_ratio"])
    topk = sorted_by_log_ratio[:k]

    if agg_mode == "topk_mean":
        novelty = sum(d["weighted_log_ratio"] for d in topk) / k
    elif agg_mode == "max":
        novelty = topk[0]["weighted_log_ratio"]
    elif agg_mode == "mean_all":
        novelty = sum(d["weighted_log_ratio"] for d in token_infos) / len(token_infos)
    elif agg_mode == "final":
        novelty = sum(d["weighted_log_ratio"] for d in token_infos) / (
            len(token_infos) ** length_penalty_alpha
        )
        novelty *= (1 - avg_penalty_param) ** avg_penalty_alpha
    else:
        raise ValueError(f"Unknown agg_mode: {agg_mode}")

    return {
        "token_infos": token_infos,
        "topk": topk,
        "novelty_log_ratio": novelty,
    }

def hard_retrieval(quotes, label_list, threshold=0.6):
    core_domains, core_insights, _, _, _, _, _ = label_extract(label_list)
    insights = [quote["core_insights"] for quote in quotes]
    insights_bool_list = hard_label_filter(insights, core_insights, threshold=threshold)

    filtered_quotes = []
    aa = []
    for quote_item, keep in zip(quotes, insights_bool_list):
        if keep:
            filtered_quotes.append(quote_item)
        else:
            aa.append(quote_item)

    quotes = [i["quote"] for i in filtered_quotes]
    aa_quotes = [i["quote"] for i in aa]
    return filtered_quotes, quotes, aa, aa_quotes


def reasonable_retrieval(context, top_n, threshold, fixed_info=None, mode="ch"):
    fixed_info = fixed_info or {}
    if fixed_info:
        sm_context = fixed_info["summary_context"]
        dm_context = [fixed_info["dm_context"]]
        label_list = fixed_info["label_list"]
    else:
        sm_context, dm_context, label_list = context_summary(context)

    dm_filter_result, dm_filter_quotes = dm_retrieval(dm_context, top_n=top_n)
    hd_filter_result, hd_filter_quotes, aa_result, aa_quotes = hard_retrieval(
        dm_filter_result, label_list, threshold=threshold
    )
    return hd_filter_quotes, sm_context, dm_filter_quotes, dm_context[0], label_list


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


def reranking(quotes, n_alpha=1.0, p_alpha=1.0):
    for quote in quotes:
        novelty_score = quote["novelty"]
        popularity_score = quote["popularity"]
        final_score = round(novelty_score * n_alpha + popularity_score * p_alpha, 2)
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
    threshold=0.7,
):
    hd_quotes, summary_context, row_quotes, dm_context, label_list = reasonable_retrieval(
        context, threshold=threshold, top_n=top_n, mode=detect_language(context)
    )

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
    quotes = popularity_score(quotes, popularity_lst)
    final_quotes = reranking(quotes, n_alpha=n_alpha, p_alpha=p_alpha)

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

    # produce concise version
    concise = []
    for item in updated:
        new_item = dict(item)
        new_item.pop("result", None)
        concise.append(new_item)
    concise_path = os.path.splitext(args.output_path)[0] + "_concise.json"
    with open(concise_path, "w", encoding="utf-8") as f:
        json.dump(concise, f, ensure_ascii=False, indent=2)

    print(f"Completed, results saved in: {args.output_path} and {concise_path}")


if __name__ == "__main__":
    main()

