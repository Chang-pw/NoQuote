import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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