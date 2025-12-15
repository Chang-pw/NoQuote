import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_token_level_log_ratios(
    left_text: str,
    right_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str = 'cuda',
    k: int = 3,
    agg_mode: str = 'topk_mean',
    penalty_param: list = None,
    mode: str = 'ch',
    length_penalty_alpha = 1.2,
    avg_penalty_alpha = 0.8,
):
    model.eval()

    left_ids  = tokenizer.encode(left_text,  return_tensors='pt')[0].tolist()
    right_ids = tokenizer.encode(right_text, return_tensors='pt')[0].tolist()[1:]

    logp_prior = [0.0] * len(right_ids)
    logp_cond  = [0.0] * len(right_ids)

    # Prior
    with torch.no_grad():
        prefix = []
        for t, tok in enumerate(right_ids):
            input_ids = torch.tensor(prefix + [tok], dtype=torch.long).unsqueeze(0).to(device)
            logits = model(input_ids).logits
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            logp_prior[t] = log_probs[tok].item()
            prefix.append(tok)

    # Conditional
    with torch.no_grad():
        context = left_ids.copy()
        for t, tok in enumerate(right_ids):
            input_ids = torch.tensor(context + [tok], dtype=torch.long).unsqueeze(0).to(device)
            logits = model(input_ids).logits
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            logp_cond[t] = log_probs[tok].item()
            context.append(tok)
    # Compute log-ratios
    token_infos = []
    if penalty_param is None:
        penalty_param = [1.0] * len(right_ids)
    elif len(penalty_param) < len(right_ids):
        # 用最后一个值填充（或者你也可以改成固定值，比如 penalty_param += [1.0] * diff）
        last_val = penalty_param[-1] if penalty_param else 1.0
        penalty_param += [last_val] * (len(right_ids) - len(penalty_param))
    elif len(penalty_param) > len(right_ids):
        # 如果太长，就截断
        penalty_param = penalty_param[:len(right_ids)]
    penalty_param[0] = 0  # 忽略首 token 的影响
    avg_penalty_param = sum(penalty_param) / len(penalty_param)

    for idx, tok_id in enumerate(right_ids):
        tok_str = tokenizer.convert_ids_to_tokens(tok_id, skip_special_tokens=True)
        log_ratio = logp_prior[idx] - logp_cond[idx]
        weighted_log_ratio = log_ratio * penalty_param[idx]

        token_infos.append({
            'token': tok_str,
            'idx': idx,
            'logp_prior': logp_prior[idx],
            'logp_cond': logp_cond[idx],
            'log_ratio': log_ratio,
            'penalty_weight': penalty_param[idx],
            'weighted_log_ratio': weighted_log_ratio
        })

    # 聚合
    sorted_by_log_ratio = sorted(token_infos, key=lambda x: -x['weighted_log_ratio'])
    topk = sorted_by_log_ratio[:k]

    if agg_mode == 'topk_mean':
        novelty = sum(d['weighted_log_ratio'] for d in topk) / k
    elif agg_mode == 'max':
        novelty = topk[0]['weighted_log_ratio']
    elif agg_mode == 'mean_all':
        novelty = sum(d['weighted_log_ratio'] for d in token_infos) / len(token_infos)
    elif agg_mode == "final":
        novelty = sum(d['weighted_log_ratio'] for d in token_infos) / len(token_infos)**length_penalty_alpha
        novelty *= (1- avg_penalty_param)** avg_penalty_alpha
    else:
        raise ValueError(f"Unknown agg_mode: {agg_mode}")


    return {
        'token_infos': token_infos,
        'topk': topk,
        'novelty_log_ratio': novelty
    }


