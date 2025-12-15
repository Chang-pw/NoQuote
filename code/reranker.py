def bm25_reranker(candidates,context):
    from rank_bm25 import BM25Okapi
    from typing import List
    import jieba
    tokenized_corpus = [list(jieba.cut(doc)) for doc in candidates]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = list(jieba.cut(context))
    
    scores = bm25.get_scores(tokenized_query)
    
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted_indices
    
    return [candidates[i] for i in top_indices]

def flag_reranker(candidates, context):
    from FlagEmbedding import FlagReranker
    import os
    # 设置 GPU 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # 加载模型（建议在函数外部加载，避免重复加载）
    reranker_model = FlagReranker('bge-reranker-v2-m3', use_fp16=True)
    # 构造成对输入：每个候选和 context 组成一个 pair
    pairs = [[context, candidate] for candidate in candidates]
    
    # 计算得分（可选 normalize）
    scores = reranker_model.compute_score(pairs, normalize=True)
    
    # 对候选结果按得分排序
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    # 返回按得分排序后的候选项
    return [candidates[i] for i in sorted_indices]

def qwen3_reranker(candidates,context):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    import torch
    from modelscope import AutoModelForCausalLM, AutoTokenizer

    # 初始化模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen3-Reranker-0.6B", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("Qwen3-Reranker-0.6B").eval()
    # token id 映射
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192

    prefix = "<|im_start|>system\nJudge whether the Quotation is wow-worthy (reasonable and novel, genuinely unexpected yet fitting) with respect to the given Context. The answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    def qwen3_reranker(candidates, context, instruction=None):
        def format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = 'Judge whether the Quotation is wow-worthy (reasonable and novel, genuinely unexpected yet fitting) with respect to the given Context.'
            return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

        pairs = [format_instruction(instruction, context, doc) for doc in candidates]

        inputs = tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)

        with torch.no_grad():
            batch_scores = model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in sorted_indices]
    return qwen3_reranker(candidates,context)
    

def surprisal_reranker(candidates, context):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen3-8B").to(device)
    model.eval()

    def calc_surprisal(text):
        # 拼接上下文
        full_text = context + text
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
        # 找到上下文和候选的分界点
        start_idx = len(tokenizer(context)["input_ids"])
        ids = inputs["input_ids"][0]
        s = 0.0
        n = 0
        for i in range(start_idx, len(ids)):
            s -= log_probs[0, i-1, ids[i]].item()
            n += 1
        return s / max(1, n)

    scores = [calc_surprisal(c) for c in candidates]
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in sorted_indices]

def embedding_reranker(candidates, context):
    """
    使用句向量距离进行新颖性排序
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from utils import load_config
    config = load_config()
    EMD_MODEL = config['EMD_MODEL']
    model = SentenceTransformer(EMD_MODEL)
    embeddings = model.encode(candidates, normalize_embeddings=True)

    novelty_scores = []
    for i in range(len(candidates)):
        sims = cosine_similarity([embeddings[i]], embeddings)[0]
        sims[i] = -1  # 去掉自己
        max_sim = np.max(sims)
        novelty_scores.append(1 - max_sim)  # 相似度越低，新颖性越高

    sorted_indices = sorted(range(len(novelty_scores)), key=lambda i: novelty_scores[i], reverse=True)
    return [candidates[i] for i in sorted_indices]

def self_bleu_reranker(candidates, context):
    """
    使用 Self-BLEU 评估多样性
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import numpy as np

    smoothie = SmoothingFunction().method1
    scores = []
    for i in range(len(candidates)):
        refs = [candidates[j].split() for j in range(len(candidates)) if j != i]
        if len(refs) == 0:
            scores.append(0)
        else:
            bleu = sentence_bleu(refs, candidates[i].split(), smoothing_function=smoothie)
            scores.append(-bleu)  # 越小越好，所以取负数
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in sorted_indices]


def kl_reranker(candidates, context, model_name="gpt2"):
    """
    使用 KL 散度度量新颖性:
    KL(P_prior || P_cond)
    越大表示上下文改变预测分布越多，候选更意外
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn.functional as F
    import numpy as np
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen3-8B").to(device)
    model.eval()

    def calc_kl(candidate):
        # 先计算条件分布
        full_text = context + candidate
        inputs_cond = tokenizer(full_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_cond = model(**inputs_cond)
            log_probs_cond = F.log_softmax(outputs_cond.logits, dim=-1)

        # 再计算先验分布（不加context）
        inputs_prior = tokenizer(candidate, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_prior = model(**inputs_prior)
            log_probs_prior = F.log_softmax(outputs_prior.logits, dim=-1)

        # 计算 KL(P_prior || P_cond) 只对 candidate 部分 token
        ids_cand = inputs_prior["input_ids"][0]
        ids_full = inputs_cond["input_ids"][0]
        start_idx = len(tokenizer(context)["input_ids"])

        total_kl = 0.0
        count = 0
        for i in range(len(ids_cand)):
            # 对应 full_text 的 token index
            j = start_idx + i
            # 取先验分布和条件分布
            p_prior = log_probs_prior[0, i - 1] if i > 0 else log_probs_prior[0, 0]
            p_cond = log_probs_cond[0, j - 1] if j > 0 else log_probs_cond[0, 0]
            # KL = sum p_prior * (log p_prior - log p_cond)
            kl = torch.sum(torch.exp(p_prior) * (p_prior - p_cond)).item()
            total_kl += kl
            count += 1
        return total_kl / max(1, count)

    scores = [calc_kl(c) for c in candidates]
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in sorted_indices]


def mi_reranker(candidates, context, model_name="gpt2"):
    """
    使用语言模型近似计算互信息 MI = H(Q) - H(Q|C)
    分数越高表示候选与上下文关联性越强
    适合筛选“相关但不过度贴合”的引文
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    def avg_nll(logits, ids, start=0):
        """计算平均负对数概率"""
        log_probs = F.log_softmax(logits, dim=-1)
        nll = 0.0
        count = 0
        for i in range(start, len(ids)):
            prev_i = i - 1 if i > 0 else 0
            nll -= log_probs[0, prev_i, ids[i]].item()
            count += 1
        return nll / max(1, count)

    def calc_mi(candidate):
        # 1. prior (没有context)
        inputs_prior = tokenizer(candidate, return_tensors="pt").to(device)
        with torch.no_grad():
            out_prior = model(**inputs_prior)
        nll_prior = avg_nll(out_prior.logits, inputs_prior["input_ids"][0])

        # 2. conditional (有context)
        full_text = context + candidate
        inputs_cond = tokenizer(full_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out_cond = model(**inputs_cond)
        # context部分不计算熵
        start_idx = len(tokenizer(context)["input_ids"])
        nll_cond = avg_nll(out_cond.logits, inputs_cond["input_ids"][0], start=start_idx)

        return nll_prior - nll_cond  # MI = H(Q) - H(Q|C)

    scores = [calc_mi(c) for c in candidates]
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in sorted_indices]
