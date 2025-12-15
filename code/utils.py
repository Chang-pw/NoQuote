import ast
import json
import yaml
import numpy as np

## Def
def convert_str_to_list(input_str):
    list_obj = ast.literal_eval(input_str)
    json_str = list_obj[0].strip()
    data = json.loads(json_str)
    return data

def load_yaml(yaml_output):
    return yaml.safe_load(yaml_output)

def label_extract(label_list):
    core_domains = label_list['core_domains']
    core_insights = label_list['core_insights']
    applicability = label_list['applicability']
    core_values = label_list['core_values']
    metaphors = label_list['metaphors']
    style = label_list['style']
    sentiment_tone = label_list['sentiment_tone']
    return core_domains,core_insights,applicability,core_values,metaphors,style,sentiment_tone

def load_config():
    config_path = "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def hit_rate(gt_set, pred_list, k):
    return int(any(item in gt_set for item in pred_list[:k]))


def ndcg(gt_set, pred_list, k):
    for i, item in enumerate(pred_list[:k]):
        if item in gt_set:
            return 1 / np.log2(i + 2) 
    return 0.0

def mrr(gt_set, pred_list, k):
    for i, item in enumerate(pred_list[:k]):
        if item in gt_set:
            return 1 / (i + 1)
    return 0.0

def detect_language(text: str) -> str:
    chinese_count = 0
    english_count = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
        elif char.isalpha():
            english_count += 1

    return 'ch' if chinese_count > english_count else 'en'

