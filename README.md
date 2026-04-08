<p align="center">
  <img src="https://img.shields.io/badge/ACL-2026-blue" alt="ACL 2026">
  <a href="https://arxiv.org/abs/2602.22220"><img src="https://img.shields.io/badge/arXiv-2602.22220-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

# What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty

English | [中文](README_zh.md)

Our work is accepted at **ACL 2026 Main Conference**.

> **TL;DR:** We propose NovelQR for quotation recommendation to find quotes that are unexpected yet rational, combining deep-meaning retrieval with token-level novelty reranking to enhance aesthetic value in writing.

## Overview

Quotation recommendation aims to enrich writing by suggesting quotes that complement a given context, yet existing systems mostly optimize surface-level topical relevance and ignore the deeper semantic and aesthetic properties that make quotations memorable. We start from two empirical observations: (1) people consistently prefer quotations that are "unexpected yet rational" in context, and (2) strong existing models struggle to fully understand the deep meanings of quotations.

We operationalize this objective with **NovelQR**, a novelty-driven quotation recommendation framework:

- **Label Enhancement**: A generative label agent interprets each quotation and its surrounding context into multi-dimensional deep-meaning labels, enabling label-enhanced retrieval.
- **Rationality Retrieval**: Retrieves candidate quotations that are semantically relevant and filters them by deep meaning and labels.
- **Novelty Reranking**: A token-level novelty estimator reranks candidates while mitigating auto-regressive continuation bias.

## Method

![method](figure/method.png)

## Project Structure

```
NoQuote/
├── main.py                    # Main entry point for retrieval pipeline
├── config.json                # Configuration (API key, model paths)
├── requirements.txt           # Python dependencies
├── run_GPUS.sh                # Multi-GPU launch script
├── code/
│   ├── label_agent.py         # Generative Label Agent
│   ├── retrieval.py           # Retrieval pipeline
│   ├── logit_token_level.py   # Token-level novelty estimator
│   ├── relevance_score.py     # Relevance scoring
│   ├── context_summary.py     # Context summarization
│   ├── model.py               # Model utilities
│   ├── utils.py               # General utilities
│   └── rag/                   # RAG modules
│       ├── rag_module.py
│       ├── rag_retrieval.py
│       ├── rag_add.py
│       └── hard_filter.py
├── data/
│   ├── prompt/                # Prompt templates
│   └── quote/                 # Quotation data
└── figure/
    └── method.png
```

## Quick Start

### Installation

```bash
conda create -n NovelQuote python=3.10
conda activate NovelQuote
pip install -r requirements.txt
```

### Download the Quote Data

The original quotation data comes from [QUILL](https://github.com/GraceXiaoo/QUILL). The label-enhanced data processed by our Label Agent is available on [Hugging Face](https://huggingface.co/datasets/Changpw/Noquote). Put the data in the `data/quote/` folder.

### Configuration

Set the config in `config.json`:

| Parameter | Description |
|---|---|
| `API_KEY` | API key for GPT-4o (used as Label Agent) |
| `BASE_MODEL` | Path to local backbone model (recommend Qwen3-8B) |
| `EMD_MODEL` | Path to embedding model |

### Run the Retrieval Pipeline

8 GPUs Batch Inference + KVCache for deployment:

```bash
bash run_GPUS.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026noquote,
      title={What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty},
      author={Bowei Zhang and Jin Xiao and Guanglei Yue and Qianyu He and Yanghua Xiao and Deqing Yang and Jiaqing Liang},
      year={2026},
      booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
      url={https://arxiv.org/abs/2602.22220},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
