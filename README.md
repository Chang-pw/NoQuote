# What Makes an Ideal Quote?
**A Novelty-Driven Quotation Recommendation System**

---

## Overview
Quotation recommendation systems aim to enhance writing by suggesting contextually appropriate quotations.  
This project introduces **quotation novelty**—“unexpected yet rational” quotations—to go beyond mere semantic relevance, thereby enhancing the aesthetically valuable  of writing.

## Method
![intro](figure/method.png)

We complete the entire recommendation pipeline in three steps:

(1) **Label Enhancement**: We leverage a Generative Label Agent to enhance the understanding of both the quotation knowledge base and the user query. The agent outputs their deep semantic meaning and corresponding labels for subsequent retrieval and filtering. In implementation, we pre-assign labels to the entire quotations in advance.
Code in ``code/label_agent.py``

(2) **Rationality Retrieval**: We retrieve candidate quotations that are both semantically relevant to the user query and filter the candidate quotations by the given deep meaning and labels.
Code in ``code/retrieval_pipeline.py``

(3) **Novelty Reranking**: We got a set of semantically reasonable candidate quotations after retrieval. We rerank them by novelty. We measure novelty by token-level logit difference combined with novelty token recognition. Code in ``code/logit_token_level``. And we also consider the popularity and semantic matching to the final rerank score.  We will priorly identify novelty tokens and popularity for each quotation.

## Start

### Install packages

```bash
conda create -n NovelQuote python=3.10
conda activate NovelQuote
pip install -r requirements.txt
```

### Download the Quote Data

Due to anonymity requirements, we are unable to upload the dataset directly to GitHub. However, the label-enhanced quotation data processed by our Label Agent is included in the code supplementary materials submitted with this paper. The dataset will be made publicly available on Hugging Face after the review process.

Put the Quote Data in the ``data/quote/`` folder.

### Config

Please set the config in ``config.json``. Since our method relies on GPT-4o as the Label Agent with API ``API_KEY`` and requires an additional model for computation, please make sure to set up the path to your local model in ``BASE_MODEL``. We recommend using Qwen3-8B as the backbone model for this stage. And for the Embedding model, please also set the model path ``EMD_MODEL`` in the config.

### Run the Retrieval Pipeline

Signle GPU for test:

```bash
bash run_single_GPU.sh
```

8 GPUS Batch Inference + KVCache for deployment:

```bash
bash run_GPUs.sh
```