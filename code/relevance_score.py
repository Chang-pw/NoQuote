from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize

from .utils import load_config


config = load_config()
EMD_MODEL = config["EMD_MODEL"]

# load embedding model once (shared across calls)
_embedding_model = SentenceTransformer(EMD_MODEL)


def _get_embeddings(texts, dim: int = 1024) -> np.ndarray:
    embeddings = _embedding_model.encode(texts, normalize_embeddings=False)
    embeddings = embeddings[..., :dim]
    embeddings = normalize(embeddings, norm="l2", axis=1)
    return embeddings


def compute_relevance_dict(context: str, quotes, dim: int = 1024):
    if not quotes:
        return {}

    quote_list = list(quotes)

    context_emb = _get_embeddings([context], dim=dim)[0]
    quotes_emb = _get_embeddings(quote_list, dim=dim)

    # cosine similarity because embeddings are L2-normalized
    sims = quotes_emb @ context_emb

    # rescale from [-1,1] to [0,1]
    sims = (sims + 1.0) / 2.0

    return {q: float(s) for q, s in zip(quote_list, sims)}
