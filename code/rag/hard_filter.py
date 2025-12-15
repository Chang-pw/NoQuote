from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize

# Load your embedding model
from utils import load_config
config = load_config()
EMD_MODEL = config['EMD_MODEL']
model = SentenceTransformer(EMD_MODEL)

# Function to get normalized embeddings with dimension shrinkage

def get_embeddings(documents, dim=1024):
    embeddings = model.encode(documents, normalize_embeddings=False)
    # Shrink dimensions
    embeddings = embeddings[..., :dim]
    # L2 normalize
    embeddings = normalize(embeddings, norm="l2", axis=1)
    return embeddings



def hard_label_filter(quote_labels, context_label, threshold=0.8,dim=1024):
    filtered = []

    emb_refs = get_embeddings(context_label, dim=dim)

    for quote_label in quote_labels:
        emb_ins = get_embeddings(quote_label, dim=dim)
        # Compute cosine similarity matrix: (len(insights), len(context_insights))
        sims_matrix = np.dot(emb_ins, emb_refs.T)

        score = float(sims_matrix.max())

        if score >= threshold:
            filtered.append(True)
        else:
            filtered.append(False)

    return filtered

