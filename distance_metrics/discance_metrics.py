import faiss
import numpy as np
from scipy.spatial import KDTree


def batch_compute_distances(
    feature_space: np.ndarray,
    query: np.ndarray,
    metric=faiss.METRIC_INNER_PRODUCT,
    k: int = 10,
):

    # Indexing setup
    if metric == "HNSWFlat":
        print("HNSWFlat")
        index = faiss.IndexHNSWFlat(feature_space.shape[1], 120)
        index.hnsw.efConstruction = 320
        index.hnsw.efSearch = 170
    else:
        index = faiss.index_factory(feature_space.shape[1], "Flat", metric)

    # Metric specific tasks
    if metric is faiss.METRIC_INNER_PRODUCT:
        print("normalizing")
        faiss.normalize_L2(feature_space)

    # Build the index
    index.add(feature_space)

    # Query
    if query.shape[0] == 1:
        query = np.array([query])
    distance, index = index.search(query, k=k)

    return distance, index


def batch_compute_kd_trees(feature_space: np.ndarray, query: np.ndarray, k: int = 10):

    result = KDTree(
        feature_space,
        leafsize=1,
        balanced_tree=True,
        compact_nodes=True,
        copy_data=True,
    )
    distance, index = result.query(query, k=k)
    return distance, index


def evaluate_index_matched_results(index: np.ndarray):
    total = index.shape[0]
    top1 = 0
    top5 = 0
    top10 = 0
    for i, indeces in enumerate(index):
        if i == indeces[0]:
            top1 += 1
        if i in indeces[:5]:
            top5 += 1
        if i in indeces:
            top10 += 1
    return {"top1": top1 / total, "top5": top5 / total, "top10": top10 / total}
