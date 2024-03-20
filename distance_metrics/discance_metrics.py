import faiss
import numpy as np


def norml2_innerproduct(feature_space, query):

    index = faiss.index_factory(
        feature_space.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(feature_space)
    index.add(feature_space)
    distance, index = index.search(np.array([query]), k=feature_space.shape[0])

    return distance, index


