import unittest
from distance_metrics import (
    batch_compute_distances,
    evaluate_index_matched_results,
)
import pickle
import numpy as np

with open("lib/artifacts/dicts/sources_emb.pickle", "rb") as handle:
    sources_dict = pickle.load(handle)
with open("lib/artifacts/dicts/targets_emb.pickle", "rb") as handle:
    targets_dict = pickle.load(handle)

sources_emb = np.array([v for v in sources_dict.values()])
targets_emb = np.array([v for v in targets_dict.values()])

K = 10

distance, index = batch_compute_distances(targets_emb, sources_emb, k=K)


class TestDistance(unittest.TestCase):

    def test_results_length(self):
        self.assertEqual(index.shape[0], sources_emb.shape[0])

    def test_k_length(self):
        self.assertEqual(index.shape[1], K)


if __name__ == "__main__":
    unittest.main()
