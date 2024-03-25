import pickle
from sentence_transformers import SentenceTransformer
import re
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def encode_sources_targets(plm):

    with open("../lib/artifacts/dicts/sources.pickle", "rb") as handle:
        sources_dict = pickle.load(handle)
    with open("../lib/artifacts/dicts/targets.pickle", "rb") as handle:
        targets_dict = pickle.load(handle)

    # Clean
    for k, v in sources_dict.items():
        sources_dict[k] = " ".join(re.sub('[!,*)@#%(&$_?.^"]', " ", v).split())

    sources = [("query: " + v) for k, v in sources_dict.items()]
    targets = [("query: " + v) for k, v in targets_dict.items()]

    model = SentenceTransformer(plm, trust_remote_code=False)

    # Encode
    sources_emb = model.encode(sources, normalize_embeddings=True)
    targets_emb = model.encode(targets, normalize_embeddings=True)

    with open("../lib/artifacts/dicts/sources_emb.pickle", "wb") as handle:
        pickle.dump(
            dict(zip(range(len(sources_emb)), sources_emb)),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with open("../lib/artifacts/dicts/targets_emb.pickle", "wb") as handle:
        pickle.dump(
            dict(zip(range(len(targets_emb)), targets_emb)),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("Encoding Complete")


if __name__ == "__main__":
    plm = "intfloat/multilingual-e5-small"
    encode_sources_targets(plm)
