# Create Typesense Schema

Previously, build typesense docker `docker-compose -up`

Then run `crate_schema.py`

# Prep test data


```python
import sys
sys.path.insert(0, '..') # add parent folder path
from typesense_test.create_schema import client, schema
```


```python
client.collections['targets'].documents.delete({'filter_by': 'concept_id: >=0'})

```




    {'num_deleted': 2222}




```python
import pickle

with open('../lib/artifacts/dicts/sources.pickle', 'rb') as handle:
    sources_dict = pickle.load(handle)
with open('../lib/artifacts/dicts/targets.pickle', 'rb') as handle:
    targets_dict = pickle.load(handle)
```

Clean source descriptions prior to quering


```python
import re
for k,v in sources_dict.items():
    sources_dict[k] = " ".join(re.sub('[!,*)@#%(&$_?.^"]', ' ', v).split())
```


```python
target_dict = [{"concept_id": k, "target": v} for k, v in targets_dict.items()]
```

Examples


```python
list(sources_dict.items())[:5]
```




    [(0, 'HUC-OFALMOLOGY UVEITES'),
     (1, 'Here-oftalmologia'),
     (2, 'HECE-OFTALMOLOGIA CIR Blanket'),
     (3, 'Huc-Oftalmology Contactology'),
     (4, 'Huc-Oftalmology Contactology')]




```python
list(targets_dict.items())[:5]
```




    [(0, 'Uveitis and Ocular Inflammatory Disease Ophthalmology'),
     (1, 'Ophthalmology'),
     (2, 'Retina Ophthalmology'),
     (3, 'Ophthalmology'),
     (4, 'Corneal and Contact Management Optometrist')]



# Load typesense with target data


```python
client_response = client.collections['targets'].documents.import_(target_dict, {'action': 'upsert'}, batch_size=10000)
print(client_response[:5])
```

    [{'success': True}, {'success': True}, {'success': True}, {'success': True}, {'success': True}]


# Test top1, top5 and top10 recall on 2222 test samples


```python
from time import time


def test_func(parameters):

    total = len(sources_dict)
    top1 = 0
    top5 = 0
    top10 = 0

    # Track Encoding Time
    start = time()

    for i in range(total):
        query = sources_dict[i]

        search_parameters = {"q": query}
        search_parameters.update(parameters)

        results = client.collections["targets"].documents.search(search_parameters)
        matches = [i["document"]["concept_id"] for i in results["hits"]]
        if i == matches[0]:
            top1 += 1
        if i in matches[:5]:
            top5 += 1
        if i in matches[:10]:
            top10 += 1

    # Compute time
    end = time()
    elapsed_seconds = end - start

    print(
        f"top1: {top1/total:.2%}, top5: {top5/total:.2%}, top10: {top10/total:.2%}, n_tests: {total}, elapsed_seconds: {elapsed_seconds}, preds_per_second_X1000: {total/elapsed_seconds/1000}"
    )
```

By setting a high k and a high flat_search_cutoff we can force it to use brute-force search, which is much better. 


```python
parameters = {
    "query_by": "embedding",
    "vector_query": "embedding:([],k:1000,distance_threshold:1.00,flat_search_cutoff:1000)",
}
test_func(parameters)
```

    top1: 49.05%, top5: 72.05%, top10: 77.86%, n_tests: 2222, elapsed_seconds: 35.875524044036865, preds_per_second_X1000: 0.06193637749437517


But this is a bit sketchy, since it depends on K (which is the number of retrieved items). If we set K to 10 and maintain the high cutoff the results are poorer because it is not using brute-force


```python
parameters = {
    "query_by": "embedding",
    "vector_query": "embedding:([],k:10,distance_threshold:1.00,flat_search_cutoff:1000)",
}
test_func(parameters)
```

    top1: 42.35%, top5: 60.76%, top10: 65.17%, n_tests: 2222, elapsed_seconds: 34.33804202079773, preds_per_second_X1000: 0.06470957192766517


Still, HNSW still has some parameters that are not accessible through this API. 

# Hybrid Search

One option is to do hybrid search, with vector and word based search. The weight for each component is controled by a single `alpha` parameter


```python
parameters = {
    "query_by": "embedding,target",
    "vector_query": "embedding:([],k:1000,distance_threshold:1.00,flat_search_cutoff:1000,alpha:0.7)",
}
test_func(parameters)
```

    top1: 48.69%, top5: 73.04%, top10: 78.94%, n_tests: 2222, elapsed_seconds: 33.37315487861633, preds_per_second_X1000: 0.0665804598960386

