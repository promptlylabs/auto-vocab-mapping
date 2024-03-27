# Auto Vocab Mapping
___

## POC 1 - Vector Space Search

For the first POC I'll focus on source and target descriptions only. So I just need previously matched sources and targets.


```python
import pandas as pd
```

Read CHUC example files and see what's in it


```python
chuc_s_df = pd.read_csv("../lib/data/raw/source_codes_description/chuc/analises_cod_acto.csv")
chuc_s2c_df = pd.read_csv("../lib/data/raw/source_to_concept/chuc/source_to_standard_analises_cod_acto.csv")
concept = pd.read_csv("../lib/data/raw/vocabularies/CONCEPT.csv", low_memory=False)
```


```python
concept.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>concept_id</th>
      <th>concept_name</th>
      <th>domain_id</th>
      <th>vocabulary_id</th>
      <th>concept_class_id</th>
      <th>standard_concept</th>
      <th>concept_code</th>
      <th>valid_start_date</th>
      <th>valid_end_date</th>
      <th>invalid_reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45756805</td>
      <td>Pediatric Cardiology</td>
      <td>Provider</td>
      <td>ABMS</td>
      <td>Physician Specialty</td>
      <td>S</td>
      <td>OMOP4821938</td>
      <td>19700101</td>
      <td>20991231</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45756804</td>
      <td>Pediatric Anesthesiology</td>
      <td>Provider</td>
      <td>ABMS</td>
      <td>Physician Specialty</td>
      <td>S</td>
      <td>OMOP4821939</td>
      <td>19700101</td>
      <td>20991231</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45756803</td>
      <td>Pathology-Anatomic / Pathology-Clinical</td>
      <td>Provider</td>
      <td>ABMS</td>
      <td>Physician Specialty</td>
      <td>S</td>
      <td>OMOP4821940</td>
      <td>19700101</td>
      <td>20991231</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45756802</td>
      <td>Pathology - Pediatric</td>
      <td>Provider</td>
      <td>ABMS</td>
      <td>Physician Specialty</td>
      <td>S</td>
      <td>OMOP4821941</td>
      <td>19700101</td>
      <td>20991231</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45756801</td>
      <td>Pathology - Molecular Genetic</td>
      <td>Provider</td>
      <td>ABMS</td>
      <td>Physician Specialty</td>
      <td>S</td>
      <td>OMOP4821942</td>
      <td>19700101</td>
      <td>20991231</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
concept['concept_id'].dtype
```




    dtype('int64')




```python
set_dtype = concept['concept_id'].dtype
```

Make dict to map quickly


```python
target_dict = dict(zip(concept['concept_id'], concept['concept_name']))
```

From here I need concept_id and concept_name to map


```python
chuc_s_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_code</th>
      <th>source_description</th>
      <th>translated_source_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A21900</td>
      <td>FERRO, S</td>
      <td>Ferro, s</td>
    </tr>
    <tr>
      <th>1</th>
      <td>X34281</td>
      <td>AN¡LISE POR SEQUENCIA«√O EM LARGA ESCALA (~0,5MB</td>
      <td>LARGE SCALE SEQUENCE ANALYSIS (~0.5MB</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A22375</td>
      <td>CYFRA 21-1</td>
      <td>DIGIT 21-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A21646</td>
      <td>DELTA-4-ANDROSTENEDIONA, S</td>
      <td>DELTA-4-ANDROSTENEDIONA, S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A25520</td>
      <td>ANTICORPOS ANTI-NUCLEARES E CITOPLASMATICOS (A...</td>
      <td>ANTI-NUCLEAR AND CYTOPLASMATIC ANTIBODIES (ANT...</td>
    </tr>
  </tbody>
</table>
</div>



These are translations. We're not going into this for now. A separate exploration will be carried out for this topic alone. We could fine-tune our own medical data whichi has its specificities. We'll need: 
- Medical terms translation
- Acronym desambiguation


```python
chuc_s2c_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_code</th>
      <th>source_concept_id</th>
      <th>source_vocabulary_id</th>
      <th>source_code_description</th>
      <th>target_concept_id</th>
      <th>target_vocabulary_id</th>
      <th>valid_start_date</th>
      <th>valid_end_date</th>
      <th>invalid_reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A22793</td>
      <td>0</td>
      <td>analises_cod_acto</td>
      <td>SODIO, S/U</td>
      <td>3022810</td>
      <td>LOINC</td>
      <td>1970-01-01</td>
      <td>2099-12-31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A24347</td>
      <td>0</td>
      <td>analises_cod_acto</td>
      <td>TEMPO DE PROTROMBINA, S</td>
      <td>4245261</td>
      <td>SNOMED</td>
      <td>1970-01-01</td>
      <td>2099-12-31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A21789</td>
      <td>0</td>
      <td>analises_cod_acto</td>
      <td>EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S</td>
      <td>3013290</td>
      <td>LOINC</td>
      <td>1970-01-01</td>
      <td>2099-12-31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A21789</td>
      <td>0</td>
      <td>analises_cod_acto</td>
      <td>EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S</td>
      <td>3027315</td>
      <td>LOINC</td>
      <td>1970-01-01</td>
      <td>2099-12-31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A21789</td>
      <td>0</td>
      <td>analises_cod_acto</td>
      <td>EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S</td>
      <td>4013965</td>
      <td>SNOMED</td>
      <td>1970-01-01</td>
      <td>2099-12-31</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



From here I need the source code description and the target concept id. This is what well need in large quantities if we want to train a translator or a classifier. 


```python
chuc_df = chuc_s2c_df[["source_code_description", "target_concept_id"]]
chuc_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_code_description</th>
      <th>target_concept_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SODIO, S/U</td>
      <td>3022810</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEMPO DE PROTROMBINA, S</td>
      <td>4245261</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S</td>
      <td>3013290</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S</td>
      <td>3027315</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S</td>
      <td>4013965</td>
    </tr>
  </tbody>
</table>
</div>



### Map target concepts and check missing values


```python
chuc_df.loc[:, 'concept_name'] = chuc_df['target_concept_id'].astype(set_dtype).map(target_dict)
```

    /var/folders/d4/zjykh2gs0ddchpt100y0llr40000gp/T/ipykernel_10268/3434074019.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      chuc_df.loc[:, 'concept_name'] = chuc_df['target_concept_id'].astype(set_dtype).map(target_dict)



```python
chuc_df[chuc_df.isna().any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_code_description</th>
      <th>target_concept_id</th>
      <th>concept_name</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
chuc_s2c = chuc_df.dropna()
```


```python
chuc_s2c.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_code_description</th>
      <th>target_concept_id</th>
      <th>concept_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SODIO, S/U</td>
      <td>3022810</td>
      <td>Sodium [Moles/volume] in Body fluid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEMPO DE PROTROMBINA, S</td>
      <td>4245261</td>
      <td>Prothrombin time</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S</td>
      <td>3013290</td>
      <td>Carbon dioxide [Partial pressure] in Blood</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S</td>
      <td>3027315</td>
      <td>Oxygen [Partial pressure] in Blood</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S</td>
      <td>4013965</td>
      <td>Oxygen saturation measurement, arterial</td>
    </tr>
  </tbody>
</table>
</div>




```python
sources = chuc_s2c["source_code_description"].tolist()
sources[:10]
```




    ['SODIO, S/U',
     'TEMPO DE PROTROMBINA, S',
     'EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S',
     'EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S',
     'EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S',
     'EQUILIBRIO ACIDO BASICO (PH, PC02,SAT O2,CO2,), S',
     'PESQUISA DE RNA DO VÕRUS SARS-COV-2 POR PCR EM TEMPO REAL',
     'SODIO, S/U',
     'POTASSIO, S/U',
     'GLUCOSE, DOSEAMENTO, S/U/L']




```python
targets = chuc_s2c["concept_name"].tolist()
targets[:10]
```




    ['Sodium [Moles/volume] in Body fluid',
     'Prothrombin time',
     'Carbon dioxide [Partial pressure] in Blood',
     'Oxygen [Partial pressure] in Blood',
     'Oxygen saturation measurement, arterial',
     'Hydrogen ion concentration',
     'PCR test for SARS',
     'Sodium measurement, serum',
     'Potassium level',
     'Glucose measurement, plasma']



Some lm are trained as seq2seq and need the `query` and `passage` prefixes.


```python
sources = [("query: " + i) for i in sources]
targets = [("query: " + i) for i in targets]
```


```python
assert len(sources) == len(targets)
```

### Encode texts into fixed sized mean pooled vectors. 

Encode using torch. 


```python
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np


class TextEncoder:
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

    def encode(self, texts):
        # Tokenize the input texts
        batch_dict = self.tokenizer(texts,
                                    max_length=512,
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = TextEncoder.__average_pool(
            outputs.last_hidden_state, batch_dict['attention_mask'])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return np.array(embeddings.detach(), dtype=np.float32)

    @staticmethod
    def __average_pool(last_hidden_states: Tensor,
                       attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
```


```python
model_name = 'intfloat/multilingual-e5-small'
embeddings = TextEncoder(model_name).encode(sources)
```

By default, sentence_transformers disables the parallelism to avoid any hidden deadlock that would be hard to debug


```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
```


```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-small')
sources_emb = model.encode(sources, normalize_embeddings=True)
targets_emb = model.encode(targets, normalize_embeddings=True)
```

Sentence_transformer's implementation is faster than my manual approach so I'll stick to that. If in any case it has some incopatibility with a newer model I'll use mine. 


```python
embeddings.shape
```




    (428, 384)




```python
embeddings[:10]
```




    array([[ 0.04107222, -0.02139925, -0.0353516 , ...,  0.10612641,
             0.06803897,  0.00305798],
           [ 0.03053902,  0.00532008, -0.00736805, ...,  0.07866557,
             0.07970123,  0.03390224],
           [ 0.03296087, -0.02778542, -0.01535674, ...,  0.06343906,
             0.08069205,  0.02741423],
           ...,
           [ 0.04107222, -0.02139925, -0.0353516 , ...,  0.10612641,
             0.06803897,  0.00305798],
           [ 0.06119867, -0.0139116 , -0.02081711, ...,  0.07180168,
             0.05883745,  0.0364945 ],
           [ 0.04395493,  0.00539457, -0.05504538, ...,  0.07089277,
             0.07933093,  0.05188119]], dtype=float32)



Everything seems fine with the resulting vector space.

# PCA
Exploring projections in the vector space


```python
from sklearn.decomposition import PCA

def compute_pca(vectors):
    pca = PCA()
    pca.fit(vectors)
    pcs = pca.transform(vectors)
    return pcs
```


```python
import sys
sys.path.insert(0, '..') # add parent folder path

from utils.plotting import plot_pca, parallel
```


```python
stacked = np.vstack([sources_emb, targets_emb])
print(stacked.shape)
```

    (856, 384)



```python
stacked_pcs = compute_pca(stacked)
```

Prepare labels, clusters and colors for PCA


```python
# labels
names = sources + targets
sources_ids = ["source" for _ in sources]
targets_ids = ["target" for _ in targets]
group_names = sources_ids + targets_ids
# colors
color_by_group = sources_ids + targets_ids
individual_names = targets + targets
```


```python
plot_pca(pcs=stacked_pcs, colors=color_by_group, names=group_names, title='PCA colored by group (source, target)')
```



Clusters relate to the languages.

matches (sources - targets) should be closer if we color them the same


```python
plot_pca(pcs=stacked_pcs[:20], colors=individual_names[:20], names=individual_names[:20], title="PCA of 20 matched examples (same colors should be closer)")
```




```python
source_dict = dict(zip(range(len(sources)), sources))
target_dict = dict(zip(range(len(targets)), targets))
```


```python
rand_number = np.random.choice(len(sources), 1, replace=True)[0]
source_example = sources_emb[rand_number]
```


```python
print(f' source: {source_dict[rand_number]};\n target: {target_dict[rand_number]}')
```

     source: query: HEMOGRAMA COM FORMULA LEUCOCITARIA (ERITROGRAMA, CONTAGEM DE LEUCOCITOS, CONTAGEM DE PLAQU;
     target: query: Complete blood count with white cell differential, automated


# Test distance: Compute nomalized L2 inner product


```python
import faiss


def norml2_innerproduct(feature_space, query):

    index = faiss.index_factory(
        feature_space.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(feature_space)
    index.add(feature_space)
    distance, index = index.search(np.array([query]), k=feature_space.shape[0])

    return distance, index
```


```python
distance, index = norml2_innerproduct(targets_emb, source_example)
```


```python
print(f' source: {source_dict[rand_number]};\n target: {target_dict[index[0][0]]}')
```

     source: query: HEMOGRAMA COM FORMULA LEUCOCITARIA (ERITROGRAMA, CONTAGEM DE LEUCOCITOS, CONTAGEM DE PLAQU;
     target: query: Hemoglobin C/Hemoglobin.total in Blood by HPLC



```python
top1 = 0
top5 = 0
top10 = 0
total = len(sources)
for i in range(total):
    source_example = sources_emb[i]
    distance, index = norml2_innerproduct(targets_emb, source_example)
    
    if i == index[0][0]:
        top1+=1
        top5+=1
        top10+=1
    elif i in index[0][:5]:
        top5+=1
        top10+=1
    elif i in index[0][:10]:
        top10+=1

    
```


```python
print(f"""
      Top 1 match: {top1/total:.2%};
      Top 5 match: {top5/total:.2%};ok
      Top 10 match: {top10/total:.2%};
      Total number of tests: {len(sources)}
    """)
```

    
          Top 1 match: 44.16%;
          Top 5 match: 70.56%;ok
          Top 10 match: 78.27%;
          Total number of tests: 428
        


Distance metric seems to be correctly implemented. Now we need more examples to test on.

# Expand the number of examples



```python
import sys
sys.path.insert(0, '..') # add parent folder path
from data_preprocessors import RawDataProcessor
```


```python
hospital_folders = ["../lib/data/raw/source_to_concept/chuc/", "../lib/data/raw/source_to_concept/hds/"]
concept_vocab = "../lib/data/raw/vocabularies/CONCEPT.csv"

rdp = RawDataProcessor(vocab_file=concept_vocab, hospital_folders=hospital_folders)
sources, targets = rdp.join_source_target()
```


```python
assert len(sources) == len(targets)
```


```python
len(sources)
```




    2222




```python
source_dict = dict(zip(range(len(sources)), sources))
target_dict = dict(zip(range(len(targets)), targets))
```


```python
import pickle 
with open('../lib/artifacts/dicts/sources.pickle', 'wb') as handle:
    pickle.dump(source_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../lib/artifacts/dicts/targets.pickle', 'wb') as handle:
    pickle.dump(target_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

```

# Select models


```python
list_of_models = [ 
                  "sentence-transformers/distiluse-base-multilingual-cased-v2", # 2019 maps sentences & paragraphs to a 512 dimensional dense vector space and can be used for tasks like clustering or semantic search
                  "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", # 2019 maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search
                  "mixedbread-ai/mxbai-embed-large-v1", # 2024 It achieves SOTA performance on BERT-large scale (feature extraction)
                  'intfloat/multilingual-e5-small', # 2024 This model has 12 layers and the embedding size is 384 (sentence similarity)
                  "intfloat/multilingual-e5-base", # 2024 This model has 24 layers and the embedding size is 1024 (sentence similarity)
                  "intfloat/multilingual-e5-large", # 2024 This model has 12 layers and the embedding size is 768 (sentence similarity)
                  "sentence-transformers/all-MiniLM-L6-v2", # maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search
                  "sentence-transformers/all-MiniLM-L12-v2", # maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search
                  # "Henrychur/MMedLM2", # too large for now
                  "medicalai/ClinicalBERT" # 2023 The ClinicalBERT model was trained on a large multicenter dataset with a large corpus of 1.2B words of diverse diseases
]
```

# Top1, Top5, and Top10 recall for 2222 examples

For testing we need to apply different query prefixes since some models are trained with specific start tokens for query and response. 

Here, we'll test with a "query: " prefix and without. The performance differences can be quite unpredictable. 

The test function bellow tracks if the model needs remote code to run (some models do), and the recall@k metrics for each model whith each prefix. 

By default, the test will be conducted using the L2 search of the normalized IP. 






```python
from tqdm import tqdm
from time import time
from sentence_transformers import SentenceTransformer


def test_models(models: list, sources: list, targets: list):

    # Store results
    results = []

    for plm in tqdm(models, desc="Testing models: "):

        # Load model
        needs_remote_code = 0
        try:
            model = SentenceTransformer(plm, trust_remote_code=False)
        except ValueError:
            model = SentenceTransformer(plm, trust_remote_code=True)
            needs_remote_code = 1
        
        for query_prefix in ['', 'query: ']:

            mod_sources = [(query_prefix + i) for i in sources]
            mod_targets = [(query_prefix + i) for i in targets]

            # Track results
            top1 = 0
            top5 = 0
            top10 = 0
            total = len(sources)

            # Encode
            sources_emb = model.encode(mod_sources, normalize_embeddings=True)
            targets_emb = model.encode(mod_targets, normalize_embeddings=True)

            # Track Encoding Time
            start = time()
            for i in tqdm(range(total), leave=False):

                # Compute distances
                source_example = sources_emb[i]
                distance, index = norml2_innerproduct(targets_emb, source_example)

                # Check matches
                if i == index[0][0]:
                    top1 += 1
                    top5 += 1
                    top10 += 1
                elif i in index[0][:5]:
                    top5 += 1
                    top10 += 1
                elif i in index[0][:10]:
                    top10 += 1

            # Compute time
            end = time()
            elapsed_seconds = end - start

            results.append(
                {   
                    "plm": plm + '__query_prefix__' + query_prefix,
                    "remote_code": needs_remote_code,
                    "Top-1 match": top1/total,
                    "Top-5 match": top5/total,
                    "Top-10 match": top10/total,
                    "Total number of tests": len(sources),
                    "Elapsed seconds": elapsed_seconds,
                    "Predictions per second X 1000": len(sources)/elapsed_seconds/1000
                }
            )

    return results
```


```python
results = test_models(list_of_models, sources, targets)
```

    Testing models:  89%|████████▉ | 8/9 [02:59<00:19, 19.19s/it]No sentence-transformers model found with name medicalai/ClinicalBERT. Creating a new one with MEAN pooling.
    Testing models: 100%|██████████| 9/9 [03:11<00:00, 21.31s/it]


Append USAGI's reported results


```python
usagis = {"plm": 'USAGI', "Top-1 match": 0.42, "Top-5 match": 0.58, "Top-10 match": 0.62} # From toki paper
results.append(usagis)
```


```python
import pandas as pd
results_df = pd.DataFrame.from_dict(results)
results_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>plm</th>
      <th>remote_code</th>
      <th>Top-1 match</th>
      <th>Top-5 match</th>
      <th>Top-10 match</th>
      <th>Total number of tests</th>
      <th>Elapsed seconds</th>
      <th>Predictions per second X 1000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sentence-transformers/distiluse-base-multiling...</td>
      <td>0.0</td>
      <td>0.233573</td>
      <td>0.446445</td>
      <td>0.536004</td>
      <td>2222.0</td>
      <td>1.465041</td>
      <td>1.516681</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sentence-transformers/distiluse-base-multiling...</td>
      <td>0.0</td>
      <td>0.244824</td>
      <td>0.439244</td>
      <td>0.518902</td>
      <td>2222.0</td>
      <td>1.422760</td>
      <td>1.561753</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sentence-transformers/paraphrase-multilingual-...</td>
      <td>0.0</td>
      <td>0.365437</td>
      <td>0.617462</td>
      <td>0.680018</td>
      <td>2222.0</td>
      <td>2.086202</td>
      <td>1.065093</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sentence-transformers/paraphrase-multilingual-...</td>
      <td>0.0</td>
      <td>0.364986</td>
      <td>0.607111</td>
      <td>0.676868</td>
      <td>2222.0</td>
      <td>2.038171</td>
      <td>1.090193</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mixedbread-ai/mxbai-embed-large-v1__query_pref...</td>
      <td>0.0</td>
      <td>0.480198</td>
      <td>0.761026</td>
      <td>0.823582</td>
      <td>2222.0</td>
      <td>2.803631</td>
      <td>0.792544</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mixedbread-ai/mxbai-embed-large-v1__query_pref...</td>
      <td>0.0</td>
      <td>0.469847</td>
      <td>0.737624</td>
      <td>0.809181</td>
      <td>2222.0</td>
      <td>2.751335</td>
      <td>0.807608</td>
    </tr>
    <tr>
      <th>6</th>
      <td>intfloat/multilingual-e5-small__query_prefix__</td>
      <td>0.0</td>
      <td>0.484248</td>
      <td>0.714671</td>
      <td>0.775878</td>
      <td>2222.0</td>
      <td>1.142786</td>
      <td>1.944371</td>
    </tr>
    <tr>
      <th>7</th>
      <td>intfloat/multilingual-e5-small__query_prefix__...</td>
      <td>0.0</td>
      <td>0.479748</td>
      <td>0.718722</td>
      <td>0.784878</td>
      <td>2222.0</td>
      <td>1.154759</td>
      <td>1.924211</td>
    </tr>
    <tr>
      <th>8</th>
      <td>intfloat/multilingual-e5-base__query_prefix__</td>
      <td>0.0</td>
      <td>0.471647</td>
      <td>0.710171</td>
      <td>0.777678</td>
      <td>2222.0</td>
      <td>2.056411</td>
      <td>1.080523</td>
    </tr>
    <tr>
      <th>9</th>
      <td>intfloat/multilingual-e5-base__query_prefix__q...</td>
      <td>0.0</td>
      <td>0.469397</td>
      <td>0.693069</td>
      <td>0.763726</td>
      <td>2222.0</td>
      <td>2.052016</td>
      <td>1.082838</td>
    </tr>
    <tr>
      <th>10</th>
      <td>intfloat/multilingual-e5-large__query_prefix__</td>
      <td>0.0</td>
      <td>0.500450</td>
      <td>0.743024</td>
      <td>0.805581</td>
      <td>2222.0</td>
      <td>2.793587</td>
      <td>0.795393</td>
    </tr>
    <tr>
      <th>11</th>
      <td>intfloat/multilingual-e5-large__query_prefix__...</td>
      <td>0.0</td>
      <td>0.509001</td>
      <td>0.733123</td>
      <td>0.792979</td>
      <td>2222.0</td>
      <td>2.788705</td>
      <td>0.796786</td>
    </tr>
    <tr>
      <th>12</th>
      <td>sentence-transformers/all-MiniLM-L6-v2__query_...</td>
      <td>0.0</td>
      <td>0.414491</td>
      <td>0.682268</td>
      <td>0.754275</td>
      <td>2222.0</td>
      <td>1.157990</td>
      <td>1.918842</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sentence-transformers/all-MiniLM-L6-v2__query_...</td>
      <td>0.0</td>
      <td>0.396940</td>
      <td>0.654365</td>
      <td>0.725023</td>
      <td>2222.0</td>
      <td>1.175839</td>
      <td>1.889715</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sentence-transformers/all-MiniLM-L12-v2__query...</td>
      <td>0.0</td>
      <td>0.432493</td>
      <td>0.699820</td>
      <td>0.765077</td>
      <td>2222.0</td>
      <td>1.153568</td>
      <td>1.926198</td>
    </tr>
    <tr>
      <th>15</th>
      <td>sentence-transformers/all-MiniLM-L12-v2__query...</td>
      <td>0.0</td>
      <td>0.424842</td>
      <td>0.676868</td>
      <td>0.746175</td>
      <td>2222.0</td>
      <td>1.161286</td>
      <td>1.913396</td>
    </tr>
    <tr>
      <th>16</th>
      <td>medicalai/ClinicalBERT__query_prefix__</td>
      <td>0.0</td>
      <td>0.285329</td>
      <td>0.466247</td>
      <td>0.527453</td>
      <td>2222.0</td>
      <td>2.121597</td>
      <td>1.047324</td>
    </tr>
    <tr>
      <th>17</th>
      <td>medicalai/ClinicalBERT__query_prefix__query:</td>
      <td>0.0</td>
      <td>0.255626</td>
      <td>0.421242</td>
      <td>0.489649</td>
      <td>2222.0</td>
      <td>2.087973</td>
      <td>1.064190</td>
    </tr>
    <tr>
      <th>18</th>
      <td>USAGI</td>
      <td>NaN</td>
      <td>0.420000</td>
      <td>0.580000</td>
      <td>0.620000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Filter best performant pretrained models


```python
results_df = results_df.loc[
    (results_df['Top-1 match'] >= usagis['Top-1 match']) &
    (results_df['Top-5 match'] >= usagis['Top-5 match']) &
    (results_df['Top-10 match'] >= usagis['Top-10 match']), :]

results_df.sort_values(by=['Top-1 match', 'Top-5 match', 'Top-10 match'], ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>plm</th>
      <th>remote_code</th>
      <th>Top-1 match</th>
      <th>Top-5 match</th>
      <th>Top-10 match</th>
      <th>Total number of tests</th>
      <th>Elapsed seconds</th>
      <th>Predictions per second X 1000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>intfloat/multilingual-e5-large__query_prefix__...</td>
      <td>0.0</td>
      <td>0.509001</td>
      <td>0.733123</td>
      <td>0.792979</td>
      <td>2222.0</td>
      <td>2.788705</td>
      <td>0.796786</td>
    </tr>
    <tr>
      <th>10</th>
      <td>intfloat/multilingual-e5-large__query_prefix__</td>
      <td>0.0</td>
      <td>0.500450</td>
      <td>0.743024</td>
      <td>0.805581</td>
      <td>2222.0</td>
      <td>2.793587</td>
      <td>0.795393</td>
    </tr>
    <tr>
      <th>6</th>
      <td>intfloat/multilingual-e5-small__query_prefix__</td>
      <td>0.0</td>
      <td>0.484248</td>
      <td>0.714671</td>
      <td>0.775878</td>
      <td>2222.0</td>
      <td>1.142786</td>
      <td>1.944371</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mixedbread-ai/mxbai-embed-large-v1__query_pref...</td>
      <td>0.0</td>
      <td>0.480198</td>
      <td>0.761026</td>
      <td>0.823582</td>
      <td>2222.0</td>
      <td>2.803631</td>
      <td>0.792544</td>
    </tr>
    <tr>
      <th>7</th>
      <td>intfloat/multilingual-e5-small__query_prefix__...</td>
      <td>0.0</td>
      <td>0.479748</td>
      <td>0.718722</td>
      <td>0.784878</td>
      <td>2222.0</td>
      <td>1.154759</td>
      <td>1.924211</td>
    </tr>
    <tr>
      <th>8</th>
      <td>intfloat/multilingual-e5-base__query_prefix__</td>
      <td>0.0</td>
      <td>0.471647</td>
      <td>0.710171</td>
      <td>0.777678</td>
      <td>2222.0</td>
      <td>2.056411</td>
      <td>1.080523</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mixedbread-ai/mxbai-embed-large-v1__query_pref...</td>
      <td>0.0</td>
      <td>0.469847</td>
      <td>0.737624</td>
      <td>0.809181</td>
      <td>2222.0</td>
      <td>2.751335</td>
      <td>0.807608</td>
    </tr>
    <tr>
      <th>9</th>
      <td>intfloat/multilingual-e5-base__query_prefix__q...</td>
      <td>0.0</td>
      <td>0.469397</td>
      <td>0.693069</td>
      <td>0.763726</td>
      <td>2222.0</td>
      <td>2.052016</td>
      <td>1.082838</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sentence-transformers/all-MiniLM-L12-v2__query...</td>
      <td>0.0</td>
      <td>0.432493</td>
      <td>0.699820</td>
      <td>0.765077</td>
      <td>2222.0</td>
      <td>1.153568</td>
      <td>1.926198</td>
    </tr>
    <tr>
      <th>15</th>
      <td>sentence-transformers/all-MiniLM-L12-v2__query...</td>
      <td>0.0</td>
      <td>0.424842</td>
      <td>0.676868</td>
      <td>0.746175</td>
      <td>2222.0</td>
      <td>1.161286</td>
      <td>1.913396</td>
    </tr>
    <tr>
      <th>18</th>
      <td>USAGI</td>
      <td>NaN</td>
      <td>0.420000</td>
      <td>0.580000</td>
      <td>0.620000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_df.drop(['Total number of tests'], axis=1, inplace=True)
```

# Plot results


```python

parallel(results_df, label='plm')         
```



# Conclusions

We can see that through this approach we can easily beat USAGI's reported performance (values from the literature - TOKI paper). 

Another curious finding is that bigger is not obviously better. This actually makes sense, since by raising the amount of dimensions, although more information is being captured, the points in space start to become equally distant to each other, so the gains in information don't translate equally to discriminant power. As an example, `multilingual-e5-small` maps tokens to a 384 dimensional vector while `multilingual-e5-large` maps to a 1024 dimensional one. It presents only a very slight improvement at the cost of ram and inference speed. For now the small one seems to be the best suited for a first POC, but these results can be marginally incresed with a bigger one in the future. 
