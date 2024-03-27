# Auto Vocab Map

## POC 1: Vector Space Search 🛰️

This approach frames the problem as information-retrieval. It uses pretrained language models to encode text information (in this case, source descriptions). These latent representations are used to search in the multidimensional space the k-nearest neighbors. For that various models and metrics exist. In this case we need multilingual models pretrained for semantic similarity and various metrics to chose the most appropriate for this kind of problem. 

Here you'll find notebooks for concept exploration and model selection, distance metrics implementation and testing and typesense tests. 

Markdown exports of the jupyter notebooks are available as reports, detailing data processing needs, methods, evaluation and conclusions.
___

### Project structure
- `configs`: Where data is accessed and stored and system/model general parameters.
- `lib`: Library with raw or treated data, including figures if aplicable
    - `artifacts`: Data needed asyncronously for different processes. Can be dictionaries that keep track of indeces for features or labels needed to train or predict. Can be pytorch models or pickle files. 
    - `data`: Raw and intermediate data used.
    - `figures`: Exported images from analysis.
- `notebooks`: Jupyter notebooks with the detaild steps for each major component and the respective markdown export.
- `data_processors`: Implemented methods derived from data processing needs.
- `distance_metrics`: Methods for computing and evaluating distance metrics. 
- `tests`: Method unit tests.
- `typesense_test`: Test using Typesense. Aditional info in the respective notebook.
- `utils`: general utilities, such as plotting or logging.
- `requirements.txt`: requirements file to reproduce the analysis environment. (`python=3.12.2`)

___





