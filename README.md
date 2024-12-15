## Repo of ZeroC

A zero-overhead KV dimensionality compression system for KV cache reduction in LLM inference.

## Datasets

### Accuracy Metric Selection

- For classification tasks and information retrieval tasks, we use the **accuracy** as the metric.
- For summarization tasks, we use **ROUGE-1** as the accuracy score.
- For code completion, we calculate the accuracy by **dividing the number of generated code that can be compiled by the total number of generated code**.

### Dataset Dir
IMDb movie genre classification: /zeroc/datasets/imdb

arXiv summarization: /zeroc/datasets/arxiv

Cocktail for information retrieval: /zeroc/datasets/cocktail

HumanEval for code completion: /zeroc/datasets/humaneval

### ZeroC Dir
/zeroc  # ZeroC related code
- datasets
- exp
- kernels
- measurements
- quantization
- svd_qkv

/vllm  # vLLM base code

```
datasets: it contains the datasets we use for validation.
exp: it has the implementation code of zeroc.
       # zeroc.py/keyformer(-z).py/kvquant(-z).py
kernels: kernel functions.
measurements: it has the measurement code for SVD and model analysis.
quantization: it has the code of quantization methods.
svd_qkv: SVD related code for QKV compression and analysis.
```
