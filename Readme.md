![Clarifai logo](https://www.clarifai.com/hs-fs/hubfs/logo/Clarifai/clarifai-740x150.png?width=240)

# Clarifai Python Model Utils


[Website](https://www.clarifai.com/) | [Schedule Demo](https://www.clarifai.com/company/schedule-demo) | [Signup for a Free Account](https://clarifai.com/signup) | [API Docs](https://docs.clarifai.com/) | [Clarifai Community](https://clarifai.com/explore) | [Python SDK Docs](https://docs.clarifai.com/python-sdk/api-reference) | [Examples](https://github.com/Clarifai/examples) | [Colab Notebooks](https://github.com/Clarifai/colab-notebooks) | [Discord](https://discord.gg/XAPE3Vtg)

---
## Table Of Contents

* **[Installation](#installation)**
* **[Getting Started](#getting-started)**
* **[Examples](#examples)**

## Installation

Install from Source:

```bash
git clone https://github.com/Clarifai/clarifai-model-utils
cd clarifai-model-utils
python3 -m venv env
source env/bin/activate
pip install -e .
```

## Getting started

Set your CLARIFAI_PAT as an environment variable.

Quick demo of `LLM Evaluation`

```python
from clarifai_model_utils import ClarifaiEvaluator
from clarifai_model_utils.llm_eval.constant import JUDGE_LLMS

from clarifai.client.model import Model
from clarifai.client.dataset import Dataset

model = Model(model_url)
ds = Dataset(ds_url)

evaluator = ClarifaiEvaluator(predictor=model)

out = evaluator.evaluate(
  template="llm_as_judge",
  judge_llm_url=JUDGE_LLMS.GPT3_5_TURBO,
  upload=True,
  dataset=ds,
)
print(out)
```

## Different Dataset Sources
Using Hugging Face 🤗 Datasets is also supported, with column names `question` and `answer`. 

```diff
from clarifai_model_utils import ClarifaiEvaluator
from clarifai_model_utils.llm_eval.constant import JUDGE_LLMS

from clarifai.client.model import Model

+ from datasets import load_dataset
- from clarifai.client.dataset import Dataset

model = Model(model_url)

+ ds = load_dataset("stanfordnlp/coqa", split="train").rename_columns{"questions": "question", "answers": "answer:}
- ds = Dataset(ds_url)

evaluator = ClarifaiEvaluator(predictor=model)

out = evaluator.evaluate(
  template="llm_as_judge",
  judge_llm_url=JUDGE_LLMS.GPT3_5_TURBO,
  upload=True,
  dataset=ds,
)
print(out)
```

## Generating Synthetic Data
Given a dataset of contexts / chunks, questions and answers can be generated using the integration with RAGAS. The dataset can be used directly in the evaluator.

```diff
from clarifai_model_utils import ClarifaiEvaluator
from clarifai_model_utils.llm_eval.constant import JUDGE_LLMS

from clarifai.client.model import Model
from clarifai.client.dataset import Dataset

model = Model(model_url)
ds = Dataset(ds_url)  ## This dataset only has text chunks from source. There are no questions or answers yet. 

evaluator = ClarifaiEvaluator(predictor=model)

out = evaluator.evaluate(
  template="llm_as_judge",
  judge_llm_url=JUDGE_LLMS.GPT3_5_TURBO,
  upload=True,
  dataset=ds,
+  generate_qa=True
)
print(out)

```

## Examples

* [llm-eval example notebook](./examples/llm_eval.ipynb)
