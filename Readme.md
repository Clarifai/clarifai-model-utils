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
pip3 install -r requirements.txt
```

## Getting started

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

## Examples

* [llm-eval example notebook](./examples/llm_eval.ipynb)
