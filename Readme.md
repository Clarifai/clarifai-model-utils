![Clarifai logo](docs/logo.png)

# Clarifai Python Data Utils


[![Discord](https://img.shields.io/discord/1145701543228735582)](https://discord.gg/M32V7a7a)
[![codecov](https://img.shields.io/pypi/dm/clarifai)](https://pypi.org/project/clarifai-datautils)

This is a collection of utilities for handling various types of multimedia data. Enhance your experience by seamlessly integrating these utilities with the Clarifai Python SDK. This powerful combination empowers you to address both visual and textual use cases effortlessly through the capabilities of Artificial Intelligence. Unlock new possibilities and elevate your projects with the synergy of versatile data utilities and the robust features offered by the [Clarifai Python SDK](https://github.com/Clarifai/clarifai-python). Explore the fusion of these tools to amplify the intelligence in your applications! 🌐🚀

[Website](https://www.clarifai.com/) | [Schedule Demo](https://www.clarifai.com/company/schedule-demo) | [Signup for a Free Account](https://clarifai.com/signup) | [API Docs](https://docs.clarifai.com/) | [Clarifai Community](https://clarifai.com/explore) | [Python SDK Docs](https://docs.clarifai.com/python-sdk/api-reference) | [Examples](https://github.com/Clarifai/examples) | [Colab Notebooks](https://github.com/Clarifai/colab-notebooks) | [Discord](https://discord.gg/XAPE3Vtg)

---
## Table Of Contents

* **[Installation](#installation)**
* **[Getting Started](#getting-started)**
* **[Features](#features)**
  * [Image Utils](#image-utils)
* **[Usage](#usage)**
* **[Examples](#more-examples)**


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
