import os

from custom_template import config

from clarifai.client.dataset import Dataset
from clarifai.client.model import Model
from clarifai_model_utils import ClarifaiEvaluator

os.environ["CLARIFAI_PAT"] = "xxx"

model = Model(
    url=
    "https://clarifai.com/phatvo/lm_eval/models/dummy_text/model_version_id/c630093c80104d2582a2b76132efd64a"
)

ds = Dataset(url="https://clarifai.com/phatvo/lm_eval/datasets/alpaca-eval-5")

evaluator = ClarifaiEvaluator(predictor=model)

out = evaluator.evaluate(
    template=config.to_dict(),
    upload=False,
    dataset=ds,
)
print(out.df_to_pandas())
print(out.dataset)
print(out.template)
