import os

import pandas as pd

from clarifai.client import Workflow
from clarifai_model_utils import ClarifaiEvaluator
from clarifai_model_utils.llm_eval.constant import JUDGE_LLMS

# set PAT
os.environ["CLARIFAI_PAT"] = "54fefdef9b344d269633cad0d0e0b0d9"

# Load Clarifai RAG workflow
wf = Workflow(url= ...)

evaluator = ClarifaiEvaluator(predictor=wf, is_rag_workflow=True)

# Create a dummy dataset
df = [dict(question="What is WC 2022"), dict(question="Who won the title?")]
df = pd.DataFrame(df)

# Run evaluate
out = evaluator.evaluate(
    template="ragas",
    upload=False,
    judge_llm_url=JUDGE_LLMS.GPT3_5_TURBO,  # use GPT3.5 in RAGAS
    dataset=df,
)

print(out.df_to_pandas())
