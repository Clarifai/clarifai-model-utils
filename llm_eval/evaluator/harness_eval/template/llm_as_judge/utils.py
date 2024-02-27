# This is a dummy file, the idea is to get how to implement custom processing functions in harness-llm
# See ../judge_eval.py to get code of llm-as-judge
# Again, the code in this file won't be loaded but ../judge_eval.py
import os

from langchain.evaluation import load_evaluator
from langchain.llms import Clarifai

CLARIFAI_PAT = os.environ.get("CLARIFAI_PAT")

judge = Clarifai(pat=CLARIFAI_PAT, user_id="meta", app_id="Llama-2", model_id="llama2-70b-chat")

llm_metric_list = ["relevance", "depth", "creativity", "correctness", "helpfulness"]

llm_metric_list.sort()
llm_metrics = {
    metric: load_evaluator("labeled_criteria", llm=judge, criteria=metric)
    for metric in llm_metric_list
}


def process_results(doc, results):

  prediction = results[0]
  question, answer = doc["question"], doc["answer"]
  results = {
      metric: executor.evaluate_strings(input=question, prediction=prediction,
                                        reference=answer)['score']
      for metric, executor in llm_metrics.items()
  }
  for m in results:
    if results[m] is None:
      results[m] = 0
  return results
