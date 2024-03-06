import os

from langchain.evaluation import load_evaluator
from langchain.llms import Clarifai

from clarifai.urls.helper import ClarifaiUrlHelper


class Judge:
  """ Implement LLM as Judge class"""

  def __init__(self, url, pat=None, eval_rag_workflow: bool = None):
    user_id, app_id, _, model_id, _ = ClarifaiUrlHelper.split_clarifai_url(url)
    CLARIFAI_PAT = os.environ.get("CLARIFAI_PAT", pat)
    self.judge = Clarifai(pat=CLARIFAI_PAT, user_id=user_id, app_id=app_id, model_id=model_id)
    llm_metric_list = ["relevance", "depth", "creativity", "correctness", "helpfulness"]
    llm_metric_list.sort()

    self.llm_metrics = {
        metric: load_evaluator("labeled_criteria", llm=self.judge, criteria=metric)
        for metric in llm_metric_list
    }

    self.rag_metrics = {
        metric: load_evaluator("criteria", llm=self.judge, criteria=metric)
        for metric in llm_metric_list if metric != "correctness"
    }

  def process_results(self, doc, results):
    """ LLM as judge evaluation processor

    Args:
      doc: a list of dicts or a hf dataset row, where keys are column names of dataframe
      results: list of predictions
    """
    prediction = results[0]
    # Take value of `question` and `answer`
    question, answer = doc["question"], doc["answer"]
    results = {
        metric: executor.evaluate_strings(input=question, prediction=prediction,
                                          reference=answer)['score']
        for metric, executor in self.llm_metrics.items()
    }
    for m in results:
      if results[m] is None:
        results[m] = 0

    return results

  def process_rag_result(self, doc, results):
    query = results[0][0]
    response = results[0][1]
    # Take value of `question
    doc["question"]
    results = dict()
    for metric, executor in self.rag_metrics.items():
      if metric != "correctness":
        results.update({
            metric: executor.evaluate_strings(input=query, prediction=response)['score']
        })

    for m in results:
      if results[m] is None:
        results[m] = 0

    return results
