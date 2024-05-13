import logging
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from datasets import Dataset as HFDataset
from langchain_community.embeddings import ClarifaiEmbeddings
from langchain_community.llms import Clarifai
from lm_eval.api.task import TaskConfig
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from ....constant import BGE_BASE_EMBED_MODEL
from .base import _BasePythonTemplate

logger = logging.getLogger("ragas")
logger.disabled = True


@dataclass
class RAGAS(_BasePythonTemplate):

  langchain_llm_kwargs: dict = field(default_factory=lambda: {})
  langchain_llm: Optional[Clarifai] = None
  has_ground_truth: Optional[bool] = True
  embedder: Optional[ClarifaiEmbeddings] = None
  config: TaskConfig = field(default_factory=TaskConfig)

  def __post_init__(self) -> None:

    self.config.task = "ragas"
    self.config.group = "ragas"
    self.config.dataset_path = "csv"
    self.config.dataset_name = None
    self.config.output_type = "generate_until"
    self.config.validation_split = "validation"
    self.config.doc_to_text = "{{question}}"
    self.config.doc_to_target = ""
    self.config.repeats = 1
    self.config.num_fewshot = 0

    self.config.metric_list = [{
        "metric": "faithfulness",
        "aggregation": "mean",
        "higher_is_better": True,
    }, {
        "metric": "answer_relevancy",
        "aggregation": "mean",
        "higher_is_better": True,
    }]

    self.ragas_metrics = [
        faithfulness,
        answer_relevancy,
    ]

    if self.has_ground_truth:
      self.config.metric_list.extend([{
          "metric": "context_precision",
          "aggregation": "mean",
          "higher_is_better": True,
      }, {
          "metric": "context_recall",
          "aggregation": "mean",
          "higher_is_better": True,
      }])
      self.ragas_metrics.extend([context_precision, context_recall])

    self.config.process_results = self.process_results_func

  def process_results_func(self, doc: dict, results: List[List]):
    """Compute RAGAS metrics per row of dataset

    Args:
        doc (dict): row data of dataset
        results (List[List]): result list has length equal to batch size (1) contains [context, answer] of RAG workflow

    Returns:
        _type_: _description_
    """
    assert isinstance(results,
                      list) and len(results[0]) > 1, "results must be a list of [context, answer]"

    pat = self.langchain_llm_kwargs.get("pat", None)
    token = self.langchain_llm_kwargs.get("token", None)
    if self.embedder is None:
      self.embedder = ClarifaiEmbeddings(model_url=BGE_BASE_EMBED_MODEL, pat=pat, token=token)
    self.langchain_llm = Clarifai(**self.langchain_llm_kwargs)
    # context from Clarifai RAG workflow
    # NOTE: discard context in dataset only use context of RAG workflow
    context = results[0][0]
    # answer from Clarifai RAG workflow
    answer = results[0][1]
    # Take value of question
    question = doc["question"]
    try:
      ground_truth = doc["ground_truths"]
    except:
      ground_truth = ""
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[context]],
        "ground_truth": [ground_truth]
    }
    dataset = HFDataset.from_dict(data)

    ragas_results = evaluate(
        dataset=dataset,
        llm=LangchainLLMWrapper(self.langchain_llm),
        embeddings=self.embedder,
        metrics=self.ragas_metrics)

    # FIXME: replace nan value as 0.
    return {k: 0. if math.isnan(v) else v for k, v in ragas_results.items()}

  def to_harness_dict_config(self) -> dict:
    d = asdict(self.config)
    return d
