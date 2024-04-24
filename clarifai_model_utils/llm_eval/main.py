import datetime
import os
from collections import defaultdict
from typing import Tuple, TypeVar, Union

import pandas as pd
from google.protobuf.json_format import MessageToDict
from datasets import Dataset as HFDataset
import ragas

from clarifai.client.dataset import Dataset
from clarifai.client.model import Model
from clarifai.client.workflow import Workflow
from clarifai.utils.logging import get_logger

from .constant import BGE_BASE_EMBED_MDOEL, WORKFLOW, JUDGE_LLMS
from .evaluator import ClarifaiModelHarnessEval, EvaluateResult, convert_dict_to_eval_result
from .utils import get_timestamp, make_dataset

logger = get_logger(name=__file__)

PREDICTOR_TYPES = TypeVar('PREDICTOR_TYPES', str, Model, Workflow)


class ClarifaiEvaluator():
  """Clarifai LLM evaluattor

  Args:
      predictor (PREDICTOR_TYPES): Accept instance of Model/Workflow or url with additional 'predictor_kwargs'
      type (str, optional): Type is 'workflow' or 'model', applicable if pass 'url' to predictor. Defaults to None.
      inference_parameters (dict, optional): Model inference parameters. Defaults to {}.
      workflow_output_node (int, optional): Output node of Workflow, applicable for Workflow predictor. Defaults to 1. Ignore when evaluate RAG.
      is_rag_workflow (bool, optional): Evaluate a RAG workflow, applicable when using Workflow predictor.

  Example:
  >>> from clarifai.client.model import Model
  >>> model = Model(model_url)
  >>> evaluator = ClarifaiEvaluator(predictor=model)

  """

  def __init__(self,
               predictor: PREDICTOR_TYPES,
               type: str = None,
               inference_parameters: dict = {},
               workflow_output_node: int = 1,
               is_rag_workflow: bool = None,
               refresh_enabled: bool = True,
               **predictor_kwargs):
    if isinstance(predictor, str):
      _pred_clss = Workflow if type == WORKFLOW else Model
      predictor = _pred_clss(url=predictor, **predictor_kwargs)

    self.predictor = predictor
    self.evaluator = ClarifaiModelHarnessEval()
    self.inference_parameters = inference_parameters
    self._data = dict()
    self.refresh_enabled = refresh_enabled
    if refresh_enabled:
      self.refresh()
    self._is_rag_workflow = is_rag_workflow
    self._workflow_output_node = workflow_output_node

  @classmethod
  def with_rag_workflow(cls, predictor: Union[str, Workflow], **kwargs):
    return cls(predictor=predictor, type=WORKFLOW, is_rag_workflow=True, **kwargs)

  def get_metric_name(self, template: str) -> list:
    """
    Get metric names of predefined templates

    Args:
        template (str): template name

    Returns:
        list: list of templates
    """
    return self.evaluator.get_metrics(template)

  @property
  def predefined_templates(self) -> list:
    """ Get all predefined templates"""
    return self.evaluator.templates

  def get_template_desc(self, template: str) -> str:
    """Get template description (readme)

    Args:
        template (str): template name

    Returns:
        str: desc
    """
    return self.evaluator.get_template_desc(template)

  @property
  def is_model(self):
    return isinstance(self.predictor, Model)

  @property
  def data(self) -> dict:
    if not self.is_model:
      logger.warning(
          "Workflow predictor does not have history of evaluation. Return empty or None")
      return {}

    return self._data

  @property
  def ts_to_eval_id(self):
    return self._ts_to_id

  @staticmethod
  def _make_eval_id(template, dataset_app_id, dataset_id,
                    with_ts: bool = True) -> Union[Tuple[str, str], str]:
    """ Return: (id, timestamp) if with_ts otherwise id"""
    if with_ts:
      _ts = get_timestamp()
      return f"{template},{dataset_app_id},{dataset_id},{_ts}", _ts
    else:
      return f"{template},{dataset_app_id},{dataset_id}"

  @staticmethod
  def parse_eval_id(_id) -> list:
    """
    Returns template, dataset_app_id, dataset_id, ts
    """
    data = _id.split(",")
    return data if len(data) == 4 else []

  def refresh(self):
    """
    Run `list_evaluations` method in order to update `data` attribute
    """
    if not self.is_model:
      logger.warning("Not support `refresh` method for workflow predictor. Return empty value")
      return {}

    logger.info("Refresh eval metrics")
    response = self.predictor.list_evaluations()
    results = defaultdict(lambda: {})
    _ts_to_id = defaultdict(lambda: {})
    for each in response:
      ext_metrics_dict = MessageToDict(each.extended_metrics.user_metrics)

      def __parse_id(_id):
        _parsed_data = self.parse_eval_id(_id)
        if len(_parsed_data) == 4:
          ts = int(_parsed_data[3])  # timestamp
          # to template,dataset_app,dataset_id
          _id = ",".join(_parsed_data[:-1])
          return _id, ts
        return None

      eval_result = convert_dict_to_eval_result(ext_metrics_dict)
      _parsed_id_ts = __parse_id(each.id)  # id of eval_metrics
      if _parsed_id_ts:
        _id, ts = _parsed_id_ts
        eval_result.timestamp = ts
        if not eval_result.id:
          eval_result.id = each.id
      elif eval_result.id:
        _id, ts = __parse_id(eval_result.id)  # id of eval_result
      else:
        logger.warning(f"eval id {each.id} doen't have right format.")
        continue
      results[_id].update({ts: eval_result})
      _ts_to_id[ts] = _id
    self._data = results
    self._ts_to_id = _ts_to_id

  def is_in_evaluated_ids(self, _id):
    """ Check if id in evalated ids """
    return _id in self.get_eval_ids()

  def get_eval_ids(self):
    """ Get all eval ids of predictor """
    return list(self.data.keys())

  def get_timestamps_of_eval_ids(self, eval_id: str):
    """
    Get timestamps of specific eval id

    Args:
        eval_id (str): format "template,app,dataset"

    Returns:
        str: timestamp
    """
    return list(self.data[eval_id].keys())

  def get_eval_result_of_eval_id(self, eval_id: str, ts: str) -> EvaluateResult:
    """
    Args:
        eval_id (str): format "template,app,dataset"
        ts (str): timestamp

    Returns:
        EvaluateResult
    """
    return self.data[eval_id][ts]

  def get_latest_eval_result_of_eval_id(self,
                                        eval_id=None,
                                        template: str = None,
                                        app_id: str = None,
                                        dataset_id: str = None) -> EvaluateResult:
    """
    Get latest eval result by eval_id

    Args:
        eval_id (str): format "template,app_id,dataset"
    Returns:
        EvaluateResult
    """

    if not eval_id:
      assert template and app_id and dataset_id, ValueError(
          f"Expected setting `template`, `app_id`, `datset_id` when not using `eval_id`")
      eval_id = self._make_eval_id(
          template=template, dataset_app_id=app_id, dataset_id=dataset_id, with_ts=False)

    ts = max(self.get_timestamps_of_eval_ids(eval_id))
    return self.get_eval_result_of_eval_id(eval_id, ts)

  def get_latest_eval(self) -> EvaluateResult:
    """
    Get latest eval result

    Returns:
        EvaluateResult
    """
    ts = max(self.ts_to_eval_id.keys())
    _id = self.ts_to_eval_id[ts]
    return self.data[_id][ts]

  @staticmethod
  def convert_to_timestamp(datetime_str):
    # Convert the datetime string back to a datetime object
    parsed_datetime = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    # Convert the datetime object back to a Unix timestamp
    parsed_timestamp = int(parsed_datetime.timestamp())

    return parsed_timestamp

  def _get_template_name(self, template: Union[str, dict]) -> str:
    if isinstance(template, dict):
      name = template.get("task", None)
      assert name, Exception("'task' is not set for the config")
      return name
    elif os.path.exists(template):
      cfg = self.evaluator.load_config(template)
      return self._get_template_name(cfg)
    elif isinstance(template, str):
      return template
    else:
      raise ValueError("Not supported template type")

  def upload_result(self, result: EvaluateResult):
    if not self.is_model:
      raise Exception("Not support `upload_result` for Workflow predictor")
    else:
      logger.info("Uploading result...")
      eval_id = result.id
      assert eval_id or len(
          eval_id.split(",")
      ) == 4, "Invalid eval id, expected to have format {template},{dataset_app_id},{dataset_id},{timestamp}, got " + eval_id
      self.predictor.evaluate(
          dataset_id='', extended_metrics=result.model_dump(mode='python'), eval_id=eval_id)

  def evaluate(self,
               template: str,
               dataset: Union[Dataset, pd.DataFrame, HFDataset],
               upload: bool = False,
               inference_parameters: dict = {},
               weights: dict = {},
               regex_code: str = "",
               input_prompt: str = "",
               judge_llm_url: str = "",
               extra_harness_config: dict = {
                   "num_fewshot": 0,
               },
               split_word: str = "### Response:",
               generate_qa: int = None,
               **kwargs) -> EvaluateResult:
    """Evaluating llm model

    Args:
        template (str): name of defined template or path to folder contains harness config/yaml
        dataset (Union[Dataset, pd.DataFrame, HFDataset]): Dataset to evaluate
        upload (bool, optional): Upload result to the platform. Defaults to False.
        inference_parameters (dict, optional): inference parameters. Defaults to {}.
        weights (dict, optional): Normalized (to 1) weights of metrics to compute average score. Defaults to {}.
        regex_code (str, optional): Python regex code to filter model prediction. Defaults to "".
        input_prompt (str, optional): Harness prompt template. Defaults to "".
        judge_llm_url (str, optional): Url of judge model, required when using llm_as_judge template. Defaults to "".
        extra_harness_config (dict, optional): Other custom harness config. Defaults to { "num_fewshot": 0, }.
        split_word (str, optional): Split word for non-jsonify dataset. Defaults to "### Response:".
        generate_qa (int, optional): How many questions and answers to generate from the provided dataset. The dataset is expected to be contexts.

    Returns:
        EvaluateResult
    """
    if isinstance(dataset, Dataset):
      dataset_id = dataset.id
      app_id = dataset.app_id
      version_id = dataset.version.id
      user_id = dataset.user_id
      df = make_dataset(
          auth=dataset.auth_helper,
          dataset_id=dataset.id,
          app_id=dataset.app_id,
          split_word=split_word,
          generate_qa=generate_qa)
    elif isinstance(dataset, pd.DataFrame):  # local data
      df = dataset
      dataset_id = kwargs.pop("dataset_id", "")
      app_id = kwargs.pop("app_id", "")
      user_id = kwargs.pop("user_id", "")
      version_id = kwargs.pop("version_id", "")
      assert dataset_id or app_id, ValueError(
          f"`dataset_id` or `app_id` is empty when using local dataset. Please pass them to kwargs"
      )
    elif isinstance(dataset, HFDataset):
      df = dataset.to_pandas(batched=True)
    else:
      raise Exception("Only Dataset, pd.DataFrame, and HFDataset types are handled.")
    
    if generate_qa:
      from ragas.testset.generator import TestsetGenerator
      from ragas.testset.evolutions import simple, reasoning, multi_context
      from langchain_community.llms import Clarifai
      from langchain_community.embeddings import ClarifaiEmbeddings
      from langchain_community.document_loaders import DataFrameLoader

      ## Initialize models.
      llm = Clarifai(model_url=JUDGE_LLMS.GPT4)
      embeddings = ClarifaiEmbeddings(model_url=BGE_BASE_EMBED_MDOEL)
      generator = TestsetGenerator.from_langchain(
        llm,
        llm,
        embeddings
      )

      # Convert dataframe to langchain docs.
      loader = DataFrameLoader(df)
      documents = loader.load()

      generated_test_set = generator.generate_with_langchain_docs(
        documents, 
        test_size=generate_qa, 
        distributions={
            simple: 0.5, 
            reasoning: 0.25, 
            multi_context: 0.25})

      df = generated_test_set.to_pandas()
      df.to_csv("generated_qa_test_set.csv", index=False)

      # Get answers from predictor
      pass

    import pdb; pdb.set_trace()
    logger.info("Start evaluating...")
    output = self.evaluator.evaluate(
        self.predictor,
        data_frame=df,
        weights=weights,
        template=template,
        regex_code=regex_code,
        input_prompt=input_prompt,
        judge_llm_url=judge_llm_url,
        custom_config=extra_harness_config,
        inference_parameters=inference_parameters or self.inference_parameters,
        dataset_info=dict(id=dataset_id, app_id=app_id, user_id=user_id, version_id=version_id),
        workflow_output_node=self._workflow_output_node,
        is_rag_workflow=self._is_rag_workflow,
    )
    logger.info("Evaluated!")

    eval_id, timestamp = self._make_eval_id(
        template=output.template, dataset_app_id=app_id, dataset_id=dataset_id, with_ts=True)
    output.timestamp = timestamp
    output.id = eval_id

    if upload and isinstance(self.predictor, Model):
      self.upload_result(output)

    if self.refresh_enabled:
      self.refresh()

    return output
