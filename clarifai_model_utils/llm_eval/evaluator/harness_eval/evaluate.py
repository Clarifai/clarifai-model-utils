import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Union

import lm_eval
import pandas as pd
from lm_eval import evaluator, utils
from lm_eval.api.task import TaskConfig
from lm_eval.tasks import TaskManager, get_task_dict

from clarifai.client.model import Model
from clarifai.client.workflow import Workflow
from clarifai.utils.logging import get_logger

try:
  from .llm import ClarifaiLM  # noqa # pylint: disable=unused-import
except:
  pass

from .output import DatasetInfo, EvaluateResult, LmJudgeInfo, PromptTemplate, make_result_dataframe

logger = get_logger(name=__file__)

HARNESS_EVAL_TASK_MANAGER = TaskManager()


def construct_regex_filter(regex_code: str):
  return [{'function': 'regex', 'regex_pattern': regex_code}, {"function": "take_first"}]


def construct_filters(filters: list, name='get-answer'):
  return {'filter_list': [{'name': name, 'filter': filters}]}


class ClarifaiModelHarnessEval:

  def __init__(self,) -> None:
    self.template_dir = os.path.join(os.path.dirname(__file__), "template")
    templates = os.listdir(self.template_dir)

    self.template_configs = defaultdict(lambda: dict())  # {template_name: {yaml: path}}
    #for template in templates:
    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = {
          executor.submit(self.load_config, os.path.join(self.template_dir, template)): template
          for template in templates
      }
      for job in as_completed(futures):
        template = futures[job]
        configs = job.result()
        self.template_configs.update({template: configs})

  @staticmethod
  def load_metrics_list_from_config(config: dict):
    # load metric
    metrics = []
    for each in config.get("metric_list"):
      n = each.get("metric")
      if callable(n):
        metrics.append(n.__name__)
      else:
        metrics.append(n)

    return metrics

  def load_config(self, template_folder: str) -> list:
    _yaml_file = [
        each for each in os.listdir(template_folder)
        if each.endswith(".yaml") or each.endswith('.yml')
    ]
    assert len(_yaml_file) == 1, f"No yaml file found in {template_folder}"
    template_configs = defaultdict()
    template_configs.update(dict(yaml=os.path.join(template_folder, _yaml_file[0])))
    # load readme
    _readme = [each for each in os.listdir(template_folder) if each.endswith(".md")]
    readme = ""
    if len(_readme) > 0:
      with open(os.path.join(template_folder, _readme[0]), "r") as f:
        readme = f.read()
    config = utils.load_yaml_config(template_configs["yaml"])

    # load metrics
    metrics = self.load_metrics_list_from_config(config)

    template_configs.update(dict(readme=readme))
    template_configs.update(dict(metrics=metrics))
    template_configs.update(dict(config=config))

    return template_configs

  @property
  def templates(self):
    return list(self.template_configs.keys())

  def get_template_desc(self, template):
    assert template in self.templates, f"Support templates `{self.templates}`, got {template}"
    return self.template_configs[template]["readme"]

  def get_metrics(self, template: str) -> list:
    assert template in self.templates, f"Support templates `{self.templates}`, got {template}"
    return self.template_configs[template]["metrics"]

  def call_harness_eval(self,
                        predictor: Union[str, Model, Workflow],
                        task_dict: dict,
                        inference_parameters: dict = {},
                        workflow_output_node: int = 1,
                        is_rag_workflow: bool = None,
                        predictor_kwargs: dict = {}):
    """Harness eval

    Args:
        predictor (Union[str, Model, Workflow])
        task_dict (dict): harness lm eval task dict.
        inference_parameters (dict, optional): Clarifai mode inference_params that put into `predict_by..` function
        custom_config (dict, optional): harness eval config, see yaml file. Defaults to {}.
        workflow_output_node (int, optional): Index of output node in workflow, applicable when using Workflow predictor.
        is_rag_workflow (bool, optional): Evaluate a RAG workflow, applicable when using Workflow predictor.
        predictor_kwargs (dict, optional): kwargs for initializing model class when passing an url.

    Returns:
        dict: with keys summary, configs, samples
    """
    lm = lm_eval.api.registry.get_model("clarifai")(
        predictor=predictor,
        inference_parameters=inference_parameters,
        workflow_output_node=workflow_output_node,
        is_rag_workflow=is_rag_workflow,
        **predictor_kwargs)

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        write_out=True,
        log_samples=True,
    )
    task_name = next(iter(task_dict.keys()))
    summary = {}
    metrics = self.load_metrics_list_from_config(
        task_dict[task_name].config.to_dict(keep_callable=True))
    for k, v in results.get("results")[task_name].items():
      _k = k.split(",")[0]
      if _k in metrics:
        summary.update({_k: v})

    results = dict(
        summary=summary,
        configs=results.get("configs")[task_name],
        samples=results.get("samples")[task_name],
    )

    return results

  def prepare_config(self, template, data_file: str = None):
    """Verify if 'template' is in defined templates or dict

    Args:
        template (Union[str, dict])
        data_file (str, optional): path to dataframe if using dataset_path in config. Defaults to None.

    Returns:
        dict
    """
    config = {}
    if template in self.templates:
      config = deepcopy(self.template_configs[template]["config"])
    elif type(template) == TaskConfig:
      config = template.to_dict(keep_callable=True)
    elif isinstance(template, dict):
      config = template
    elif isinstance(template, str) and os.path.exists(template):
      config = self.load_config(template)['config']
    elif isinstance(template, str) and template in HARNESS_EVAL_TASK_MANAGER.all_tasks:
      config = get_task_dict(task_name_list=template)[template].config.to_dict()
    else:
      raise ValueError("Supported template type as one of [str, dict, Task]")
    if config.get("dataset_path", "") == "csv":
      if config.get('dataset_kwargs', None):
        if config.get('dataset_kwargs').get("data_files", None):
          if config.get('dataset_kwargs').get("data_files").get("validation", None):
            assert data_file, "`data_file` must be provided when using `dataset_path=csv`"
      config['dataset_kwargs'] = dict(data_files=dict(validation=data_file))
    print(config)
    return config

  def evaluate(
      self,
      predictor: Union[Model, Workflow],
      data_frame: pd.DataFrame,
      template: str,
      weights: dict,
      regex_code: str = "",
      input_prompt: str = "",
      judge_llm_url: str = "",
      custom_config: dict = {
          "num_fewshot": 0,
      },
      inference_parameters: dict = {},
      predictor_kwargs: dict = {},
      eval_id: str = None,
      dataset_info: dict = None,
      workflow_output_node: int = 1,
      is_rag_workflow: bool = None,
  ) -> EvaluateResult:
    """Evaluate
    Args:
        predictor (Union[Model, Workflow]): Model/Workflow or Url
        data_frame (pd.DataFrame): a dataframe has column names [question, answer]
        template (str): template name
        weights (dict): weights of sub metrics
        regex_code (str): regex code that makes `filters`
        input_prompt (str): input prompt tempplate that makes `doc_to_text`
        judge_llm_url (str): Clarifai model url for `process_results`
        custom_config (dict, optional): a config type of dict of harness eval, see the yaml file or https://github.com/EleutherAI/lm-evaluation-harness/blob/ae74b808e43cd1ee6d88a157777f27eacd6b12dc/lm_eval/api/task.py#L52 for key value pair format. Defaults to {}.
        inference_params (dict, optional): LLM model inference params. Defaults to {}.
        eval_id (str, optional): custom eval id
        dataset_info (dict,): a dict has keys: {id, app_id, version_id, user_id}
        workflow_output_node (int, optional): Index of output node in workflow, applicable when using Workflow predictor. Ignore when evaluate RAG
        is_rag_workflow (bool, optional): Evaluate a RAG workflow, applicable when using Workflow predictor.
    Returns:
        EvaluateResult

    """
    _file = tempfile.NamedTemporaryFile(prefix="lm_eval_", suffix=".csv")
    _file.close()
    try:
      data_frame.to_csv(_file.name, index=False)
      _template = deepcopy(template)

      config = self.prepare_config(_template, _file.name)
      config.update(custom_config)
      template_name = config.get("task", None) or template
      # checking weights config
      if weights:
        _loaded_metrics = self.load_metrics_list_from_config(config)
        assert all([each in _loaded_metrics for each in weights]), Exception(
            f"Defined metrics {weights} in `weights` are not same as template metrics {_loaded_metrics}"
        )

      logger.debug(config)
      if regex_code:
        filters = construct_filters(construct_regex_filter(eval(regex_code)))
        config.update(filters)

      judge_model = None
      if template_name in ['llm_as_judge', 'rag']:
        assert judge_llm_url, ValueError(
            f"Please provide judge_llm_url for template llm_as_judge or rag")
        judge_model = LmJudgeInfo(
            url=judge_llm_url, pat=predictor.auth_helper._pat, token=predictor.auth_helper._token)
        logger.debug(judge_model)
        if 'rag' in template_name or is_rag_workflow:
          assert isinstance(predictor, Workflow), "Require Workflow predictor to evaluate RAG"
          config.update(dict(process_results=judge_model.judge.process_rag_result))
          is_rag_workflow = True
        elif template_name == 'llm_as_judge':
          config.update(dict(process_results=judge_model.judge_process_result_func))

      if dataset_info and isinstance(dataset_info, dict):
        dataset_info = DatasetInfo(**dataset_info)

      prompter = None
      if input_prompt:
        prompter = PromptTemplate(harness_prompt=input_prompt)
        config.update(dict(doc_to_text=input_prompt))

      task_dict = get_task_dict([config], task_manager=HARNESS_EVAL_TASK_MANAGER)
      results = self.call_harness_eval(
          predictor=predictor,
          predictor_kwargs=predictor_kwargs,
          task_dict=task_dict,
          inference_parameters=inference_parameters,
          workflow_output_node=workflow_output_node,
          is_rag_workflow=is_rag_workflow,
      )
      results = make_result_dataframe(results=results, weights=weights)
      results.prompter = prompter
      results.llm_judge_info = judge_model
      results.regex_code = regex_code
      results.inference_params = inference_parameters
      results.template = template_name
      if dataset_info:
        results.dataset = dataset_info
      if eval_id:
        results.id = eval_id

    finally:
      if os.path.exists(_file.name):
        os.remove(_file.name)

    return results
