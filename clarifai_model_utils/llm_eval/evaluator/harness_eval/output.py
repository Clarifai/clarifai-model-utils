from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, computed_field, validator

from .judge_llm import Judge
from .utils import compute_overall, convert_to_datetime


class LmJudgeInfo(BaseModel):
  url: str
  model_id: Optional[str] = ""
  pat: Optional[str] = Field(default="", exclude=True)
  token: Optional[str] = Field(default="", exclude=True)
  judge: Judge = Field(default=None, exclude=True)

  class Config:
    arbitrary_types_allowed = True

  def model_post_init(self, __context) -> None:
    self.judge = Judge(self.url, self.pat, self.token)
    if self.model_id == "":
      self.model_id = self.judge.judge.model_id

  @property
  def judge_process_result_func(self) -> callable:
    return self.judge.process_results


class PromptTemplate(BaseModel):
  harness_prompt: str = ''

  @validator("harness_prompt")
  def prompt_must_contain_question(cls, v):
    if not v:
      return v
    if "{{question}}" not in v:
      raise ValueError("{{question}} must be in prompt")
    return v

  @computed_field
  @property
  def clarifai_prompt(self) -> str:
    if self.harness_prompt:
      return self.harness_prompt.replace("{{question}}", "{data.text.raw}")
    else:
      return ""

  def harness_prompt_to_text(self, text: str):
    if self.harness_prompt:
      return self.harness_prompt.replace("{{question}}", text)
    return ""


class DatasetInfo(BaseModel):
  id: str
  app_id: str
  user_id: Optional[str] = Field(default="")
  version_id: Optional[str] = Field(default="")


class EvaluateResult(BaseModel, validate_assignment=True):
  id: str = Field(default="")
  template: str = Field(default="")
  dataset: Optional[DatasetInfo] = Field(default=None)
  df: dict = Field(default={})
  summary: dict = Field(default={})
  weights: dict = Field(default={})
  llm_judge_info: Optional[LmJudgeInfo] = Field(default=None)
  inference_params: dict = Field(default={})
  prompter: Optional[PromptTemplate] = Field(default=None)
  regex_code: Optional[str] = Field(default="")
  timestamp: int = Field(default=None, exclude=True)
  meta: dict = Field(default={})

  @validator("df")
  def df_must_have_columns(cls, v):
    if not v:
      return v
    cols = v.keys()
    main_cols = ["prediction", "filtered_prediction"]
    if not all([True if each in cols else False for each in main_cols]):
      raise ValueError(f"df must contain these columns {main_cols}")
    return v

  @property
  def normalized_weights(self):
    sum_weights = sum(self.weights.values())
    metric_weights = {k: v / sum_weights for k, v in self.weights.items()}
    return metric_weights

  @computed_field
  @property
  def average(self) -> float:
    if self.weights and all([each in self.summary for each in self.weights]):
      return compute_overall(self.summary, self.normalized_weights)
    else:
      return sum(self.summary.values()) / len(self.summary) if len(self.summary) else 0.

  @property
  def eval_datetime(self) -> Union[str, None]:
    if self.timestamp:
      return convert_to_datetime(self.timestamp)
    else:
      return None

  def df_to_pandas(self):
    return pd.DataFrame(self.df)

  def __setattr__(self, name, value):

    if name == "weights":
      if self.meta.get("other_prompts", None):
        current_best = deepcopy(self)
        input_df = deepcopy(self.df)
        metrics = list(current_best.summary.keys())
        pred_cols = ["prediction", "filtered_prediction"]
        output_cols = metrics + pred_cols
        # make input df aka quesiton,answer cols
        _ = [input_df.pop(col, None) for col in output_cols]
        # keep output cols of best
        _ = [current_best.df.pop(col, None) for col in input_df.keys()]
        # remove meta
        current_best.meta = {}
        # assign weights to new
        others = deepcopy(self.meta.get("other_prompts"))
        current_best.__dict__["weights"] = value
        for each in others:
          each.__dict__["weights"] = value

        new_best = merge_prompt_eval_result([current_best] + others)
        new_best.df.update(input_df)

        _vars = [each for each in new_best.__dict__.keys()]
        for _var in _vars:
          self.__dict__[_var] = new_best.__dict__[_var]

    super().__setattr__(name, value)


def merge_prompt_eval_result(data: List[EvaluateResult], higher_is_better=True):
  eval_data = deepcopy(data)

  # get max overall score
  overall_scores = [each.average for each in eval_data]
  #
  if higher_is_better:
    best_prompt = eval_data.pop(overall_scores.index(max(overall_scores)))
  else:
    best_prompt = eval_data.pop(overall_scores.index(min(overall_scores)))
  #
  metrics = list(best_prompt.summary.keys())
  pred_cols = ["prediction", "filtered_prediction"]
  keep_cols = metrics + pred_cols
  removed_cols = set(best_prompt.df.keys()) - set(keep_cols)
  #
  for each in eval_data:
    _ = [each.df.pop(col, None) for col in removed_cols]

  if higher_is_better:
    eval_data = sorted(eval_data, key=lambda x: x.average, reverse=True)
  else:
    eval_data = sorted(eval_data, key=lambda x: x.average)

  best_prompt.meta = dict(other_prompts=eval_data)

  return best_prompt


def convert_dict_to_eval_result(data: dict, **kwargs) -> EvaluateResult:
  prompter = data.pop("prompter", None)
  llm_judge_info = data.pop("llm_judge_info", None)
  dataset = data.pop("dataset", None)
  other_prompts = data.pop("meta", {}).get("other_prompts", {})
  _metrics_info = {
      "llm_judge_info": LmJudgeInfo(**llm_judge_info) if llm_judge_info else None,
      "prompter": PromptTemplate(**prompter) if prompter else None,
      "dataset": DatasetInfo(**dataset) if dataset else None,
  }
  if other_prompts:
    others = []
    for each in other_prompts:
      others.append(convert_dict_to_eval_result(each))
    data.update({"meta": {"other_prompts": others}})

  data.update(**kwargs)

  return EvaluateResult(**data, **_metrics_info)


def merge_prompt_eval_result(data: List[EvaluateResult], higher_is_better=True):
  eval_data = deepcopy(data)

  # get max overall score
  overall_scores = [each.average for each in eval_data]
  #
  if higher_is_better:
    best_prompt = eval_data.pop(overall_scores.index(max(overall_scores)))
  else:
    best_prompt = eval_data.pop(overall_scores.index(min(overall_scores)))
  #
  metrics = list(best_prompt.summary.keys())
  pred_cols = ["prediction", "filtered_prediction"]
  keep_cols = metrics + pred_cols
  removed_cols = set(best_prompt.df.keys()) - set(keep_cols)
  #
  for each in eval_data:
    _ = [each.df.pop(col, None) for col in removed_cols]

  if higher_is_better:
    eval_data = sorted(eval_data, key=lambda x: x.average, reverse=True)
  else:
    eval_data = sorted(eval_data, key=lambda x: x.average)

  best_prompt.meta = dict(other_prompts=eval_data)

  return best_prompt


def make_result_dataframe(results: dict, weights: dict) -> EvaluateResult:
  """Make result to render and persist

  Args:
      results (dict): evaluation result from BE.post_eval
      weights (dict): metric weights

  Returns:
      dict: {df: pd.DataFrame, overall_score: int, summary: list, weights: weights}
  """
  summary = results['summary']
  all_metrics_names = list(summary.keys())
  # selecting metrics
  all_short_metric_names = []
  # Find principal metrics, ignore stderr
  for metric in all_metrics_names:
    metric.split("_")
    if not metric.endswith("stderr"):
      all_short_metric_names.append(metric)
  all_short_metric_names = set(all_short_metric_names)
  short_list_summary = {k: v for k, v in summary.items() if k in all_short_metric_names}

  # make dataframe
  df = defaultdict(lambda: [])
  for sample in results["samples"]:
    # get dataset
    for col, value in sample["doc"].items():
      if not "unnamed:" in col.lower():
        df[col].append(value)
    # get output
    df["prediction"].append(sample["prediction"]
                            if len(sample["resps"]) > 1 else sample["resps"][0][0])
    df["filtered_prediction"].append(sample["filtered_resps"] if len(sample["filtered_resps"]) > 1
                                     else sample["filtered_resps"][0])
    # make metric columns
    for metric in short_list_summary:
      if metric.endswith("stderr"):
        continue
      if metric in sample:
        df[metric].append(float(sample[metric]))
      else:
        df[metric].append(None)

  return EvaluateResult(df=df, summary=short_list_summary, weights=weights)
