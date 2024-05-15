from abc import ABC, abstractmethod


class _BasePythonTemplate(ABC):

  @abstractmethod
  def to_harness_dict_config(self) -> dict:
    """convert current config to Harness Eval TaskConfig Dictionary"""
