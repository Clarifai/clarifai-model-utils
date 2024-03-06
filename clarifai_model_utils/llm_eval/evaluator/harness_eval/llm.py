import time
from typing import List, Tuple, Union

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from clarifai.client.model import Model
from clarifai.client.workflow import Workflow

from ...constant import MODEL


def clarifailm_completion(_self, prompt, inference_params=None, **kwargs):
  """Query Clarifai API for completion.

    Retry with back-off until they respond
    """

  backoff_time = 5
  while True:
    try:
      prompt = bytes(prompt, 'utf-8')
      # get final output of workflow/model
      if _self.is_model:
        response = _self.client.predict_by_bytes(
              input_bytes=prompt, input_type="text",
            inference_params=_self.inference_parameters).outputs[-1].data.text.raw
      else:
        if _self.is_rag_workflow:
          output = _self.client.predict_by_bytes(input_bytes=prompt, input_type="text")
          query = output.results[0].outputs[0].data.text.raw
          _response = output.results[0].outputs[1].data.text.raw
          response = [query, _response]
        else:
          response = _self.client.predict_by_bytes(
              input_bytes=prompt,
              input_type="text").results[0].outputs[_self.workflow_output_node].data.text.raw
      return response

    except RuntimeError:
      import traceback

      traceback.print_exc()
      time.sleep(backoff_time)
      backoff_time *= 1.5


@register_model("clarifai")
class ClarifaiLM(LM):
  REQ_CHUNK_SIZE = 20

  def __init__(self,
               predictor: Union[Model, Workflow, str],
               type: str = None,
               inference_parameters: dict = {},
               workflow_output_node: int = 1,
               is_rag_workflow: bool = None,
               **kwargs):
    """
    """
    super().__init__()
    if isinstance(predictor, str):
      if type == MODEL:
        self.client = Model(**kwargs)
      else:
        self.client = Workflow(**kwargs)
    else:
      self.client = predictor

    self.inference_parameters = inference_parameters
    self.kwargs = kwargs
    self._is_rag_workflow = is_rag_workflow
    self._workflow_output_node = workflow_output_node

    if self._is_rag_workflow:
      assert isinstance(self.client,
                        Workflow), f"is_rag_workflow is True but predictor is not Workflow"

  @property
  def is_rag_workflow(self):
    return self._is_rag_workflow

  @property
  def workflow_output_node(self):
    return self._workflow_output_node

  @property
  def is_model(self):
    return isinstance(self.client, Model)

  @property
  def eot_token_id(self):
    raise NotImplementedError("Not implemented clarifai tokenization.")

  @property
  def max_length(self):
    return 2048

  @property
  def max_gen_toks(self):
    return 256

  @property
  def batch_size(self):
    # Isn't used because we override _loglikelihood_tokens
    raise NotImplementedError()

  @property
  def device(self):
    # Isn't used because we override _loglikelihood_tokens
    raise NotImplementedError()

  def tok_encode(self, string: str):
    raise NotImplementedError("Not implemented clarifai tokenization.")

  def tok_decode(self, tokens):
    raise NotImplementedError("Not implemented clarifai tokenization.")

  def _loglikelihood_tokens(self, requests, disable_tqdm=False):
    raise NotImplementedError("No support for logits.")

  def generate_until(self, requests):
    if not requests:
      return []

    _requests: List[Tuple[str, dict]] = [req.args for req in requests]

    res = []
    for request in tqdm(_requests):
      inp = request[0]
      request[1]
      #until = request_args["until"]
      response = clarifailm_completion(_self=self, prompt=inp)

      res.append(response)
    return res

  def _model_call(self, inps):
    # Isn't used because we override _loglikelihood_tokens
    raise NotImplementedError()

  def _model_generate(self, context, max_length, eos_token_id):
    # Isn't used because we override greedy_until
    raise NotImplementedError()

  def loglikelihood(self, requests):
    raise NotImplementedError("No support for logits.")

  def loglikelihood_rolling(self, requests):
    raise NotImplementedError("No support for logits.")
