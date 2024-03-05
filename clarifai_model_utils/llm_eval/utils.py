import calendar
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.service_pb2_grpc import V2Stub
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import struct_pb2

from clarifai.utils.logging import get_logger
from clarifai.client.app import App
from clarifai.client.input import Inputs

logger = get_logger(name='clarifai_llm_eval-' + __file__)


def get_timestamp():
  gmt = time.gmtime()
  ts = calendar.timegm(gmt)
  return ts


def split_sample_general_template(text, split_word) -> tuple:
  index = text.find(split_word) + len(split_word)
  question = text[:index]
  answer = text[index:]

  return question, answer


def get_text_dataset_inputs(auth, user_id: str, app_id: str, dataset_id: str, max_input=100):
  stub: V2Stub = auth.get_stub()
  user_app_id = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

  # get number of samples of dataset
  app = App(app_id=app_id, user_id=user_id)
  dataset = app.dataset(dataset_id=dataset_id)

  max_per_page = 128
  total_samples = dataset.version.metrics['/'].inputs_count.value
  if not max_input:
    per_page = max_per_page
    chunks = total_samples // per_page
  else:
    chunks = max_input // max_per_page
    if chunks == 0:
      per_page = max_input
    else:
      per_page = max_per_page

  input_obj = Inputs(user_id=user_id, app_id=app_id, pat=auth._pat)
  
  urls = []
  for page in range(chunks + 1):
    all_inputs = list(input_obj.list_inputs(input_type='text', dataset_id=dataset_id, per_page=per_page, page_no=page))
    _urls = [item.input.data.text.url for item in all_inputs.dataset_inputs]
    if len(_urls) < 1:
      break
    urls += _urls

  def download_text(url):
    metadata = auth.metadata
    headers = dict(metadata)
    response = requests.request("GET", url, headers=headers, data="")
    if response.status_code == 200:
      return response.text
    else:
      return None

  texts = []
  with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(download_text, url) for url in urls]
    for job in futures:
      texts.append(job.result())

  return True, texts


def post_ext_metrics_eval(auth, model_id, version_id, eval_id, ext_metrics):
  metrics = struct_pb2.Struct()
  metrics.update(ext_metrics)
  metrics = resources_pb2.ExtendedMetrics(user_metrics=metrics)

  stub = auth.get_stub()
  user_app_id = resources_pb2.UserAppIDSet(user_id=auth.user_id, app_id=auth.app_id)
  post_eval = stub.PostEvaluations(
      service_pb2.PostEvaluationsRequest(
          user_app_id=user_app_id,
          eval_metrics=[
              resources_pb2.EvalMetrics(
                  id=eval_id,
                  model=resources_pb2.Model(
                      id=model_id,
                      app_id=auth.app_id,
                      user_id=auth.user_id,
                      model_version=resources_pb2.ModelVersion(id=version_id),
                  ),
                  extended_metrics=metrics if ext_metrics else None)
          ],
      ),
      metadata=auth.metadata,
  )
  return post_eval


def make_dataset(auth, app_id, dataset_id, split_word: str = "", max_input=None):
  """Pull dataset from Clarifai platform

  Args:
      auth (auth): Clarifai Auth
      template (str): Name of template
      split_word (str): A string uses to seperate question and answer
      max_input (Union[NoneType, int]): limit maximum inputs when pulling. Set as None to pull everything.
  Returns:
      Union[pd.DataFrame, None]: pd.DataFrame if success, otherwise None
  """
  flag, texts = get_text_dataset_inputs(
      auth,
      user_id=auth.user_id,
      app_id=app_id,
      dataset_id=dataset_id,
      max_input=max_input,
  )
  _df = None
  if flag:
    try:
      json_texts = [json.loads(each) for each in texts]
      _df = pd.DataFrame.from_dict(json_texts, orient="columns")
    except Exception as e:
      logger.error(f"Dataset has non json format, error {e}, try using split word")
      _df = defaultdict(lambda: [])
      for text in texts:
        question, answer = split_sample_general_template(text, split_word)
        _df["question"].append(question)
        _df["answer"].append(answer)
      _df = pd.DataFrame(_df)
  else:
    logger.error(texts)

  return _df
