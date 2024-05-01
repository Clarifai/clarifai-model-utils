There are 2 ways to add new task:

# 1. Create task folder:

Create new folder in [template](../template/) named it as the task name. The structure follows this [doc](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)

Suppose you want to use single metric says `f1` and you have your own implementation. So you will do:

- Create folder named `f1` in `template`
- Add files:

  * f1.yaml: contains TaskConfig
  ```yaml
  group:
  - f1
  task: f1
  dataset_path: csv
  dataset_name:
  output_type: generate_until
  validation_split: validation
  fewshot_split: null
  test_split: null
  doc_to_text: "{{question}}"
  doc_to_target: "{{answer}}"
  metric_list:
  - metric: !function utils.f1 # # <--------------------- your function computes this metrics in python file
    aggregation: mean
    higher_is_better: true
  repeats: 1
  num_fewshot: 0
  ```
  * python file named anything, in this case named as `utils.py`: contains functions to process results, in this case is the metric.
  ```python
  import collections
  import re
  import string
  import evaluate


  def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
      regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
      return re.sub(regex, " ", text)

    def white_space_fix(text):
      return " ".join(text.split())

    def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
      return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

  def get_tokens(s):
    if not s:
      return []
    return normalize_answer(s).split()

  def f1(predictions, references): # <--------------------- Assign this to yaml
    gold_toks = get_tokens(references[0])
    pred_toks = get_tokens(predictions[0])
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
      # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
      return int(gold_toks == pred_toks)
    if num_same == 0:
      return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1 * 100

  ```

  The new added template will be loaded in `evaluate.ClarifaiModelHarnessEval` and you can use it in `ClarifaiEvaluator.evaluate` by set args `template="f1"`.

  ```python
  out = evaluator.evaluate(
    template="f1",
    upload=False,
    dataset=ds,
  )
  ```

  For more examples, please see [template](../template/) or visit tasks in `lm_eval` [repo](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)

  This way is good if your template is static all time and you don't want change any parameter at runtime.

  This also means it's hard to change configuration in code, so that we move to 2nd way.

# 2. Use TaskConfig directly

harness-eval loads yaml config into [TaskDict](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py#L55) enventually. So you can configure the template directly on this class. Take the example in (1.) we can turn it to:

```python
from lm_eval.api.task import TaskConfig
import collections
import re
import string
import evaluate

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s:
    return []
  return normalize_answer(s).split()

def f1(predictions, references): # <--------------------- Assign this to yaml
  gold_toks = get_tokens(references[0])
  pred_toks = get_tokens(predictions[0])
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1 * 100

config = TaskConfig(
  task="f1",
  dataset_path="csv",
  dataset_name="",
  output_type="generate_until",
  validation_split="validation",
  doc_to_text="{{question}}",
  doc_to_target="{{answer}}",
  metric_list= [
    {
      "metric": f1,
      "aggregation": "mean",
      "higher_is_better": True
    },
  ],
  repeats=1,
  num_fewshot=0,
)
```

Now using it
```python
out = evaluator.evaluate(
    template=config.to_dict(), # convert TaskConfig to dict
    upload=False,
    dataset=ds,
)
```
