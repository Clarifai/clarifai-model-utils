import collections
import re
import string

import evaluate

ROUGE_SCORER = evaluate.load("rouge")
SACREBLEU = evaluate.load("sacrebleu")


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


def f1(predictions, references):
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


def sacrebleu(predictions, references):
  """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
  score = SACREBLEU.compute(
      predictions=predictions,
      references=[references],
      smooth_method="exp",
      smooth_value=0.0,
      force=False,
      lowercase=False,
      tokenize="intl",
      use_effective_order=False,
  )

  return score['score']


def prepare_summary(summary):
  summary = summary.replace(" . ", ".\n")

  return summary


def _rouge(predictions, references, rouge_type):
  """
  calculate rouge types score
  """
  _refs = list(map(prepare_summary, references))
  _preds = list(map(prepare_summary, predictions))
  result = ROUGE_SCORER.compute(
      predictions=_preds, references=_refs, rouge_types=[rouge_type], use_aggregator=True)

  return result[rouge_type] * 100


def rouge1(predictions, references):
  return _rouge(predictions, references, "rouge1")


def rouge2(predictions, references):
  return _rouge(predictions, references, "rouge2")


def rougeL(predictions, references):
  return _rouge(predictions, references, "rougeL")
