import datetime


def handle_arg_string(arg):
  if arg.lower() == "true":
    return True
  elif arg.lower() == "false":
    return False
  return arg


def simple_parse_args_string(args_string):
  """
  Parses something like
      args1=val1,arg2=val2
  Into a dictionary
  """
  args_string = args_string.strip()
  if not args_string:
    return {}
  arg_list = [arg for arg in args_string.split(",") if arg]
  args_dict = {k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]}
  return args_dict


def compute_overall(summary: dict, normalized_weights: dict):
  overall_score = 0
  for k, v in summary.items():
    overall_score += normalized_weights[k] * v
  return overall_score


def split_sample_general_template(text, split_word) -> tuple:
  index = text.find(split_word) + len(split_word)
  question = text[:index]
  answer = text[index:]

  return question, answer


def convert_to_datetime(timestamp):
  # Convert the Unix timestamp to a datetime object
  datetime_obj = datetime.datetime.utcfromtimestamp(timestamp)
  # Convert the datetime object to a string in a specific format
  datetime_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

  return datetime_str
