from dataclasses import dataclass

WORKFLOW = "workflow"
MODEL = "model"

BGE_BASE_EMBED_MDOEL = "https://clarifai.com/clarifai/main/models/BAAI-bge-base-en-v15"

@dataclass
class JUDGE_LLMS:
  GPT3_5_TURBO = "https://clarifai.com/openai/chat-completion/models/GPT-3_5-turbo"
  LLAMA2_CHAT_70B = "https://clarifai.com/meta/Llama-2/models/llama2-70b-chat"
  GPT4 = "https://clarifai.com/openai/chat-completion/models/GPT-4"
