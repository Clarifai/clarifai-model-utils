This template employs LLM (llama2-70b-chat) to evaluate model response based on input question context and grouth truth. Computing binary score using the [CriteriaEvalChain](https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain) of the 'langchain':
* relevance
* depth
* creativity
* correctness
* helpfulness
