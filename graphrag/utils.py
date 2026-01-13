from llama_index.llms.ollama import Ollama

QWEN_LARGE = "qwen2.5:14b"
QWEN_MED = "qwen2.5:7b"
QWEN_SMALL = "qwen2.5:3b"

def load_llm(model):
    return Ollama(model=model, request_timeout=300.0)