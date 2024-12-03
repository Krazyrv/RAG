from ollama import Ollama
from transformers import TransformerModel

class LLM(TransformerModel):
    def __init__(self, model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct", causal = True, quantized = True, use_gpu = True):


        super().__init__(model_name, causal, quantized, use_gpu)
