# from ollama import Ollama
from transformers_model import TransformerModel
import transformers
import os
import torch

class LLM(TransformerModel):
    def __init__(self, model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct", causal = True, quantized = True, use_gpu = True):
        super().__init__(model_name, causal, quantized, use_gpu)

        # This use to load model from local
        # if os.path.exists("models/Meta-Llama-3.1-8B-Instruct"):
        #     model_name = "models/Meta-Llama-3.1-8B-Instruct"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # token=os.getenv('HF_TOKEN'),
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate_response(self, input : str | list, max_new_tokens : int = 256, truncation : bool = True):
        """
        Generate a response to the user's query.

        Args:
            input (str/list): Can either be a direct input of data type ``str``, or a chat history of type ``list``.
            max_tokens (int): The maximum number of tokens the LLM is allowed to generate for the response.
            truncation (bool, optional): Whether to force the generated text to be of length max_new_tokens by cutting it off.
            This indicates that you want the model to continue the last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
            By default this is ``True`` when the final message in the input chat has the assistant role and ``False`` otherwise, but you can manually override that behaviour by setting this flag.
        
        Returns:
            chat_history (list): The entire chat history, including the user's prompt and the LLM's response.
            assistant_response (str): The LLM's response to the input.
        """

        chat_history = self.pipeline(
            input,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            truncation = truncation,
            max_new_tokens=max_new_tokens,
            # continue_final_message=continue_final_message
        )
        # chat_history = chat_history
        # assistant_response = chat_history[-1]['content']
        try:
            chat_history = chat_history[0]['generated_text']
            assistant_response = chat_history[-1]['content']
        except Exception as e:
            # print(f"Error retrieving assistant response for LLM generation.\nChat history: {str(chat_history)}\n\nTraceback:\n{str(e)}")
            chat_history = chat_history[0][0]['generated_text']
            assistant_response = chat_history[0][-1]['content']
            # assistant_response = None

        return chat_history, assistant_response