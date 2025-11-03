# shared_llm.py
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, GenerationConfig

_llm = None
_tokenizer = None
def get_hf_llm():
    global _llm, _tokenizer
    if _llm is None:
        local_model_path = "/app/models/llama-3.1-8b-awq-instruct"

        # 2. Optional: 4-bit quantization (saves VRAM, faster on DGX)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        
        _llm = LLM(
            model=local_model_path,  # ← AWQ path
            quantization="awq",
            tensor_parallel_size=1,  # Adjust for multi-GPU
            dtype=torch.float16,  # vLLM detects 4-bit
            gpu_memory_utilization=0.9,
        )

        _tokenizer = get_tokenizer(local_model_path)

        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
            _tokenizer.pad_token_id = _tokenizer.eos_token_id
    return _llm

import gc
import sys

def cleanup_and_exit(signum=None, frame=None):

    print("\nCleaning up GPU memory...")

    # 1. Delete references
    if _llm is not None:
        del llm

    # 2. Force garbage collection
    gc.collect()

    # 3. Empty CUDA cache (this is the key!)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all kernels to finish

    print("GPU memory released. Exiting.")
    sys.exit(0)

def generate(prompt: str, system: str = "You are a helpful assistant.", **kwargs) -> str:
    """
    Generate using Llama 3.1 format (works for base model).
    """
    if 'temperature' in kwargs:
        temperature = kwargs['temperature']
    else:
        temperature = 0.8

    if 'max_tokens' in kwargs:
        max_tokens = kwargs['max_tokens']
    else:
        max_tokens = 150

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.5,
    )
    # Use chat template if available, else fallback
    # model_inputs = _tokenizer([prompt], padding=True, return_tensors="pt").to(_llm.device)
    messages = [
        {"role": "system", "content": system,},
        {"role": "user", "content": prompt},
    ]
    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return string, not IDs
        add_generation_prompt=True,  # Add assistant header
    )

    outputs = _llm.generate([formatted], sampling)[0]
    outputs = outputs.outputs[0].text
    return outputs

def summarize(prompt: str, system: str = "You are a helpful assistant.", **kwargs) -> str:
    """
    summarize using Llama 3.1 format (works for instruct model).
    """
    if 'temperature' in kwargs:
        temperature = kwargs['temperature']
    else:
        temperature = 0.8
    if 'max_tokens' in kwargs:
        max_tokens = kwargs['max_tokens']
    else:
        max_tokens = 150

    messages = [
        {"role": "system", "content": system,},
        {"role": "user", "content": prompt},
    ]
 
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.8,
    )

    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return string, not IDs
        add_generation_prompt=True,  # Add assistant header
    )

    #generated_ids = _llm.generate(**tokenized_chat, **gen_kwargs)
    # outputs = _tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs = _llm.generate([formatted], sampling)[0]
    return outputs.outputs[0].text