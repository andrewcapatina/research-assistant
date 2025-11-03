# shared_llm.py
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, GenerationConfig

import benchmarking

_llm = None
_tokenizer = None
def get_hf_llm():
    global _llm, _tokenizer
    if _llm is None:
        local_model_path = "/app/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
      
        _llm = LLM(
            model=local_model_path,
            quantization="awq",
            tensor_parallel_size=1,  # Adjust for multi-GPU
            max_model_len=2048,
            block_size=16,
            enable_prefix_caching=True,
            dtype=torch.float16,
            max_num_batched_tokens=2048,  # ← FORCE BATCHING
            max_num_seqs=16,              # ← Max concurrent sequences
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

def generate_plan(prompt: str, system: str = "You are a helpful assistant.", **kwargs) -> str:
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

def summarize(papers: str, system: str = "You are a helpful assistant.", **kwargs) -> str:
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

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.8,
    )

    input_tokens = []
    print("prompts")
    for paper in papers:
        message = [
            {"role": "system", "content": system,},
            {"role": "user", "content": paper},
        ]
        input_tokens.append(_tokenizer.apply_chat_template(
            message,
            tokenize=False,  # Return string, not IDs
            add_generation_prompt=True,  # Add assistant header
        ))
        print(message[1])

    #generated_ids = _llm.generate(**tokenized_chat, **gen_kwargs)
    outputs = benchmarking.benchmark_summarization_inference(_llm, input_tokens, sampling)
    print("LLM_OUTPUT")
    summaries = []
    for i, out in enumerate(outputs):
        summary = out.outputs[0].text.strip()
        summaries.append(summary)
        print(f"  Paper {i+1}: {summary[:100]}...")
    
    print(f"Generated {len(summaries)} summaries")
    return summaries

def analyze(prompt: str, system: str = "You are a helpful assistant.", **kwargs) -> str:
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

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.8,
    )
    message = [
        {"role": "system", "content": system,},
        {"role": "user", "content": prompt},
    ]
    formatted =_tokenizer.apply_chat_template(
        message,
        tokenize=False,  # Return string, not IDs
        add_generation_prompt=True,  # Add assistant header
    )

    #generated_ids = _llm.generate(**tokenized_chat, **gen_kwargs)
    outputs = _llm.generate(formatted, sampling)[0]
    return outputs.outputs[0].text