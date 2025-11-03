from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/app/models/llama-3.1-8b-instruct"  # Your existing model path
quant_path = "/app/models/llama-3.1-8b-instruct-awq"  # Output path for quantized model

# Step 1: Load the model (no quant config here)
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,  # Reduces memory during load
    use_cache=False,         # Disables cache for quantization
)

# Step 2: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Step 3: Define quant_config (this is where zero_point and q_group_size go)
quant_config = {
    "zero_point": True,      # Enables zero-point quantization
    "q_group_size": 128,     # Group size for quantization (128 is common for 4-bit)
    "w_bit": 4,              # 4-bit weights
    "version": "GEMM",       # GEMM version for compatibility
}

# Step 4: Quantize the model using the config
model.quantize(tokenizer, quant_config=quant_config)

# Step 5: Save the quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"Quantized model saved to '{quant_path}'")