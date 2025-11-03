import subprocess
import time

import shared_llm

def benchmark_summarization_inference(input_text, system):
    start_time = time.time()
    # Get initial power draw
    power_cmd = "nvidia-smi --query-gpu=power.draw --format=csv,noheader"
    initial_power = float(subprocess.check_output(power_cmd.split()).decode().strip().split()[0])
    
    result = shared_llm.summarize(system, input_text, temperature=0.9, max_new_tokens=1000) # Run the summarization
    
    end_time = time.time()
    final_power = float(subprocess.check_output(power_cmd.split()).decode().strip().split()[0])
    
    latency = end_time - start_time
    avg_power = (initial_power + final_power) / 2
    print(f"Latency: {latency:.2f}s | Avg Power: {avg_power:.2f}W")
    return result