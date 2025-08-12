from vllm import LLM
import os
from huggingface_hub import snapshot_download

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


print(f"Running on {os.environ['SLURM_JOB_NODELIST']}")
print(prompts)

# Set "tensor_parallel_size" to the TOTAL number of GPUs on all nodes.

model_path = snapshot_download("facebook/opt-125m", local_files_only=True)
llm = LLM(model=model_path, tensor_parallel_size=4)

print(llm)

outputs = llm.generate(prompts)

print(outputs)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
print("Done")