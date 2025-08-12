import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
import time

if __name__ == "__main__":

    # Set multiprocessing start method to 'spawn' to avoid CUDA issues with vLLM
    multiprocessing.set_start_method('spawn', force=True)
    mp.set_start_method('spawn', force=True)

    print("Step 1: Starting snapshot_download for model 'moonshotai/Kimi-VL-A3B-Thinking-2506' (local files only)...")
    model_path = snapshot_download("moonshotai/Kimi-VL-A3B-Thinking-2506", local_files_only=True)
    print(f"Step 1 complete: Model available at: {model_path}")

    print("Step 2: Initializing LLM...")
    _llm_start = time.perf_counter()
    llm = LLM(
        model_path,
        trust_remote_code=True,
        max_num_seqs=8,
        max_model_len=131072,
        limit_mm_per_prompt={"image": 256}
    )
    print(f"Step 2 complete: LLM initialized in {time.perf_counter() - _llm_start:.2f}s")

    print("Step 3: Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Step 3 complete: Processor loaded.")

    print("Step 4: Creating sampling params...")
    sampling_params = SamplingParams(max_tokens=32768, temperature=0.8)
    print("Step 4 complete: Sampling params ready.")


    from PIL import Image

    def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
        if bot in text and eot not in text:
            return ""
        if eot in text:
            return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot) :].strip()
        return "", text

    OUTPUT_FORMAT = "--------Thinking--------\n{thinking}\n\n--------Summary--------\n{summary}"

    print("Step 5: Loading image from ./demo6.jpeg ...")
    image = Image.open("./demo6.jpeg")
    print(f"Step 5 complete: Image loaded. Size: {getattr(image, 'size', 'unknown')}")

    print("Step 6: Preparing messages...")
    messages = [
        {"role": "user", "content": [{"type": "image", "image": ""}, {"type": "text", "text": "What kind of cat is this? Answer with one word."}]}
    ]
    print("Step 6 complete: Messages prepared.")

    print("Step 7: Applying chat template...")
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    print("Step 7 complete: Chat template applied.")

    print("Step 8: Starting generation...")
    _gen_start = time.perf_counter()
    outputs = llm.generate([{"prompt": text, "multi_modal_data": {"image": image}}], sampling_params=sampling_params)
    print(f"Step 8 complete: Generation finished in {time.perf_counter() - _gen_start:.2f}s")
    generated_text = outputs[0].outputs[0].text
    print(f"Step 9: Generated text length: {len(generated_text)}")

    print("Step 10: Parsing thinking and summary...")
    thinking, summary = extract_thinking_and_summary(generated_text)
    print("Step 10 complete: Parsed thinking and summary.")
    print(OUTPUT_FORMAT.format(thinking=thinking, summary=summary))
    print("All steps complete.")
