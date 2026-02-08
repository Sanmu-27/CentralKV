"""
最基础的生成测试 - 不使用 CentralKV
用于验证模型基础功能是否正常
"""
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print("="*60)
print("Basic Generation Test (WITHOUT CentralKV)")
print("="*60)

model_path = "HuggingFaceTB/SmolLM-135M"
cache_dir = "../experiments/models"

print("\n1. Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

print("   Model loaded!")

print("\n2. Testing short generation...")
prompt = "The secret code is 42. Question: What is the secret code? Answer:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
print(f"   Input: {prompt}")
print(f"   Input tokens: {input_ids.shape[1]}")

start_time = time.time()
print("   Generating (5 tokens)...")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=5,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

elapsed = time.time() - start_time
result = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"   ✓ Generation completed in {elapsed:.2f}s")
print(f"   Output: {result}")

# Check if "42" is in the output
if "42" in result:
    print("\n✓ SUCCESS: Model can retrieve '42' from short context!")
else:
    print("\n✗ WARNING: Model did not retrieve '42'")

print("\n3. Testing longer context (500 tokens)...")
long_text = "This is text. " * 100  # ~400 tokens
long_text += "The secret code is 42. "
long_text += "This is text. " * 50  # More padding
long_text += "\nQuestion: What is the secret code? Answer:"

input_ids = tokenizer(long_text, return_tensors="pt").input_ids.to(device)
print(f"   Input tokens: {input_ids.shape[1]}")

start_time = time.time()
print("   Generating...")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

elapsed = time.time() - start_time
pred = tokenizer.decode(output[0][-10:], skip_special_tokens=True)

print(f"   ✓ Generation completed in {elapsed:.2f}s")
print(f"   Predicted: '{pred.strip()}'")

if "42" in pred:
    print("\n✓ SUCCESS: Model can retrieve from longer context!")
else:
    print("\n✗ FAILED: Model did not retrieve '42' from longer context")

print("\n" + "="*60)
print("Basic generation test completed!")
print("="*60)


