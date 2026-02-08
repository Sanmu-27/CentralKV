import sys
import os
# Add project root to path to allow direct execution without install
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import traceback
from tqdm import tqdm
from centralkv.models import enable_centralkv_for_llama
from transformers import AutoModelForCausalLM, AutoTokenizer


DEBUG_LOG = os.path.join(os.path.dirname(__file__), "niah_debug.log")

def log(msg: str):
    print(msg, flush=True)
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        # Logging should never crash the run
        pass

def generate_haystack(length, needle_depth, needle="The secret code is 42."):
    """Generates a long context with a needle inserted at specific depth."""
    # Length here refers to number of sentences, not tokens
    # Each sentence is ~5 tokens, so we adjust for model max length
    context = ["This is text. " for _ in range(length)]
    insert_idx = int(length * needle_depth)
    context.insert(insert_idx, needle + " ")
    return "".join(context)

def run_test(args):
    # Model Setup
    log(f"Downloading/Loading model to {args.cache_dir}...")
    log("Step 1/5: Determining device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  Device: {device}")
    
    log(f"Step 2/5: Loading model from {args.model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            device_map="auto", 
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            cache_dir=args.cache_dir  # Save to local folder
        )
        log("  Model loaded successfully!")
    except Exception as e:
        log(f"  ERROR loading model: {e}")
        raise
    
    log("Step 3/5: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir
        )
        log("  Tokenizer loaded successfully!")
    except Exception as e:
        log(f"  ERROR loading tokenizer: {e}")
        raise
    
    # Get model's max length and adjust test parameters
    log("Step 4/5: Reading model configuration...")
    max_model_length = getattr(model.config, 'max_position_embeddings', 2048)
    log(f"  Model: {args.model_path}")
    log(f"  Device: {device}")
    log(f"  Max position embeddings: {max_model_length} tokens")
    
    log("Step 5/5: Enabling CentralKV...")
    if args.method == "centralkv":
        enable_centralkv_for_llama(model, budget_ratio=args.budget, lambda_val=0.5)
        log("  CentralKV enabled!")
    
    # Grid Search
    # Note: SmolLM-135M has max_position_embeddings=2048, so we use shorter lengths
    lengths = [500, 1000, 1500]  # Shorter lengths suitable for small models
    depths = [0.1, 0.5, 0.9]  # Fewer depth points for faster testing
    results = np.zeros((len(lengths), len(depths)))
    
    total_tests = len(lengths) * len(depths)
    current_test = 0
    start_time = time.time()
    test_times = []
    
    log(f"\n{'='*60}")
    log(f"Starting NIAH Benchmark: {total_tests} tests total")
    log(f"Lengths: {lengths}")
    log(f"Depths: {depths}")
    log(f"Method: {args.method}, Budget: {args.budget}")
    log(f"{'='*60}\n")
    
    for i, length in enumerate(lengths):
        for j, depth in enumerate(depths):
            current_test += 1
            test_start = time.time()
            
            log(f"\n[{current_test}/{total_tests}] Testing Length={length}, Depth={depth:.1f}")
            log(f"Progress: [{'='*int(current_test/total_tests*40)}{' '*(40-int(current_test/total_tests*40))}] {current_test/total_tests*100:.1f}%")
            
            # Time estimate
            if test_times:
                avg_time = np.mean(test_times)
                remaining_tests = total_tests - current_test
                eta_seconds = avg_time * remaining_tests
                eta_mins = int(eta_seconds // 60)
                eta_secs = int(eta_seconds % 60)
                log(f"  ETA: ~{eta_mins}m {eta_secs}s (Avg: {avg_time:.1f}s/test)")
            
            text = generate_haystack(length, depth)
            
            # Simple retrieval prompt
            prompt = text + "\nQuestion: What is the secret code? Answer:"
            tokenized = tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            
            input_len = input_ids.shape[1]
            log(f"  Input tokens: {input_len}")
            
            # Skip if input exceeds model capacity
            if input_len > max_model_length - 20:  # Leave room for generation
                log(f"  [SKIPPED - exceeds max {max_model_length}]")
                results[i, j] = 0  # Count as failure
                test_times.append(1.0)  # Minimal time
                continue
            
            log("  Generating...")
            
            try:
                with torch.no_grad():
                    output = model.generate(
                        input_ids, 
                        attention_mask=attention_mask,
                        max_new_tokens=10,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        use_cache=True
                    )
                log("  Generation completed.")
            except Exception as gen_error:
                log(f"  ERROR during generation: {type(gen_error).__name__}: {gen_error}")
                log(traceback.format_exc())
                results[i, j] = 0
                test_times.append(1.0)
                continue
            
            pred = tokenizer.decode(output[0][-10:], skip_special_tokens=True)
            success = "42" in pred
            results[i, j] = 1 if success else 0
            
            test_elapsed = time.time() - test_start
            test_times.append(test_elapsed)
            
            status = "✓ SUCCESS" if success else "✗ FAILED"
            log(f"  Result: {status} (Predicted: '{pred.strip()}') [{test_elapsed:.1f}s]")
            log(f"  Cumulative accuracy: {results.sum()}/{current_test} = {results.sum()/current_test*100:.1f}%")
            
    # Final Summary
    total_time = time.time() - start_time
    total_mins = int(total_time // 60)
    total_secs = int(total_time % 60)
    
    log(f"\n{'='*60}")
    log("NIAH Benchmark Completed!")
    log(f"{'='*60}")
    log(f"Total Time: {total_mins}m {total_secs}s")
    log(f"Average Time per Test: {np.mean(test_times):.1f}s")
    log(f"Overall Accuracy: {results.sum()}/{total_tests} = {results.mean()*100:.2f}%")
    log("\nResults by Length:")
    for i, length in enumerate(lengths):
        acc = results[i, :].mean() * 100
        log(f"  Length {length:5d}: {acc:.1f}% ({int(results[i, :].sum())}/{len(depths)} tests)")
    log("\nResults by Depth:")
    for j, depth in enumerate(depths):
        acc = results[:, j].mean() * 100
        log(f"  Depth {depth:.1f}: {acc:.1f}% ({int(results[:, j].sum())}/{len(lengths)} tests)")
    log(f"{'='*60}\n")
    
    # Plotting
    log("Generating heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(results, xticklabels=depths, yticklabels=lengths, annot=True, cmap="YlGnBu", 
                fmt='.0f', cbar_kws={'label': 'Success (1) / Failure (0)'})
    plt.xlabel("Needle Depth")
    plt.ylabel("Context Length")
    plt.title(f"NIAH Results - {args.method} (Budget: {args.budget}, Accuracy: {results.mean()*100:.1f}%)")
    output_file = f"niah_{args.method}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    log(f"Heatmap saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Use SmolLM-135M (Llama architecture, very small like GPT-2) for fast local testing
    parser.add_argument("--model_path", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--method", type=str, default="centralkv", choices=["centralkv", "h2o", "full"])
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--cache_dir", type=str, default="./local_models")
    args = parser.parse_args()
    
    # Create local model directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    try:
        run_test(args)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\n{'='*60}")
        print(f"ERROR OCCURRED")
        print(f"{'='*60}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")

