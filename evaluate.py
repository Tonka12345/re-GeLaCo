#!/usr/bin/env python3
"""
GeLaCo Prototype — Single Evaluation Script
=============================================
A pure Python prototype that:
1. Loads original Llama-2-7B model
2. Creates a deep copy for the compressed model
3. Applies ONE manual layer merge operation
4. Runs inference on calibration sentences
5. Computes GeLaCo module-wise similarity fitness
6. Prints final fitness score

This is the FIRST prototype step — no evolutionary search, no ECF.
Eventually, ECF's evaluate() will call this as an external evaluator.
"""

import gc
import time
import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    MODEL_NAME,
    TORCH_DTYPE,
    MERGE_OPERATIONS,
    NUM_ORIGINAL_LAYERS,
    DEVICE,
)
from data_loader import load_calibration_sentences
from layer_merge import apply_merge_operations
from fitness import compute_fitness


def print_separator(title: str) -> None:
    """Print a formatted section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_gpu_memory() -> None:
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[GPU] Memory: {allocated:.2f} GB allocated, "
              f"{reserved:.2f} GB reserved")


def main():
    start_time = time.time()

    print_separator("GeLaCo Prototype — Single Evaluation")
    print(f"Model:            {MODEL_NAME}")
    print(f"Device:           {DEVICE}")
    print(f"Dtype:            {TORCH_DTYPE}")
    print(f"Merge operations: {MERGE_OPERATIONS}")

    # =========================================================================
    # Step 1: Load original model and tokenizer
    # =========================================================================
    print_separator("Step 1: Loading Original Model")
    print(f"Loading {MODEL_NAME}...")

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(TORCH_DTYPE, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        original_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=DEVICE,
            low_cpu_mem_usage=True,
        )
    except OSError as e:
        print(f"ERROR: Failed to load model. Check that:")
        print(f"  1. HF_TOKEN is set (export HF_TOKEN='hf_...')")
        print(f"  2. You accepted the Llama-2 license on HuggingFace")
        print(f"  3. Proxy is set (export http_proxy=...)")
        raise SystemExit(1) from e

    original_model.eval()

    num_layers = len(original_model.model.layers)
    print(f"✓ Model loaded: {num_layers} layers")
    print_gpu_memory()

    assert num_layers == NUM_ORIGINAL_LAYERS, (
        f"Expected {NUM_ORIGINAL_LAYERS} layers, got {num_layers}"
    )

    # =========================================================================
    # Step 2: Deep copy for compressed model
    # =========================================================================
    print_separator("Step 2: Creating Deep Copy for Compressed Model")
    print("Creating deep copy (this may take a moment)...")

    compressed_model = copy.deepcopy(original_model)
    compressed_model.eval()

    # Free any fragmented GPU memory after the large allocation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print(f"✓ Deep copy created: {len(compressed_model.model.layers)} layers")
    print_gpu_memory()

    # =========================================================================
    # Step 3: Apply layer merge operations
    # =========================================================================
    print_separator("Step 3: Applying Layer Merge Operations")

    layer_mapping = apply_merge_operations(compressed_model, MERGE_OPERATIONS)

    compressed_layers = len(compressed_model.model.layers)
    removed = num_layers - compressed_layers
    compression_ratio = removed / num_layers

    print(f"\n✓ Merge complete:")
    print(f"  Original layers:    {num_layers}")
    print(f"  Compressed layers:  {compressed_layers}")
    print(f"  Layers removed:     {removed}")
    print(f"  Compression ratio:  {compression_ratio:.4f} ({compression_ratio*100:.1f}%)")
    print_gpu_memory()

    # =========================================================================
    # Step 4: Load calibration sentences
    # =========================================================================
    print_separator("Step 4: Loading Calibration Sentences")

    sentences = load_calibration_sentences()

    # =========================================================================
    # Step 5: Compute fitness
    # =========================================================================
    print_separator("Step 5: Computing Module-wise Similarity Fitness")

    results = compute_fitness(
        original_model=original_model,
        compressed_model=compressed_model,
        layer_mapping=layer_mapping,
        tokenizer=tokenizer,
        sentences=sentences,
        num_original_layers=num_layers,
        device=DEVICE,
    )

    # =========================================================================
    # Step 6: Print final results
    # =========================================================================
    elapsed = time.time() - start_time

    print_separator("FINAL RESULTS")
    print(f"Model:                  {MODEL_NAME}")
    print(f"Merge operations:       {MERGE_OPERATIONS}")
    print(f"Compression:            {num_layers} → {compressed_layers} layers "
          f"({compression_ratio*100:.1f}%)")
    print(f"Calibration sentences:  {len(sentences)}")
    print()
    print(f"  Attention similarity:    {results['attention_similarity']:.6f}")
    print(f"  FFN similarity:          {results['ffn_similarity']:.6f}")
    print(f"  Hidden state similarity: {results['hidden_state_similarity']:.6f}")
    print(f"  ─────────────────────────────────")
    print(f"  Overall fitness:         {results['fitness']:.6f}")
    print()
    print(f"  Elapsed time: {elapsed:.1f}s")
    print_separator("Done")


if __name__ == "__main__":
    main()
