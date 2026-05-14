#!/usr/bin/env python3
"""
GeLaCo Persistent Evaluator Server
====================================
Loads Llama-2-7B once, then services fitness-evaluation requests from the
ECF (C++) NSGA-II driver over two named FIFOs.

Wire protocol (line-oriented, blocking):

  Request  (ECF → server):
    JSON array of [b, e, a] triples, e.g.  [[5,7,1],[10,12,1]]   then "\n"
    Sentinel:  the literal string "QUIT\n"  causes a clean shutdown.

  Response (server → ECF):
    "OK <similarity> <compression>\n"   on success
    "ERR 0.0 0.0 <message>\n"           on failure (one-line, no newlines inside)

The server writes a READY sentinel file once the model is loaded and the
FIFOs have been opened — the PBS script polls this file before launching ECF.

VRAM strategy:
  The original model is loaded once. For each request we deepcopy it,
  apply the merge operations, evaluate, then drop the copy and call
  torch.cuda.empty_cache(). Peak VRAM stays at ~2x model size (~28 GB
  for Llama-2-7B fp16), comfortably under A100 40 GB.
"""

import copy
import gc
import json
import math
import os
import sys
import time
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    MODEL_NAME,
    TORCH_DTYPE,
    NUM_ORIGINAL_LAYERS,
    DEVICE,
    FIFO_REQ_PATH,
    FIFO_RSP_PATH,
    READY_FILE,
)
from data_loader import load_calibration_sentences
from layer_merge import apply_merge_operations
from fitness import compute_fitness


def _log(msg: str) -> None:
    print(f"[server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "no-cuda"
    a = torch.cuda.memory_allocated() / (1024**3)
    r = torch.cuda.memory_reserved() / (1024**3)
    return f"alloc={a:.2f}GB reserved={r:.2f}GB"


def load_model_once():
    _log(f"loading {MODEL_NAME} (dtype={TORCH_DTYPE}, device={DEVICE})")
    dtype = torch.float16 if TORCH_DTYPE == "float16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    model.eval()

    n = len(model.model.layers)
    assert n == NUM_ORIGINAL_LAYERS, f"expected {NUM_ORIGINAL_LAYERS} layers, got {n}"
    _log(f"model loaded: {n} layers; {_gpu_mem()}")
    return model, tokenizer


def evaluate_one(
    original_model,
    tokenizer,
    sentences,
    merge_ops_triples,
):
    """
    Run one evaluation: deepcopy → merge → fitness → return (similarity, compression).
    """
    compressed = copy.deepcopy(original_model)
    compressed.eval()
    try:
        layer_mapping = apply_merge_operations(
            compressed, merge_ops_triples, NUM_ORIGINAL_LAYERS,
        )
        res = compute_fitness(
            original_model=original_model,
            compressed_model=compressed,
            layer_mapping=layer_mapping,
            tokenizer=tokenizer,
            sentences=sentences,
            num_original_layers=NUM_ORIGINAL_LAYERS,
            device=DEVICE,
            verbose=False,
        )
        removed = NUM_ORIGINAL_LAYERS - len(compressed.model.layers)
        compression = removed / NUM_ORIGINAL_LAYERS
        sim = float(res["fitness"])
        # Over-aggressive compression (e.g. 30/32 layers removed) makes
        # the surviving stack numerically unstable → logits become NaN/Inf
        # and every similarity reduces to NaN. Map those to a worst-case
        # similarity so NSGA-II dominates them out cleanly instead of
        # poisoning the front (NaN compares are undefined behavior).
        if not math.isfinite(sim):
            sim = -1.0
        return sim, float(compression)
    finally:
        del compressed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def parse_request(line: str):
    """
    Parse a request line into a list of (b, e, a) tuples.
    Accepts either a JSON list of triples or the QUIT sentinel.
    """
    line = line.strip()
    if line == "QUIT":
        return None
    raw = json.loads(line)
    ops = []
    for item in raw:
        if len(item) != 3:
            raise ValueError(f"expected triple, got {item!r}")
        b, e, a = int(item[0]), int(item[1]), int(item[2])
        ops.append((b, e, a))
    return ops


def main():
    # Cleanly tear down any stale READY from a previous run before loading.
    try:
        if os.path.exists(READY_FILE):
            os.remove(READY_FILE)
    except OSError:
        pass

    original_model, tokenizer = load_model_once()
    sentences = load_calibration_sentences()
    _log(f"loaded {len(sentences)} calibration sentences")

    # Open FIFOs. The PBS script mkfifo'd them before launching us.
    if not (os.path.exists(FIFO_REQ_PATH) and os.path.exists(FIFO_RSP_PATH)):
        _log(f"FIFOs missing: req={FIFO_REQ_PATH} rsp={FIFO_RSP_PATH}")
        sys.exit(2)

    # Touch the READY sentinel BEFORE opening the FIFOs, so the launcher can
    # safely open the other ends without us deadlocking on an absent peer.
    with open(READY_FILE, "w") as f:
        f.write("ready\n")
    _log(f"READY sentinel written to {READY_FILE}")

    # Open req for read first (blocks until the writer connects), then rsp for write.
    _log(f"opening req FIFO {FIFO_REQ_PATH} (blocking)")
    req = open(FIFO_REQ_PATH, "r")
    _log(f"opening rsp FIFO {FIFO_RSP_PATH} (blocking)")
    rsp = open(FIFO_RSP_PATH, "w")
    _log("FIFOs connected; entering eval loop")

    n_served = 0
    try:
        while True:
            line = req.readline()
            if not line:
                _log("req FIFO closed by peer; exiting")
                break
            try:
                ops = parse_request(line)
            except Exception as e:
                rsp.write(f"ERR 0.0 0.0 parse_error: {type(e).__name__}: {e}\n")
                rsp.flush()
                continue

            if ops is None:
                _log("received QUIT")
                break

            n_served += 1
            t0 = time.time()
            try:
                sim, comp = evaluate_one(original_model, tokenizer, sentences, ops)
                dt = time.time() - t0
                tag = " [NaN→-1.0]" if sim == -1.0 else ""
                _log(f"eval #{n_served} ops={ops} sim={sim:.6f} comp={comp:.6f}"
                     f"{tag} ({dt:.1f}s, {_gpu_mem()})")
                rsp.write(f"OK {sim:.6f} {comp:.6f}\n")
            except Exception as e:
                tb = traceback.format_exc().replace("\n", " | ")
                _log(f"eval #{n_served} FAILED: {tb}")
                # Sanitize: no newlines in the error message.
                msg = f"{type(e).__name__}: {e}".replace("\n", " ")
                rsp.write(f"ERR 0.0 0.0 {msg}\n")
            rsp.flush()
    finally:
        try: req.close()
        except Exception: pass
        try: rsp.close()
        except Exception: pass
        try:
            if os.path.exists(READY_FILE):
                os.remove(READY_FILE)
        except OSError:
            pass
        _log(f"shutdown: served {n_served} evaluation(s)")


if __name__ == "__main__":
    main()
    # Hard-exit to avoid PyArrow/streaming-dataset background-thread segfaults
    # at Python finalization (same workaround as data_loader.py).
    os._exit(0)
