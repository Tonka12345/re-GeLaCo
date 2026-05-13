"""
GeLaCo Prototype Configuration
===============================
All constants and configuration for the single-evaluation prototype.
"""

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
TORCH_DTYPE = "float16"  # fp16 inference

# =============================================================================
# Merge Configuration
# =============================================================================
# Each merge operation is a tuple (base, end, active)
# - base: starting layer index (0-indexed)
# - end: ending layer index (inclusive)
# - active: 1 = apply merge, 0 = skip
#
# Example: (5, 7, 1) means merge layers 5, 6, 7 into layer 5
# The differential weight merging formula:
#   θ*_base = θ_base + Σ_{k=1}^{end-base} (θ_{base+k} - θ_base)
#
MERGE_OPERATIONS = [(5, 7, 1)]

# Total layers in Llama-2-7B
NUM_ORIGINAL_LAYERS = 32

# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
NUM_CALIBRATION_SENTENCES = 64  # Start small; scale to 64 later
MAX_SENTENCE_LENGTH = 128      # Max tokens per sentence
RANDOM_SEED = 42

# =============================================================================
# Device Configuration
# =============================================================================
DEVICE = "cuda"  # Use "cpu" for testing without GPU

# =============================================================================
# Server / IPC Configuration (used by server.py)
# =============================================================================
# Paths are taken from environment variables when set (the PBS job script
# generates per-job FIFO paths under /tmp); the defaults below are convenient
# for interactive smoke testing.
import os as _os

FIFO_REQ_PATH = _os.environ.get("GELACO_REQ_FIFO", "/tmp/gelaco_req")
FIFO_RSP_PATH = _os.environ.get("GELACO_RSP_FIFO", "/tmp/gelaco_rsp")
READY_FILE    = _os.environ.get("GELACO_READY_FILE", "/tmp/gelaco_ready")
