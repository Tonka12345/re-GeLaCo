# GeLaCo Prototype — Single Evaluation

A pure Python prototype that faithfully reproduces the **GeLaCo** layer-collapse
method ([arXiv 2507.10059](https://arxiv.org/abs/2507.10059)) for a single
evaluation on **Llama-2-7B**.

## What This Does

1. Loads `meta-llama/Llama-2-7b-hf` (fp16)
2. Creates a deep copy for the compressed model
3. Applies one manual layer merge: layers 5→7 collapsed into layer 5
4. Runs inference on Wikipedia calibration sentences
5. Computes the GeLaCo module-wise similarity fitness:
   - Attention similarity (q/k/v/o_proj activations)
   - FFN similarity (gate/up/down_proj activations)
   - Hidden state similarity (final hidden states)
6. Prints the fitness score

## Project Structure

```
re-GeLaCo/
├── config.py           # Configuration constants
├── data_loader.py      # Wikipedia sentence loading
├── layer_merge.py      # Differential weight merging
├── fitness.py          # Hook-based similarity fitness
├── evaluate.py         # Main prototype script
├── requirements.txt    # Python dependencies
├── run_prototype.pbs   # PBS job script for Supek
└── README.md           # This file
```

---

## Setup on Supek Cluster

### 1. SSH into Supek

```bash
ssh tsegvic@login-gpu.hpc.srce.hr
```
(replace tsegvic with your username)


### 2. Configure ~/.bashrc (do this once)

#### 2.1 Set up proxy (required for internet access)

Worker nodes need the squid proxy. Add to your `~/.bashrc` or run manually:

```bash
echo 'export http_proxy="http://10.150.1.1:3128"' >> ~/.bashrc
echo 'export https_proxy="http://10.150.1.1:3128"' >> ~/.bashrc
source ~/.bashrc
```

#### 2.2 HuggingFace authentication

Llama-2 requires a HuggingFace token with access to `meta-llama/Llama-2-7b-hf`.

```bash
echo 'export HF_TOKEN="hf_YOUR_TOKEN_HERE"' >> ~/.bashrc
source ~/.bashrc
```

**Option B: Using huggingface-cli**

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your HF token when prompted
```

> **Note**: You need to accept the Llama-2 license at
> https://huggingface.co/meta-llama/Llama-2-7b-hf before downloading.
> By default, HuggingFace caches models in `~/.cache/huggingface/`

### 3. Set python versions

By default python version is 3.6, we need python 3.11.7:

```bash
module load cray-python/3.11.7
```

### 4. Clone the repository

```bash
cd ~
git clone https://github.com/Tonka12345/re-GeLaCo.git
cd re-GeLaCo
git checkout evaluation-testing
```

### 5. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Prototype

### Option A: Interactive GPU Session (recommended for testing)

```bash
# Request interactive session with 1 GPU
qsub -I -q gpu -l select=1:ncpus=8:ngpus=1:mem=64GB -l walltime=02:00:00

# Once on the GPU node, re-export environment variables (worker nodes start a fresh shell that may not source your .bashrc automatically):

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"
export HF_TOKEN="YOUR_TOKEN_HERE"
export HF_HOME="/lustre/home/tsegvic/.hf_cache"

cd ~/re-GeLaCo
source venv/bin/activate

# Run the prototype
python evaluate.py
```

### Option B: Batch Job

```bash
cd ~/re-GeLaCo
qsub run_prototype.pbs
```

Check job status:

```bash
qstat -u $USER
```

View output:

```bash
cat gelaco-proto.o*   # stdout
cat gelaco-proto.e*   # stderr
```

---

## Expected Output

```
============================================================
  GeLaCo Prototype — Single Evaluation
============================================================
Model:            meta-llama/Llama-2-7b-hf
Merge operations: [(5, 7, 1)]
...
  Attention similarity:    0.XXXXXX
  FFN similarity:          0.XXXXXX
  Hidden state similarity: 0.XXXXXX
  ─────────────────────────────────
  Overall fitness:         0.XXXXXX
```

## Memory Requirements

- Llama-2-7B fp16: ~14 GB VRAM
- Deep copy: ~14 GB additional
- Activations + overhead: ~4-6 GB
- **Total: ~32-34 GB** (fits on A100 40GB)

## Future: ECF Integration

This prototype is structured so `evaluate.py` can later be called from ECF's
C++ `evaluate()` operator via `subprocess` or Python/C++ bindings. The merge
operations list in `config.py` will be replaced by ECF's genotype.
