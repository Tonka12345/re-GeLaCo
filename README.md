# GeLaCo — Generative Layer Collapse for LLM Compression

A reproduction of the **GeLaCo** method from
[arXiv:2507.10059](https://arxiv.org/abs/2507.10059), targeting **Llama-2-7B**
on the **Supek** HPC cluster.

GeLaCo compresses a pre-trained LLM by **collapsing groups of consecutive
transformer layers** into one. Which groups to collapse — and how many — is
discovered by a **NSGA-II algorithm** that
trades **module-wise output similarity** (preserved quality) against
**compression ratio** (layers removed). The result is a Pareto front of
compressed checkpoints.

This repository contains the full evolutionary search:
[evaluator/server.py](evaluator/server.py) loads Llama-2-7B once and services fitness
requests over named FIFOs from a C++ NSGA-II driver in [ecf/](ecf/).

Two run configurations are provided:

- **5h variant** (current) — 4,000 evaluations, ~20 generations. Produces a
  real but under-converged Pareto front. Fits in shared GPU queues easily.
- **72h variant** (paper-faithful, planned) — 30,000 evaluations,
  ~150 generations. Matches the paper exactly. **Not yet executed** — needs a
  72 h slot in the GPU queue.

A separate single-evaluation prototype lives on the `evaluation-testing`
branch — see its `README.md` for that workflow.

---

## CURRENT STATUS

**COMPLETED**
[Run 1](#61a-run-the-5h-variant-current)
 (5h limit): 4000 evaluations (cca 20 generations), population size 200; artifacts: milestone-5h.txt, gelaco-5h.o938607

**IN PROGRESS**
[Run 2](#61b-run-the-72h-variant-future) (72h limit): 30000 evaluations (cca 150 generations), population size 200.

## Table of contents

1. [Algorithm overview](#1-algorithm-overview)
2. [Architecture](#2-architecture)
3. [Repository layout](#3-repository-layout)
4. [Configuration & parameters](#4-configuration--parameters)
5. [One-time cluster setup](#5-one-time-cluster-setup)
6. [Running the full evolutionary search](#6-running-the-full-evolutionary-search)
7. [Output artifacts](#7-output-artifacts)
8. [Deviations from the paper](#8-deviations-from-the-paper)
9. [Memory and runtime budgets](#9-memory-and-runtime-budgets)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Algorithm overview

### 1.1 Layer collapse via differential weight merging

A *merge operation* `(b, e)` collapses the consecutive layer range `[b..e]` of
the model into a single layer at index `b`, using the LaCo-style differential
weight update applied to every parameter of the base layer:

```
θ*_b  =  θ_b  +  Σ_{k=1..(e-b)} (θ_{b+k} − θ_b)
```

The collapsed layers `[b+1..e]` are then removed from the `ModuleList`, and
`config.num_hidden_layers` is updated. See
[layer_merge.py](evaluator/layer_merge.py): `apply_differential_merge`,
`remove_collapsed_layers`.

### 1.2 Genotype and ψ mapping

For an `L`-layer model (L=32 for Llama-2-7B), each individual is a fixed-length
genome of **`3L = 96` integers**, encoded as `L = 32` consecutive triples:

```
genome  =  [ (b_0, e_0, a_0), (b_1, e_1, a_1), ..., (b_{L-1}, e_{L-1}, a_{L-1}) ]
            with  b_i, e_i ∈ [0, L-1]   and   a_i ∈ {0, 1}
```

Each active triple (`a_i = 1`, `e_i > b_i`) requests a merge of the original
layers `[b_i..e_i]`. Because operations are applied **sequentially** to a
*shrinking* model, the same triple cannot mean the same thing twice — we need
to remap `(b_i, e_i)` through the current compressed indexing each time.

That mapping is **ψ**: an array of length `L` initialized to the identity,
updated after every effective merge so that
`ψ[original_idx] → compressed_idx`. The paper's algorithm:

```python
psi = [0, 1, ..., L-1]                     # identity
for (b, e, a) in genotype_order:           # NOT sorted
    if a != 1 or e <= b: continue
    cb, ce = psi[b], psi[e]                # remap to current compressed indices
    if ce <= cb: continue                  # collapsed by an earlier op → no-op
    apply_differential_merge(model, cb, ce)
    remove_collapsed_layers(model, cb, ce)
    delta = ce - cb
    for j in range(L):
        if   b <= j <= e: psi[j] = cb       # all collapsed-origins point to cb
        elif j > e:       psi[j] = max(0, psi[j] - delta)   # shift right side left
```

This is implemented in [layer_merge.py:`_replay_psi`](evaluator/layer_merge.py) and
[layer_merge.py:`apply_merge_operations`](evaluator/layer_merge.py), and **mirrored
bit-for-bit in C++** in [ecf/Evaluate.cpp:`canonicalize`](ecf/Evaluate.cpp) so
that cache keys derived from the canonical effective-op list agree across the
bridge.


### 1.3 Fitness: module-wise similarity

For each of `N = 64` Wikipedia calibration sentences:

1. Forward both the original and the compressed model.
2. With **forward hooks**, collect activations on every `(layer, submodule)`
   pair:
   - **Attention**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
   - **FFN/MLP**: `gate_proj`, `up_proj`, `down_proj`
   - **Hidden state**: the final hidden state of the model.
3. For each original layer `j`, locate its image `ψ[j]` in the compressed
   model and compute a flattened-cosine similarity between the two activations.
4. Average over the four attention submodules → `attn_sim`; over the three
   FFN submodules → `ffn_sim`; the final-layer hidden state similarity is
   `hs_sim`.
5. The sentence's fitness is `(attn_sim + ffn_sim + hs_sim) / 3`.

The individual's overall **similarity** objective is the mean fitness over all
64 sentences. The **compression** objective is the deterministic ratio
`(L − L′) / L`, where `L′` is the compressed layer count.

Both objectives are in `[0, 1]` and both are **maximized**. See
[fitness.py](evaluator/fitness.py): `compute_fitness`.

### 1.4 NSGA-II search

The C++ ECF driver runs **NSGA-II** on the 96-D real-valued genotype:

- **Population**: 200 individuals
- **Termination**: paper target is **30,000 fitness evaluations** (~150 generations, ~72 h on a single A100). **Not yet executed** in this repo — current results come from a shorter **4,000-evaluation variant** (~20 generations, ~5 h) configured in [ecf/parameters.5h.txt](ecf/parameters.5h.txt). The full 72 h run is planned future work; see §6.1 for both run modes.
- **Crossover**: integer-adapted SBX, probability 1.0, η_c = 20
- **Mutation**: one random gene per individual replaced uniformly within
  bounds — see §8 for the deviation from the paper's polynomial mutation.
- **Selection**: NSGA-II's standard non-dominated sorting + crowding distance.

ECF's `MOFitnessMin` is used with negated objectives (similarity and
compression are both maximized; we pass `-similarity` and `-compression`).
**When reading the milestone files, flip the signs back** to recover the
paper's Fig. 3 orientation.

### 1.5 Canonical caching

Two very different genotypes can decode to the same sequence of effective
merges — e.g. `[(5,7,1)]` and `[(5,7,1), (6,7,1)]` (the second op is a no-op
because layers 6, 7 are already collapsed). Their compressed models are
identical, and so is their fitness. We cache by the **canonical effective-op list** produced by ψ (Caching by the **raw genotype** is bad idea).

Implementation: [ecf/Evaluate.cpp](ecf/Evaluate.cpp):
`canonicalize()` → `cacheKey()` is `"cb1,ce1;cb2,ce2;..."` (empty string for
the identity individual, which is short-circuited without IPC at fitness
`(1.0, 0.0)`).

---

## 2. Architecture

The search runs **two co-resident processes on a single GPU node**
connected over two **named FIFOs** in `/tmp`:

```
                              GELACO_READY_FILE
                       ┌─────── (sentinel) ───────┐
                       │                          │
                       ▼                          │
            ┌────────────────────┐                │
            │  PBS launcher      │  poll until ready
            │  (run_evolution.   │                │
            │   pbs)             │                │
            └─────┬─────────┬────┘                │
                  │ exec    │ background          │
                  ▼         ▼                     │
        ┌─────────────────┐  ┌──────────────────┐ │
        │  ECF (C++)      │  │  server.py       │─┘
        │  NSGA-II        │  │                  │
        │  + ψ cache      │  │  loads Llama-2-7B│
        └────┬────────────┘  │  once, holds it  │
             │  req FIFO     │  in GPU memory   │
             │  JSON triples │                  │
             ├──────────────▶│                  │
             │               │  deepcopy → ψ →  │
             │  rsp FIFO     │  fitness         │
             │  "OK sim comp"│                  │
             │◀──────────────┤                  │
             ▼               └──────────────────┘
        cache key from
        canonical ψ ops
```

**Why this shape:** Llama-2-7B takes ~60 s to load and ~14 GB of VRAM.
30,000 evaluations × 60 s ≈ 500 h just for loading. Holding the model in a
persistent process and streaming evaluation requests over FIFOs reduces
that overhead to a one-time cost.

**Why FIFOs, not sockets/zmq:** the two processes always co-locate on the
same node and need no authentication; FIFOs are dependency-free, line-buffered,
and trivial to clean up at PBS-job teardown.

**Why the READY sentinel file:** the launcher needs an unambiguous signal that
the model is in VRAM AND the FIFOs are ready to be opened. A bare PID check
isn't enough (the process exists during the ~60 s load). The Python side
writes `GELACO_READY_FILE` after `from_pretrained` returns, and the launcher
polls for it before `exec`-ing the C++ driver.

### Wire protocol

Line-oriented, blocking on both sides.

**ECF → server (request):**
```
[[b1,e1,a1],[b2,e2,a2],...,[b32,e32,a32]]\n
```
Or the sentinel:
```
QUIT\n
```

**server → ECF (response):**
```
OK <similarity> <compression>\n        ← happy path; both floats in [0,1]
OK -1.000000 <compression>\n           ← over-aggressive merge → NaN fitness, clamped to worst-case
ERR 0.0 0.0 <single-line error>\n      ← evaluator raised an exception
```

The C++ side treats any non-`OK` tag as a worst-case fitness and continues —
the search is **never aborted** by a single bad individual.

---

## 3. Repository layout

```
re-GeLaCo/
├── README.md                       # this file
├── requirements.txt                # Python deps
│
├── evaluator/                      # Python persistent evaluator
│   ├── server.py                   # main entry: FIFO loop holding Llama-2-7B in VRAM
│   ├── config.py                   # constants + FIFO/READY env-var paths
│   ├── data_loader.py              # 64 Wikipedia calibration sentences (seeded)
│   ├── layer_merge.py              # differential merge + iterative ψ mapping
│   ├── fitness.py                  # hook-based module-wise similarity fitness
│   └── evaluate.py                 # one-shot diagnostic (single hard-coded merge)
│
├── ecf/                            # C++ NSGA-II driver
│   ├── Evaluate.h                  # evaluator class declaration
│   ├── Evaluate.cpp                # decode + ψ + cache + FIFO IPC
│   ├── main.cpp                    # ECF entry point
│   ├── Makefile                    # links against $ECF_ROOT
│   ├── parameters.txt              # full search: 30,000 evals, popSize=200
│   └── parameters.5h.txt           # short variant: 4,000 evals, ~5h walltime
│
└── pbs/                            # Cluster job scripts
    ├── run_evolution.pbs           # full 72h NSGA-II run
    ├── run_evolution.5h.pbs        # 5h variant (uses parameters.5h.txt)
    └── run_prototype.pbs           # one-shot diagnostic via evaluator/evaluate.py
```

### What each Python module does

| File | Public surface | Notes |
|---|---|---|
| [evaluator/config.py](evaluator/config.py) | Module-level constants: `MODEL_NAME`, `NUM_ORIGINAL_LAYERS`, `NUM_CALIBRATION_SENTENCES`, `MAX_SENTENCE_LENGTH`, `RANDOM_SEED`, `FIFO_REQ_PATH`, `FIFO_RSP_PATH`, `READY_FILE`. | FIFO paths read from env (`GELACO_REQ_FIFO`, …) with sensible defaults for interactive testing. |
| [evaluator/data_loader.py](evaluator/data_loader.py) | `load_calibration_sentences()` → `list[str]` | Streams Wikipedia, dedupes, picks 64 sentences with `RANDOM_SEED=42` for reproducibility. Uses a hard-exit workaround on shutdown to avoid PyArrow background-thread segfaults. |
| [evaluator/layer_merge.py](evaluator/layer_merge.py) | `apply_differential_merge`, `remove_collapsed_layers`, `apply_merge_operations`, `canonical_effective_ops`, `_replay_psi` | The paper's ψ logic lives here. The `__main__` block runs four assertions including the overlapping-genotype case. |
| [evaluator/fitness.py](evaluator/fitness.py) | `compute_fitness(..., verbose=True)` | The server passes `verbose=False` to silence the 64-line-per-eval log spam. |
| [evaluator/server.py](evaluator/server.py) | `main()` | Persistent loop: read JSON triples from req FIFO, evaluate, write `OK sim comp` to rsp FIFO. NaN sanitization, VRAM cleanup, `QUIT` shutdown, hard-exit on finalize. |
| [evaluator/evaluate.py](evaluator/evaluate.py) | `main()` | One-shot diagnostic: load model, deepcopy, apply `config.MERGE_OPERATIONS`, compute fitness, print. Useful for sanity-checking the merge math without spinning up NSGA-II. |

### What each C++ file does

| File | Role |
|---|---|
| [ecf/main.cpp](ecf/main.cpp) | ~40-line driver. Constructs `State`, registers `GeLaCoEvaluateOp`, adds a `FloatingPoint::FloatingPoint` genotype, calls `state->initialize(argc, argv)` then `state->run()`. |
| [ecf/Evaluate.h](ecf/Evaluate.h) | `class GeLaCoEvaluateOp : public EvaluateOp`. Constants `L_LAYERS=32`, `N_VARS=96`. Fields: `cache_`, `reqFd_`, `rspFd_`, `rspBuf_`, `evalCount_`, `cacheHits_`. |
| [ecf/Evaluate.cpp](ecf/Evaluate.cpp) | `initialize()` polls READY then opens FIFOs. `evaluate(individual)` does decode → canonicalize → cache lookup → (cache hit ∥ FIFO query) → `MOFitness` with two negated values. Stats every 50 evals to stderr. |
| [ecf/Makefile](ecf/Makefile) | `g++ -std=c++14 -O2 -I$ECF_ROOT/include -L$ECF_ROOT/lib -lecf -lpthread`. Builds `./gelaco_ecf`. |
| [ecf/parameters.txt](ecf/parameters.txt) | ECF XML config. Read in §4.2 below. |

---

## 4. Configuration & parameters

There are three places parameters live, depending on which side owns them.

### 4.1 Python-side ([config.py](evaluator/config.py))

| Constant | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `meta-llama/Llama-2-7b-hf` | HF repo of the base model. |
| `TORCH_DTYPE` | `"float16"` | Inference dtype. |
| `NUM_ORIGINAL_LAYERS` | `32` | Hard constant for Llama-2-7B (`config.num_hidden_layers`). |
| `NUM_CALIBRATION_SENTENCES` | `64` | Paper §4. Reducing this *biases* the fitness — only do it to reduce VRAM if forced. |
| `MAX_SENTENCE_LENGTH` | `128` | Token cap; longer sentences are truncated. |
| `RANDOM_SEED` | `42` | Used by both the dataset sampler and the ECF randomizer (`parameters.txt:randomizer.seed`). |
| `DEVICE` | `"cuda"` | `"cpu"` works for unit tests but not real evaluations. |
| `FIFO_REQ_PATH` / `FIFO_RSP_PATH` / `READY_FILE` | `/tmp/gelaco_{req,rsp,ready}` (overridable via env) | Per-job paths are injected by `run_evolution.pbs` (`/tmp/gelaco_*.$PBS_JOBID`) so concurrent jobs on the same node don't collide. |

### 4.2 ECF-side ([ecf/parameters.txt](ecf/parameters.txt))

XML schema is ECF's. Keys are the **registered names** found in ECF 1.6.1's
`Population.cpp`, `TermMaxEvalOp.cpp`, `Mutation.cpp`, and
`floatingpoint/FloatingPointCrsSbx.cpp` / `FloatingPointMutSimple.cpp`.

```xml
<Algorithm>
    <NSGA2/>               <!-- NSGA-II takes no registered parameters -->
</Algorithm>

<Genotype>
    <FloatingPoint>
        <Entry key="dimension">96</Entry>          <!-- 3 * 32 -->
        <Entry key="lbound">0.0</Entry>            <!-- per-gene lower bound -->
        <Entry key="ubound">31.0</Entry>           <!-- per-gene upper bound -->
        <Entry key="crx.sbx">1.0</Entry>           <!-- SBX probability (must sum to 1 across crx ops) -->
        <Entry key="mut.simple">1.0</Entry>        <!-- mutation operator weight; normalized -->
    </FloatingPoint>
</Genotype>

<Registry>
    <Entry key="randomizer.seed">42</Entry>
    <Entry key="population.size">200</Entry>       <!-- paper §4 -->
    <Entry key="term.eval">30000</Entry>           <!-- paper §4 termination criterion -->
    <Entry key="crx.sbx.ni">20</Entry>             <!-- SBX η_c (uint, in Registry not Genotype) -->
    <Entry key="mutation.indprob">1.0</Entry>      <!-- per-individual mutation probability -->
    <Entry key="batch.repeats">1</Entry>
    <Entry key="log.level">3</Entry>
    <Entry key="log.filename">ecf.log</Entry>
    <Entry key="log.frequency">1</Entry>
    <Entry key="milestone.filename">milestone.txt</Entry>
    <Entry key="milestone.interval">500</Entry>    <!-- Pareto-archive dump every 500 evals -->
</Registry>
```

**The C++ decoder enforces the activation-bit semantics.** ECF's
`FloatingPoint` uses scalar `lbound`/`ubound`, so slot `3i+2` is also
continuous in `[0, 31]`. In [ecf/Evaluate.cpp:`decode`](ecf/Evaluate.cpp) we
threshold at the midpoint:

```cpp
double xa = v[3*i + 2];
int a = (xa >= 0.5 * (L_LAYERS - 1)) ? 1 : 0;
```

This gives a roughly uniform 50/50 split of active vs inactive triples in a
random initial population.

### 4.3 PBS-side ([pbs/run_evolution.pbs](pbs/run_evolution.pbs))

| Directive | Value | Why |
|---|---|---|
| `#PBS -q gpu` | gpu queue | A100 nodes |
| `#PBS -l select=1:ncpus=8:ngpus=1:mem=80GB` | one A100 | enough headroom for fp16 + deepcopy (~28 GB peak) |
| `#PBS -l walltime=72:00:00` | 72 h | ~30 k evals × few s each, minus cache hits |
| `http_proxy / https_proxy` | `http://10.150.1.1:3128` | Supek squid proxy; required for HF download |
| `module load cray-python/3.11.7` | Python 3.11 | matches the prototype |
| `ECF_ROOT` | `$HOME/ecf-install` | resolves `libecf.a` and headers |
| FIFO paths | `/tmp/gelaco_{req,rsp,ready}.$PBS_JOBID` | per-job isolation |

The script (i) `mkfifo`s the two FIFOs, (ii) starts `server.py` in the
background, (iii) polls for `GELACO_READY_FILE` (up to 15 min), (iv) execs
`./ecf/gelaco_ecf ecf/parameters.txt` in the foreground, (v) on
exit/EXIT/INT/TERM writes `QUIT` to the req FIFO and waits up to 30 s before
SIGTERM/SIGKILL, then `rm`s the FIFOs.

---

## 5. One-time cluster setup

Everything below is **idempotent** — re-running it is harmless. Run it once,
then jump to §6 for normal day-to-day usage.

### 5.1 SSH and shell setup

```bash
ssh tsegvic@login-gpu.hpc.srce.hr        # replace tsegvic with your username
```

Supek worker nodes have no internet except via the squid proxy, and Llama-2 is
a gated HuggingFace model. Append these three exports to your `~/.bashrc`:

```bash
cat >> ~/.bashrc <<'EOF'
export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"
export HF_TOKEN="hf_REPLACE_ME_WITH_REAL_TOKEN"
source ~/.bashrc
```

Accept the Llama-2 license at
<https://huggingface.co/meta-llama/Llama-2-7b-hf> before the first run, or
the HF download will 403.

### 5.2 Clone the repository

```bash
cd ~
git clone https://github.com/Tonka12345/re-GeLaCo.git
cd re-GeLaCo
git checkout main
```

### 5.3 Python virtual environment

```bash
module load cray-python/3.11.7
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.4 Build and install ECF from source

ECF (Evolutionary Computation Framework) is not pre-installed on Supek. Build
it once into `$HOME/ecf-install/` on the login node:

```bash
mkdir tmp
mkdir -p ecf-install/lib
mkdir -p ecf-install/include
cd ./tmp
git clone https://github.com/djakobovic/ECF.git
cd ECF
mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS="-fpermissive"
cmake --build . --config Release

cp ./libECF.a $HOME/ecf-install/lib/
cp -r ../ECF $HOME/ecf-install/include/
```

Verify the install:

```bash
ls $HOME/ecf-install/include/ECF/    # expect AlgNSGA2.h, FloatingPoint/, ...
ls $HOME/ecf-install/lib/            # expect libECF.a
```

Export the ECF paths permanently:

```bash
cat >> ~/.bashrc <<'EOF'
export ECF_ROOT=$HOME/ecf-install
export LD_LIBRARY_PATH=$ECF_ROOT/lib:$LD_LIBRARY_PATH
EOF
source ~/.bashrc
```

### 5.5 Build the GeLaCo C++ driver

```bash
cd ~/re-GeLaCo/ecf
make
ls -la gelaco_ecf      # ~few hundred KB executable should appear
```

### 5.6 Sanity-check the Python side

The ψ-mapping unit test runs in ~1 s on the login node (no GPU needed) and
verifies the four cases from §1.2 (single merge, overlapping merges, disjoint
merges, all-inactive):

```bash
cd ~/re-GeLaCo
source venv/bin/activate
python evaluator/layer_merge.py
```

You should see four `ok` lines and `All ψ assertions passed.`. If any assertion
fails, do not proceed — the search will produce incoherent results.

---

## 6. Running the full evolutionary search

Once §5 is done, normal usage is one `qsub` command.

### 6.1 Two run modes

There are two PBS scripts, identical in structure but with different
termination criteria, walltimes, and output filenames so they can coexist
in the same workdir.

| Script | Evals | Generations | Walltime | Output suffix | Status |
|---|---|---|---|---|---|
| [pbs/run_evolution.5h.pbs](pbs/run_evolution.5h.pbs) | 4,000 | ~20 | 5 h | `*-5h.{log,txt}` | **Currently used.** Faster to schedule on shared queues; produces a real Pareto front but at lower convergence than the paper. |
| [pbs/run_evolution.pbs](pbs/run_evolution.pbs) | 30,000 | ~150 | 72 h | `*.{log,txt}` (no suffix) | **Planned future work — not yet executed.** Paper-faithful; needs a 72 h slot in the GPU queue. |

Both share the same population size (200), seed, NSGA-II hyperparameters,
and Python evaluator. The only differences are `term.eval` in the params
file and the log/milestone filenames.

#### 6.1.a Run the 5h variant (current)

```bash
cd ~/re-GeLaCo
qsub pbs/run_evolution.5h.pbs
qstat -u $USER                # watch state Q → R → C
```

Outputs: `server-5h.log`, `ecf-5h.log`, `milestone-5h.txt`, `gelaco-5h.o<JOBID>`.

#### 6.1.b Run the 72h variant (future)

```bash
cd ~/re-GeLaCo
qsub pbs/run_evolution.pbs
qstat -u $USER
```

Outputs: `server.log`, `ecf.log`, `milestone.txt`, `gelaco-evo.o<JOBID>`.

#### What the PBS script does (both variants)

1. Creates per-job FIFOs at `/tmp/gelaco_{req,rsp}.$PBS_JOBID`.
2. Launches [evaluator/server.py](evaluator/server.py) in the background; it loads
   Llama-2-7B (~60–120 s) and 64 calibration sentences, then writes
   `GELACO_READY_FILE`.
3. Polls READY (up to 15 min) and aborts with the server's last 80 log lines
   if the server dies first.
4. Launches `./ecf/gelaco_ecf ecf/parameters[.5h].txt` in the foreground. ECF
   runs NSGA-II, sending each candidate's 32 (b, e, a) triples through the
   req FIFO and reading `OK sim comp` back from the rsp FIFO.
5. On termination (`term.eval` reached, or any signal), the cleanup trap
   writes `QUIT` to the req FIFO with a 5 s timeout, then waits up to 30 s
   for the server to exit before escalating to SIGTERM/SIGKILL, and finally
   removes the FIFOs.

### 6.2 Monitor progress

For the 5h run:

```bash
tail -f ~/re-GeLaCo/server-5h.log         # one line per evaluation
tail -f ~/re-GeLaCo/gelaco-5h.o*          # combined PBS stdout/stderr
tail -f ~/re-GeLaCo/ecf-5h.log            # ECF NSGA-II per-generation stats
```

For the 72h run, drop the `-5h` suffix (`server.log`, `ecf.log`, `gelaco-evo.o*`).

Healthy signs (either variant):
- The server log produces `[server …] eval #N ops=… sim=… comp=… (Xs, alloc=…GB)`
  lines steadily. Some lines will be tagged `[NaN→-1.0]` for over-aggressive
  merges that collapse the model to too few layers — this is expected and
  handled (see §10).
- The PBS .o file shows `[GeLaCo] eval=N hits=K hitRate=…%` lines every 50
  evals. `hitRate` typically grows from ~0% in the first few generations to
  20–60% as the population converges.
- Milestone snapshots appear every 500 evals (so the 5h run with 4,000 evals
  produces ~8 snapshots; the 72h run produces ~60).

### 6.3 (Optional) smoke test before the 72 h run

If you've changed any code that could affect the IPC, the ψ logic, or the
ECF parameter parsing, run a short variant first. Make overrides for
population size and termination, plus a shorter PBS walltime:

```bash
cd ~/re-GeLaCo
cp ecf/parameters.txt ecf/parameters.smoke.txt
sed -i 's|<Entry key="population.size">200</Entry>|<Entry key="population.size">8</Entry>|'  ecf/parameters.smoke.txt
sed -i 's|<Entry key="term.eval">30000</Entry>|<Entry key="term.eval">20</Entry>|'           ecf/parameters.smoke.txt

cp pbs/run_evolution.pbs pbs/run_evolution.smoke.pbs
sed -i 's|walltime=72:00:00|walltime=01:00:00|'                pbs/run_evolution.smoke.pbs
sed -i 's|gelaco-evo|gelaco-smoke|'                            pbs/run_evolution.smoke.pbs
sed -i 's|ecf/parameters.txt|ecf/parameters.smoke.txt|'        pbs/run_evolution.smoke.pbs

qsub pbs/run_evolution.smoke.pbs
```

A successful smoke run completes in a few minutes, produces ~20 well-formed
`OK sim comp` lines with at least one cache hit, and emits zero
`Warning: key ... not registered` lines in `ecf.log`.

### 6.4 Reading the result

The deliverable is the **final Pareto archive** in the milestone file —
`milestone-5h.txt` for the 5h variant, `milestone.txt` for the 72h variant.
ECF uses `MOFitnessMin` with negated objectives, so the two values stored per
individual are `(-similarity, -compression)`. Flip both signs to recover the
paper's Fig. 3 orientation: similarity on the y-axis, compression on the x-axis.

---

## 7. Output artifacts

Filenames depend on which variant was launched. Below, `*` means either
`-5h` (for `pbs/run_evolution.5h.pbs`) or empty string (for the
yet-to-be-run `pbs/run_evolution.pbs`); for the 72h variant the PBS .o
file is `gelaco-evo.o<JOBID>` instead of `gelaco-5h.o<JOBID>`.

| Path | Producer | Contents |
|---|---|---|
| `gelaco-{evo,5h}.o<JOBID>` | PBS | Combined stdout+stderr of the launcher, server, and ECF. |
| `server*.log` | evaluator/server.py | One `[server …] eval #N ops=… sim=… comp=… (…s, alloc=…GB)` line per evaluation. NaN-clamped evals are tagged `[NaN→-1.0]`. |
| `ecf*.log` | ECF | Per-generation `Stats: fitness …` (NSGA-II reports a rank-like scalar here — the real two objectives are in the milestone file). |
| `milestone*.txt` | ECF | XML snapshot of the population and Pareto archive every 500 evals. **Objectives are negated** (`MOFitnessMin`); flip signs to plot in paper-Fig. 3 orientation. |
| stderr of ECF | C++ driver | `[GeLaCo] eval=N hits=K hitRate=…% last=(sim=…, comp=…)` every 50 evals. A healthy run grows `hitRate` from ~0% early to ~20–60% late as the population converges. |

**Post-processing into a Pareto plot:** parse the milestone file, extract the
final generation's individuals' two objectives, negate them, plot `compression`
on x and `similarity` on y. Expected shape: monotone-decreasing similarity as
compression grows (paper Fig. 3). The 5h variant produces a working but
under-converged front (~20 generations); the 72h run is needed to match the
paper's curve quality.

---

## 8. Deviations from the paper

Honest accounting of where ECF 1.6.1's available operator set forced us to
deviate from paper §4. None of these affect the bi-objective NSGA-II logic
itself; they affect only the per-gene variation distribution.

| Paper §4 | This implementation | Reason |
|---|---|---|
| SBX crossover, p = 0.9 | SBX, **p = 1.0** | ECF requires per-genotype crossover probabilities to sum to 1.0, and no other crossover operator was used. |
| SBX η_c = 20 (real) | η_c = 20 (uint) | ECF registers `crx.sbx.ni` as `UINT`. |
| Polynomial mutation, p = 1/96 per gene, η_m = 20 | `mut.simple` (uniform 1-gene replacement) at `mutation.indprob = 1.0` | The polynomial mutation operator is not implemented in ECF 1.6.1 for `FloatingPoint`. The closest available is `mut.simple`, which replaces **one** random gene per call with a uniform draw from `[lbound, ubound]`. With `mutation.indprob = 1.0`, this gives the same expected **count** of mutated genes per individual as the paper (96 × 1/96 = 1), but the perturbation distribution is uniform-global rather than polynomial-local. |

These deviations are documented in the head comment of
[ecf/parameters.txt](ecf/parameters.txt).

---

## 9. Memory and runtime budgets

### Memory (A100 40 GB)

| Item | VRAM |
|---|---|
| Llama-2-7B fp16 weights | ~13 GB |
| Deep copy during evaluation | ~13 GB (peak) |
| Activations + hooks during fwd pass | ~4–6 GB |
| **Peak** | **~28–32 GB** |

[server.py](evaluator/server.py) `del`s the compressed copy and calls
`torch.cuda.empty_cache()` + `gc.collect()` between evaluations, so
steady-state usage drops back to ~14 GB between requests.

### Runtime

| Phase | Cost |
|---|---|
| Model load + sentences | ~60–120 s (one-time, server start) |
| One uncached evaluation | ~3–5 s (forward × 64 sentences × 2 models) |
| One cache hit | ~tens of µs |
| 30,000 evals @ ~30% cache hit | ~17–20 h ECF time + 1 min server start ≈ 20 h (well inside the 72 h wall) |

The 72 h wall is generous; we want headroom for high-compression individuals
(which fail to NaN and are clamped, but still need a forward pass to detect)
and for low cache-hit phases early in the search.

---

## 10. Troubleshooting

### Server hangs at `opening req FIFO ... (blocking)`
Expected. Named-FIFO `open()` blocks until the peer connects. The PBS script
ensures ECF starts only after `GELACO_READY_FILE` is written. When testing
manually outside PBS, use the background-write pattern:
`( echo '[[5,7,1]]' > "$REQ" ) &` before reading the response.

### Server returns `sim=nan`
Over-aggressive merges (e.g. 30/32 layers collapsed) produce numerically
unstable logits and NaN similarities. [server.py](evaluator/server.py) clamps NaN/Inf
to `sim=-1.0` and tags the log line `[NaN→-1.0]`. NSGA-II then dominates these
individuals out without poisoning the front.

### `Warning: key … not registered` in `ecf.log`
The parameter key is not registered in your ECF version. Check the actual
registration in `$ECF_ROOT/include/ECF/*.cpp` via
`grep registerParameter` / `grep registerEntry`. ECF 1.6.1's correct keys are
in §4.2 above.

### `Warning: FloatingPoint crossover operators: cumulative probability not equal to 1`
The sum of `crx.*` probabilities on the FloatingPoint genotype doesn't equal
1.0. With only SBX active, set `crx.sbx = 1.0`.

### CUDA OOM mid-run
The deepcopy briefly peaks at ~28 GB. If you're at the edge, reduce
`NUM_CALIBRATION_SENTENCES` in [config.py](evaluator/config.py) — but note this biases
the fitness (paper uses 64).

### HF download fails on worker node
`http_proxy` / `https_proxy` not exported in the worker shell. Re-export them
after `qsub -I`, or add them to `~/.bashrc` (Supek worker shells inherit it).

### ECF stats show `max=1 min=1 stdev=0`
ECF's per-generation stats reports a single scalar (rank-like) for
multi-objective individuals — the real two objectives live in `milestone.txt`.
A `stdev=0` here doesn't mean the search has collapsed; check the milestone
files instead.

### Server dies before READY
Most often: missing `HF_TOKEN` in the worker shell, expired/invalid token, or
the Llama-2 license hasn't been accepted yet. The PBS script's "ERROR: server
died before becoming ready" path dumps the last 80 lines of `server.log` —
read those first.

---

## Single-evaluation prototype

A one-shot prototype that applies a hard-coded merge and prints one fitness
value lives on the `evaluation-testing` branch. See the `README.md` there for
that workflow. It's useful for sanity-checking the merge math and the
fitness function without spinning up the full NSGA-II pipeline.
