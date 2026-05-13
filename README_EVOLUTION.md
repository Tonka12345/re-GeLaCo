# GeLaCo — Full Evolutionary Search (NSGA-II + ECF) on Supek

This is the deployment guide for the **full GeLaCo bi-objective evolutionary
search** (paper arXiv:2507.10059) on the Supek HPC cluster. It assumes you
have already followed the prototype `README.md` once (you have a working
`venv`, `HF_TOKEN`, proxy in `~/.bashrc`, and the prototype runs end-to-end).

The system has two co-running processes, connected over two named FIFOs:

```
        ┌────────────────┐           req FIFO           ┌──────────────────┐
        │  ECF (C++)     │ ─────── JSON triples ──────▶ │  server.py       │
        │  NSGA-II       │ ◀──── "OK <sim> <comp>" ──── │  Llama-2-7B held │
        │  + ψ-cache     │           rsp FIFO           │  in GPU memory   │
        └────────────────┘                              └──────────────────┘
```

---

## 0. Prerequisites checklist

You should already have on Supek (from the prototype `README.md`):

- [ ] SSH access: `ssh tsegvic@login-gpu.hpc.srce.hr`
- [ ] `~/.bashrc` exports: `http_proxy`, `https_proxy`, `HF_TOKEN`
- [ ] `~/re-GeLaCo` cloned and on the `evaluation-testing` branch
- [ ] `~/re-GeLaCo/venv` created with `requirements.txt` installed
- [ ] One successful run of `python evaluate.py` in an interactive GPU session

If any of those are missing, do them first via `README.md`.

---

## 1. Pull the new code onto Supek

```bash
ssh tsegvic@login-gpu.hpc.srce.hr
cd ~/re-GeLaCo
git fetch origin
git checkout evaluation-testing
git pull
ls ecf/                    # should show: Evaluate.h Evaluate.cpp main.cpp Makefile parameters.txt
ls server.py run_evolution.pbs
```

**Verify the ψ-mapping fix works (no GPU needed, ~1 s on the login node):**

```bash
module load cray-python/3.11.7
source venv/bin/activate
python layer_merge.py
```

Expected output (5 lines, all "ok"):

```
Testing ψ for merge [(5, 7, 1)]:           ok
Testing ψ for overlapping [(5, 7, 1), (6, 7, 1)]: ok
Testing ψ for disjoint [(5, 7, 1), (10, 12, 1)]: ok
Testing ψ with all a=0:                    ok
All ψ assertions passed.
```

✅ **CHECKPOINT 1 — Send me:** the output of `python layer_merge.py`.
If the assertions pass, continue. If they fail, stop and send the failure.

---

## 2. Install ECF (Evolutionary Computation Framework) from source

ECF is not pre-installed on Supek. We build it once into `~/ecf-install/`,
then link our C++ code against it. **Do this on the login node** (no GPU
needed for the build, and we need internet access through the proxy).

### 2.1 Clone and build

```bash
cd ~
# Make sure proxy is set in this shell (the login shell may have it already
# from ~/.bashrc, but be defensive):
export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

mkdir tmp
mkdir ecf-install
cd ./tmp
git clone https://github.com/djakobovic/ECF.git
cd ECF
mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS="-fpermissive"
cmake --build . --config Release

### 2.2 Export environment variables

Add to `~/.bashrc` (one-time):

```bash
echo 'export ECF_ROOT=$HOME/ecf-install' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$ECF_ROOT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

cp ./libECF.a $ECF_ROOT/lib/
cp -r ../ECF $ECF_ROOT/include/

```

### 2.3 Verify the install layout

```bash
ls $ECF_ROOT/include/ECF/        # expect: ECF.h, FloatingPoint/, ...
ls $ECF_ROOT/lib/                # expect: libecf.{a,so}
```

✅ **CHECKPOINT 2 — Send me:**
- Output of `ls $ECF_ROOT/include/ECF/`
- Output of `ls $ECF_ROOT/lib/`
- The last 30 lines of `~/ecf/build/make.log`

If headers/libs are present, continue. If `make` errored out, send the
error and I'll patch the Makefile or guide you through the autotools path.

---

## 3. Build the GeLaCo C++ NSGA-II driver

```bash
cd ~/re-GeLaCo/ecf
make 2>&1 | tee build.log
ls -la gelaco_ecf
```

Expected: `gelaco_ecf` binary appears (~few hundred KB).

**Common breakages I'm not 100% sure about (without the cluster):**
- `MOFitnessMin` / `setObjectives` symbol names may differ slightly between
  ECF versions. If the linker complains about undefined references, **send
  me the `make` error output** and I will adjust [ecf/Evaluate.cpp:227-237](ecf/Evaluate.cpp#L227-L237).
- Missing header `ECF/FloatingPoint/FloatingPoint.h` — **send me the output
  of `find $ECF_ROOT/include -name '*.h' | head -30`** and I'll fix the
  include path in [ecf/Evaluate.h](ecf/Evaluate.h) and [ecf/Evaluate.cpp](ecf/Evaluate.cpp).
- ECF parameter keys (`crx.sbx`, `mut.polynomial`, `popSize`) may differ.
  See §5 below for how to verify.

✅ **CHECKPOINT 3 — Send me:**
- Output of `make 2>&1` (full log).
- `ls -la gelaco_ecf` (or the error if it doesn't exist).

If the build succeeds, continue. Otherwise stop and send the error.

---

## 4. Smoke test the persistent server (no ECF yet)

This proves the Python evaluator side works end-to-end on a GPU node before
we add the C++ driver into the mix.

### 4.1 Open an interactive GPU session

```bash
qsub -I -q gpu -l select=1:ncpus=8:ngpus=1:mem=64GB -l walltime=01:00:00
```

You'll be dropped onto a GPU compute node. **From this shell:**

```bash
# Worker shells don't always source ~/.bashrc — re-export the essentials:
export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"
export HF_TOKEN="hf_YOUR_TOKEN_HERE"     # paste your real token
cd ~/re-GeLaCo
module load cray-python/3.11.7
source venv/bin/activate
nvidia-smi                                # confirm a GPU is visible
```

### 4.2 Create FIFOs and run the server

```bash
REQ=/tmp/gelaco_req.smoke
RSP=/tmp/gelaco_rsp.smoke
READY=/tmp/gelaco_ready.smoke
rm -f "$REQ" "$RSP" "$READY"
mkfifo "$REQ" "$RSP"

export GELACO_REQ_FIFO="$REQ"
export GELACO_RSP_FIFO="$RSP"
export GELACO_READY_FILE="$READY"

# Start the server in background, redirect output to a log:
python server.py > server.smoke.log 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"

# Wait for READY (model load takes 60-120s on first run, less if HF cache hot):
while [ ! -f "$READY" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "SERVER DIED. Last 60 lines:"
        tail -60 server.smoke.log
        break
    fi
    echo "waiting for server..."
    sleep 5
done
echo "server is READY"
```

### 4.3 Send one request, read one response

```bash
# Send a single merge: collapse layers 5-7 (matches the prototype config)
( echo '[[5,7,1]]' > "$REQ" ) &        # background-write so we can read
read line < "$RSP"
echo "RESPONSE: $line"
# Expect: "OK 0.xxxxxx 0.062500"   (compression = 2/32 = 0.0625)
```

### 4.4 Send a no-op (all activations zero) to test the trivial path

```bash
( echo '[[5,7,0]]' > "$REQ" ) &
read line < "$RSP"
echo "RESPONSE: $line"
# Expect: "OK 1.000000 0.000000"   (no merge applied → identical model)
```

### 4.5 Shut down cleanly

```bash
( echo 'QUIT' > "$REQ" ) &
wait $SERVER_PID
rm -f "$REQ" "$RSP" "$READY"
exit              # exit the interactive PBS session
```

✅ **CHECKPOINT 4 — Send me:**
- The two `RESPONSE:` lines from §4.3 and §4.4.
- The full `server.smoke.log` file (most informative is the section right
  after "FIFOs connected" through the second response).

If both responses are well-formed `OK <number> <number>`, the Python side
is healthy. Otherwise send the log and I'll diagnose.

---

## 5. Smoke test the C++ driver (10 evaluations only)

We use a tiny override of `parameters.txt` to validate the IPC + the cache
without committing to a 72 h run. Stay on the login node for this — we're
about to submit a SHORT batch job.

### 5.1 Make a smoke parameters file

```bash
cd ~/re-GeLaCo
cp ecf/parameters.txt ecf/parameters.smoke.txt
# Override popSize and termination for a quick run:
sed -i 's|<Entry key="popSize">200</Entry>|<Entry key="popSize">8</Entry>|'    ecf/parameters.smoke.txt
sed -i 's|<Entry key="term.maxEvaluations">30000</Entry>|<Entry key="term.maxEvaluations">20</Entry>|' ecf/parameters.smoke.txt
```

### 5.2 Make a smoke PBS script

```bash
cp run_evolution.pbs run_evolution.smoke.pbs
sed -i 's|walltime=72:00:00|walltime=01:00:00|'                run_evolution.smoke.pbs
sed -i 's|gelaco-evo|gelaco-smoke|'                            run_evolution.smoke.pbs
sed -i 's|ecf/parameters.txt|ecf/parameters.smoke.txt|'        run_evolution.smoke.pbs
```

### 5.3 Submit and monitor

```bash
qsub run_evolution.smoke.pbs
qstat -u $USER
# Wait for it to start (state R), then tail the logs:
tail -f gelaco-smoke.o* server.log
# Ctrl-C to stop tailing once the job finishes (state C in qstat)
```

### 5.4 Inspect results

```bash
ls -la                                      # look for ecf.log, milestone.txt
tail -50 gelaco-smoke.o*                    # PBS stdout/stderr
tail -100 server.log                        # Python side
tail -50 ecf.log                            # ECF NSGA-II stats
```

You should see in `server.log`: ~20 lines of `[server …] eval #N ops=...
sim=… comp=… (Xs, alloc=…GB)`. You should see in `ecf.log` and on stdout:
NSGA-II generation/evaluation counters and at least a few cache hits as
the population converges.

✅ **CHECKPOINT 5 — Send me:**
- `qstat -u $USER` after submission (to confirm queueing/running).
- After the job finishes: the contents of `gelaco-smoke.o*`, the last
  100 lines of `server.log`, and the last 50 lines of `ecf.log`.
- Any `ERR …` lines from `server.log` (if present).

If the smoke job completes cleanly with sensible fitness values
(similarity in [0, 1], compression in [0, 1]) and at least one cache
hit, **we're ready to commit to the full run.** Otherwise, send the
artifacts and I'll fix.

---

## 6. The full 72 h run

Once the smoke run is green:

```bash
cd ~/re-GeLaCo
qsub run_evolution.pbs
qstat -u $USER
```

Expected total artifacts at end:
- `gelaco-evo.o<JOBID>` — combined PBS stdout/stderr
- `server.log` — Python evaluator log (one line per eval)
- `ecf.log` — NSGA-II per-generation log
- `milestone.txt` (and rotations) — periodic Pareto-archive snapshots

The Pareto archive in `milestone.txt` is the deliverable. The two
objectives there are **negated** (we use `MOFitnessMin` with `-similarity`
and `-compression`); flip signs when plotting (paper Fig. 3 style).

✅ **CHECKPOINT 6 (full run) — Send me:**
- The final `milestone.txt` (or last milestone snapshot).
- Tail of `ecf.log` (last 200 lines) showing the final cache hit rate.
- Tail of `server.log` (last 50 lines).

I'll then help you post-process the Pareto front into a plot.

---

## Reference: file map

| Path | Role |
|---|---|
| [layer_merge.py](layer_merge.py) | Differential merge + paper's iterative ψ |
| [server.py](server.py) | Persistent Python evaluator (FIFO loop) |
| [config.py](config.py) | Constants + FIFO/READY env-var paths |
| [ecf/Evaluate.h](ecf/Evaluate.h) / [ecf/Evaluate.cpp](ecf/Evaluate.cpp) | C++ NSGA-II evaluator (decode, ψ, cache, IPC) |
| [ecf/main.cpp](ecf/main.cpp) | ECF entry point |
| [ecf/Makefile](ecf/Makefile) | Build against `$ECF_ROOT` |
| [ecf/parameters.txt](ecf/parameters.txt) | NSGA-II config (paper §4 verbatim) |
| [run_evolution.pbs](run_evolution.pbs) | PBS job: spawns server + ECF + cleanup |

## Reference: troubleshooting

- **Server hangs at "opening req FIFO ... (blocking)"** — this is normal;
  the FIFO open blocks until the C++ side connects. The PBS script ensures
  ECF starts after READY is written. If you're testing manually outside a
  PBS job, use the `( echo … > "$REQ" ) &` background-write pattern shown
  in §4.3.
- **HF download fails on worker node** — proxy not exported in the worker
  shell. Re-export `http_proxy` / `https_proxy` after the `qsub -I`.
- **CUDA OOM mid-run** — `server.py` should `empty_cache()` between evals.
  If still OOM, the `deepcopy` peak briefly hits ~28 GB; reduce
  `NUM_CALIBRATION_SENTENCES` in [config.py](config.py) (paper uses 64;
  any smaller number reduces VRAM but biases fitness).
- **ECF compile errors on `MOFitnessMin::setObjectives`** — your installed
  ECF version uses a different API. Send me the error and I'll adjust
  [ecf/Evaluate.cpp:227-237](ecf/Evaluate.cpp#L227-L237).
