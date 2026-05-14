// GeLaCo ECF Evaluator — Header
// ---------------------------------------------------------------------------
// Bi-objective evaluator for the GeLaCo NSGA-II search (paper §3.1/§3.3).
//
//   - Genotype: FloatingPoint vector of size 3*L_LAYERS (= 96 for Llama-2-7B).
//     Each consecutive triple (x[3i], x[3i+1], x[3i+2]) decodes to (b, e, a).
//   - Objectives (both maximized via NSGA-II's standard maxFitness wrapper):
//       f1 = module-wise similarity in [0, 1]
//       f2 = compression ratio = layers_removed / L_LAYERS, in [0, 1]
//   - Cache key: canonical effective (compressed_b, compressed_e) list after
//     applying the paper's iterative ψ mapping — many genotypes alias to the
//     same compressed model and must share a cache entry (paper §3.3).
//   - IPC: two named FIFOs to the Python persistent evaluator (server.py).
//     Paths from env vars GELACO_REQ_FIFO / GELACO_RSP_FIFO; readiness is
//     signaled by GELACO_READY_FILE which we poll in initialize().
//
// Author note: the FloatingPoint genotype in ECF uses scalar lbound/ubound
// rather than per-dim bounds. We compensate in decode() by interpreting slot
// 3i+2 as the activation bit (clamped to {0,1}) and slots 3i, 3i+1 as the
// layer indices in [0, L_LAYERS-1]. parameters.txt must set lbound=0,
// ubound=L_LAYERS-1.
// ---------------------------------------------------------------------------
#ifndef GELACO_EVALUATE_H
#define GELACO_EVALUATE_H

#include <array>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ECF.h>

class GeLaCoEvaluateOp : public EvaluateOp {
public:
    static constexpr int L_LAYERS = 32;        // Llama-2-7B
    static constexpr int N_VARS   = 3 * L_LAYERS;

    GeLaCoEvaluateOp();
    ~GeLaCoEvaluateOp() override;

    bool initialize(StateP state) override;
    FitnessP evaluate(IndividualP individual) override;

private:
    // --- Decoding & canonicalization --------------------------------------
    // Decode 96 floats into 32 (b, e, a) triples, in genotype order.
    std::vector<std::array<int, 3>> decode(IndividualP individual) const;

    // Replay the paper's ψ over the decoded triples; return the list of
    // (compressed_b, compressed_e) pairs that would actually be applied.
    // Mirrors layer_merge._replay_psi in Python — keep the two in lockstep.
    std::vector<std::pair<int, int>> canonicalize(
        const std::vector<std::array<int, 3>>& triples) const;

    // Serialize a canonical op list into a deterministic cache key.
    static std::string cacheKey(const std::vector<std::pair<int, int>>& canon);

    // --- IPC with Python server -------------------------------------------
    void waitForReady(const std::string& readyFile, int timeoutSec = 1800);
    void openFifos(const std::string& reqPath, const std::string& rspPath);
    std::pair<double, double> queryPython(
        const std::vector<std::array<int, 3>>& triples);

    // --- State ------------------------------------------------------------
    std::unordered_map<std::string, std::pair<double, double>> cache_;
    int reqFd_ = -1;   // we write to this FIFO
    int rspFd_ = -1;   // we read from this FIFO
    std::string rspBuf_;  // line-buffered read state
    std::size_t evalCount_   = 0;
    std::size_t cacheHits_   = 0;
};

#endif  // GELACO_EVALUATE_H
