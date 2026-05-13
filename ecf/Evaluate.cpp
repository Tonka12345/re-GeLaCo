// GeLaCo ECF Evaluator — Implementation
// See Evaluate.h for the high-level design notes.
#include "Evaluate.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include <ECF/FloatingPoint/FloatingPoint.h>

namespace {

// ----- env helpers -----------------------------------------------------------
std::string requireEnv(const char* key) {
    const char* v = std::getenv(key);
    if (!v || !*v) {
        throw std::runtime_error(std::string("environment variable ") + key
                                 + " is required but not set");
    }
    return std::string(v);
}

bool fileExists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0;
}

// Read up to and including the next '\n' from `fd`. Blocks until newline
// arrives or peer closes. Returns the line WITHOUT the trailing newline.
// Throws on EOF without data or on read error.
std::string readLine(int fd, std::string& buf) {
    while (true) {
        auto nl = buf.find('\n');
        if (nl != std::string::npos) {
            std::string line = buf.substr(0, nl);
            buf.erase(0, nl + 1);
            return line;
        }
        char chunk[4096];
        ssize_t n = ::read(fd, chunk, sizeof(chunk));
        if (n > 0) {
            buf.append(chunk, static_cast<std::size_t>(n));
        } else if (n == 0) {
            throw std::runtime_error("FIFO peer closed before sending a line");
        } else {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("read() failed: ")
                                     + std::strerror(errno));
        }
    }
}

void writeAll(int fd, const std::string& s) {
    const char* p = s.data();
    std::size_t left = s.size();
    while (left > 0) {
        ssize_t n = ::write(fd, p, left);
        if (n > 0) {
            p += n;
            left -= static_cast<std::size_t>(n);
        } else {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("write() failed: ")
                                     + std::strerror(errno));
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
GeLaCoEvaluateOp::GeLaCoEvaluateOp() = default;

GeLaCoEvaluateOp::~GeLaCoEvaluateOp() {
    if (reqFd_ >= 0) ::close(reqFd_);
    if (rspFd_ >= 0) ::close(rspFd_);
    std::cerr << "[GeLaCo] evaluator destroyed: evals=" << evalCount_
              << " cacheHits=" << cacheHits_ << std::endl;
}

bool GeLaCoEvaluateOp::initialize(StateP state) {
    (void)state;
    const std::string reqPath   = requireEnv("GELACO_REQ_FIFO");
    const std::string rspPath   = requireEnv("GELACO_RSP_FIFO");
    const std::string readyFile = requireEnv("GELACO_READY_FILE");

    std::cerr << "[GeLaCo] waiting for Python server READY at "
              << readyFile << " ..." << std::endl;
    waitForReady(readyFile);
    std::cerr << "[GeLaCo] server is ready; opening FIFOs" << std::endl;
    openFifos(reqPath, rspPath);
    std::cerr << "[GeLaCo] FIFOs connected" << std::endl;
    return true;
}

void GeLaCoEvaluateOp::waitForReady(const std::string& readyFile, int timeoutSec) {
    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::seconds(timeoutSec);
    while (!fileExists(readyFile)) {
        if (clock::now() > deadline) {
            throw std::runtime_error("timed out waiting for server READY: "
                                     + readyFile);
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

void GeLaCoEvaluateOp::openFifos(const std::string& reqPath,
                                 const std::string& rspPath) {
    // IMPORTANT: open in the same order the server opens the OTHER end.
    // server.py opens req for READ first, then rsp for WRITE. We must open
    // req for WRITE first (matches server's read), then rsp for READ.
    reqFd_ = ::open(reqPath.c_str(), O_WRONLY);
    if (reqFd_ < 0) {
        throw std::runtime_error("failed to open req FIFO " + reqPath + ": "
                                 + std::strerror(errno));
    }
    rspFd_ = ::open(rspPath.c_str(), O_RDONLY);
    if (rspFd_ < 0) {
        throw std::runtime_error("failed to open rsp FIFO " + rspPath + ": "
                                 + std::strerror(errno));
    }
}

// ---------------------------------------------------------------------------
std::vector<std::array<int, 3>>
GeLaCoEvaluateOp::decode(IndividualP individual) const {
    // ECF uses boost::shared_ptr typedefs (IndividualP, GenotypeP, ...).
    FloatingPoint::FloatingPoint* gen =
        dynamic_cast<FloatingPoint::FloatingPoint*>(individual->getGenotype().get());
    if (!gen) {
        throw std::runtime_error("genotype is not FloatingPoint::FloatingPoint");
    }
    const auto& v = gen->realValue;
    if (static_cast<int>(v.size()) != N_VARS) {
        throw std::runtime_error("genotype size " + std::to_string(v.size())
                                 + " != expected " + std::to_string(N_VARS));
    }

    auto clampInt = [](double x, int lo, int hi) {
        int i = static_cast<int>(std::floor(x));
        if (i < lo) i = lo;
        if (i > hi) i = hi;
        return i;
    };

    std::vector<std::array<int, 3>> triples;
    triples.reserve(L_LAYERS);
    for (int i = 0; i < L_LAYERS; ++i) {
        int b = clampInt(v[3 * i],     0, L_LAYERS - 1);
        int e = clampInt(v[3 * i + 1], 0, L_LAYERS - 1);
        // Activation: parameters.txt bounds are [0, L-1]; we project to {0,1}
        // by thresholding at the midpoint so the search distributes activations
        // uniformly across the genotype's continuous space.
        double xa = v[3 * i + 2];
        int a = (xa >= 0.5 * static_cast<double>(L_LAYERS - 1)) ? 1 : 0;
        triples.push_back({b, e, a});
    }
    return triples;
}

// Mirrors layer_merge._replay_psi exactly. Keep in lockstep with Python.
std::vector<std::pair<int, int>>
GeLaCoEvaluateOp::canonicalize(
    const std::vector<std::array<int, 3>>& triples) const {

    std::vector<int> psi(L_LAYERS);
    for (int j = 0; j < L_LAYERS; ++j) psi[j] = j;

    std::vector<std::pair<int, int>> eff;
    eff.reserve(triples.size());

    for (const auto& t : triples) {
        int b = t[0], e = t[1], a = t[2];
        if (a != 1 || e <= b || b < 0 || e >= L_LAYERS) continue;
        int cb = psi[b], ce = psi[e];
        if (ce <= cb) continue;  // collapsed by earlier op
        eff.emplace_back(cb, ce);
        int delta = ce - cb;
        for (int j = 0; j < L_LAYERS; ++j) {
            if (j >= b && j <= e) {
                psi[j] = cb;
            } else if (j > e) {
                psi[j] = std::max(0, psi[j] - delta);
            }
        }
    }
    return eff;
}

std::string
GeLaCoEvaluateOp::cacheKey(const std::vector<std::pair<int, int>>& canon) {
    if (canon.empty()) return std::string();  // identity (no merges)
    std::ostringstream os;
    for (std::size_t i = 0; i < canon.size(); ++i) {
        if (i) os << ';';
        os << canon[i].first << ',' << canon[i].second;
    }
    return os.str();
}

std::pair<double, double>
GeLaCoEvaluateOp::queryPython(const std::vector<std::array<int, 3>>& triples) {
    // Build the JSON line: [[b,e,a],[b,e,a],...]
    std::ostringstream js;
    js << '[';
    for (std::size_t i = 0; i < triples.size(); ++i) {
        if (i) js << ',';
        js << '[' << triples[i][0] << ',' << triples[i][1] << ','
           << triples[i][2] << ']';
    }
    js << "]\n";
    writeAll(reqFd_, js.str());

    std::string line = readLine(rspFd_, rspBuf_);
    // Expected: "OK <sim> <comp>"  or  "ERR 0.0 0.0 <msg>"
    std::istringstream is(line);
    std::string tag;
    double a = 0.0, b = 0.0;
    is >> tag >> a >> b;
    if (tag == "OK") {
        return {a, b};
    }
    // Treat any other tag (ERR or garbage) as a worst-case fitness so that
    // NSGA-II discards the individual; do not abort the whole search.
    std::cerr << "[GeLaCo] server returned non-OK: " << line << std::endl;
    return {-1.0, 0.0};
}

// ---------------------------------------------------------------------------
FitnessP GeLaCoEvaluateOp::evaluate(IndividualP individual) {
    ++evalCount_;
    auto triples = decode(individual);
    auto canon   = canonicalize(triples);
    auto key     = cacheKey(canon);

    auto it = cache_.find(key);
    std::pair<double, double> fit;
    if (it != cache_.end()) {
        ++cacheHits_;
        fit = it->second;
    } else {
        if (canon.empty()) {
            // No effective merges → similarity=1, compression=0. Avoid IPC.
            fit = {1.0, 0.0};
        } else {
            fit = queryPython(triples);
        }
        cache_[key] = fit;
    }

    // ECF's NSGA-II uses MOFitnessMin (minimization). Both of our objectives
    // are "more is better" (similarity↑, compression ratio↑), so we feed the
    // negated values; the Pareto-front consumer must un-negate. This is the
    // standard pattern shown in ECF's Simple_NSGA2 example.
    FitnessP out(new MOFitnessMin);
    std::vector<double> objs = { -fit.first, -fit.second };
    out->setObjectives(objs);

    if (evalCount_ % 50 == 0) {
        std::cerr << "[GeLaCo] eval=" << evalCount_
                  << " hits=" << cacheHits_
                  << " hitRate=" << (100.0 * cacheHits_ / evalCount_) << "%"
                  << " last=(sim=" << fit.first << ", comp=" << fit.second << ")"
                  << std::endl;
    }
    return out;
}
