// GeLaCo NSGA-II driver — entry point.
//
// Reads parameters.txt from argv[1] (or "parameters.txt" if omitted),
// wires up the GeLaCo bi-objective evaluator, and runs the search.
//
// Usage:
//   ./gelaco_ecf [parameters.txt]
//
// Environment (required, set by the PBS job script):
//   GELACO_REQ_FIFO   - path to request FIFO (ECF→Python)
//   GELACO_RSP_FIFO   - path to response FIFO (Python→ECF)
//   GELACO_READY_FILE - path to READY sentinel written by server.py
#include <iostream>
#include <memory>

#include <ECF.h>
#include <floatingpoint/FloatingPoint.h>

#include "Evaluate.h"

int main(int argc, char** argv) {
    try {
        StateP state(new State);

        auto evalOp = static_cast<EvaluateOp*>(new GeLaCoEvaluateOp);
        state->setEvalOp(EvaluateOpP(evalOp));

        // FloatingPoint genotype: parameters.txt declares dimension/bounds.
        state->addGenotype(GenotypeP(new FloatingPoint::FloatingPoint));

        if (!state->initialize(argc, argv)) {
            std::cerr << "ECF state initialize failed" << std::endl;
            return 2;
        }
        state->run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << std::endl;
        return 1;
    }
}
