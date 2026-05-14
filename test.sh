cd $ECF_ROOT/include

echo "=== NSGA2 keys ==="
grep -n 'registerParameter' AlgNSGA2.cpp

echo ""
echo "=== Termination (max evals) keys ==="
grep -rn 'registerParameter' TermMaxEvalOp.cpp TermMaxGenOp.cpp 2>/dev/null

echo ""
echo "=== FloatingPoint genotype + its crossover/mutation operators ==="
grep -rn 'registerParameter' floatingpoint/ 2>/dev/null

echo ""
echo "=== Population sizing (registry vs algorithm) ==="
grep -rn 'registerParameter.*[Pp]op' . 2>/dev/null | head -20

echo ""
echo "=== Existing example parameters.txt files (so we have a known-good template) ==="
find examples -name 'parameters*.txt' -o -name '*.txt' 2>/dev/null | head -20
