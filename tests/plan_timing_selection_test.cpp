#include "detail/planTimingSelection.h"
#include <stdexcept>
#include <vector>

using namespace atspmv;
void require(bool condition) { if (!condition) throw std::runtime_error("selection gate failed"); }
SpmvCandidateTiming candidate(SpmvBackend backend, double time) {
    SpmvCandidateTiming c; c.backend = backend; c.median_us = time; return c;
}
int main() {
    using atspmv::detail::selectMeasuredCandidate;
    std::vector<SpmvCandidateTiming> c = {candidate(SpmvBackend::AtBsr, 10),
        candidate(SpmvBackend::MixedTile, 9.6), candidate(SpmvBackend::HybridDsmAtBsr, 8)};
    require(selectMeasuredCandidate(c, {true, true, false}, true, .05) == 0);
    require(selectMeasuredCandidate(c, {true, true, true}, true, .05) == 2);
    require(selectMeasuredCandidate(c, {true, true, false}, false, .05) == 2);
    c[1].median_us = 9.5;
    require(selectMeasuredCandidate(c, {true, true, false}, true, .05) == 1);
    c[1].median_us = 10;
    require(selectMeasuredCandidate(c, {true, true, false}, false, .05, 1) == 2);
    c[2].median_us = 11;
    require(selectMeasuredCandidate(c, {true, true, true}, false, .05, 1) == 1);
    require(selectMeasuredCandidate(c, {true, true, true}, true, .05, 1) == 0);
}
