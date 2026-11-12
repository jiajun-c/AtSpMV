// Replay the host-side selector without allocating a GPU Plan.
#include "csrReader.h"
#include "detail/planModelSelection.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 2 && (argc != 3 || (std::string(argv[2]) != "--v2" && std::string(argv[2]) != "--medium")))
            throw std::invalid_argument("usage: plan_model_replay MATRIX.mtx [--v2|--medium]");
        auto csr = read_mtx(argv[1]);
        std::vector<int> rows(csr.row_ptr, csr.row_ptr + csr.nrows + 1);
        const int cols = csr.ncols;
        delete[] csr.row_ptr;
        delete[] csr.col_idx;
        delete[] csr.val;

        std::cout << "mode,selected,backend,tile,balanced,parts,threads,rows_per_group,threshold,width,selected_ratio,best_vector_ratio\n";
        const bool medium = argc == 3 && std::string(argv[2]) == "--medium";
        for (int mode = 0; mode < (medium ? 4 : argc == 3 ? 3 : 2); ++mode) {
            const bool vector = mode != 0;
            atspmv::SpmvPlanOptions options;
            options.vector_model_select = vector;
            options.vector_model_v2 = mode >= 2;
            options.medium_row_model = mode == 3;
            const auto choice = atspmv::detail::choosePlanModel(rows, cols, 1.0, options);
            double selected_ratio = 1.0;
            double best_vector_ratio = std::numeric_limits<double>::infinity();
            for (const auto& prediction : choice.predictions) {
                if (prediction.backend == atspmv::SpmvBackend::CsrVector)
                    best_vector_ratio = std::min(best_vector_ratio, prediction.time_ratio_to_default);
                if (choice.selected && prediction.backend == choice.backend &&
                    prediction.tile_index == choice.tile_index && prediction.balanced == choice.balanced &&
                    prediction.long_rows.parts == choice.long_rows.parts &&
                    prediction.long_rows.threshold == choice.long_rows.threshold &&
                    prediction.vector_width == choice.vector_width)
                    selected_ratio = prediction.time_ratio_to_default;
            }
            std::cout.precision(17);
            std::cout << (mode == 3 ? "medium" : mode == 2 ? "vector_v2" : vector ? "vector" : "base") << ',' << choice.selected << ','
                      << atspmv::spmvBackendName(choice.backend) << ',' << choice.tile_index << ','
                      << choice.balanced << ',' << choice.long_rows.parts << ',' << choice.long_rows.threads << ','
                      << choice.long_rows.rows_per_group << ',' << choice.long_rows.threshold << ','
                      << choice.vector_width << ',' << selected_ratio << ',' << best_vector_ratio << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
