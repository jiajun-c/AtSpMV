// Compare absolute-reference and original host selection without GPU allocation.
#include "csrReader.h"
#include "detail/planModelSelection.h"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("usage: absolute_reference_plan_replay MATRIX.mtx");
        auto csr=read_mtx(argv[1]);
        std::vector<int> rows(csr.row_ptr,csr.row_ptr+csr.nrows+1);
        const int cols=csr.ncols;
        delete[] csr.row_ptr;delete[] csr.col_idx;delete[] csr.val;
        atspmv::SpmvPlanOptions options;options.vector_model_select=true;options.vector_model_v2=true;
        auto baseline=atspmv::detail::choosePlanModel(rows,cols,1.0,options);
        options.absolute_reference_model=true;
        auto learned=atspmv::detail::choosePlanModel(rows,cols,1.0,options);
        const auto& a=baseline.long_rows;const auto& b=learned.long_rows;
        const bool same=baseline.selected==learned.selected&&baseline.backend==learned.backend&&
            baseline.tile_index==learned.tile_index&&baseline.balanced==learned.balanced&&
            baseline.vector_width==learned.vector_width&&a.parts==b.parts&&a.threads==b.threads&&
            a.rows_per_group==b.rows_per_group&&a.threshold==b.threshold;
        std::cout << std::setprecision(17) << "same_selection,reference_us\n" << same << ',' << learned.reference_us << '\n';
    } catch(const std::exception& error) {std::cerr << error.what() << '\n';return 1;}
}
