// Count stored Matrix Market entries per row, including symmetry expansion,
// matching read_mtx's treatment of duplicate coordinates and explicit zeros.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: longest_row_ratio MATRIX_LIST DATA_ROOT > results.csv\n";
        return 2;
    }
    std::ifstream list(argv[1]);
    if (!list) { std::cerr << "Cannot open matrix list\n"; return 2; }
    std::cout << "dataset,rows,columns,nnz,longest_row,longest_row_nnz,ratio,status\n";
    std::string name;
    while (std::getline(list, name)) {
        if (!name.empty() && name.back() == '\r') name.pop_back();
        if (name.empty() || name[0] == '#') continue;
        const std::string path = std::string(argv[2]) + "/" + name + "/" + name + ".mtx";
        std::ifstream in(path);
        if (!in) { std::cout << name << ",,,,,,,missing\n"; continue; }
        std::string line, banner, object, format, field, symmetry;
        std::getline(in, line);
        std::istringstream header(line);
        header >> banner >> object >> format >> field >> symmetry;
        auto lower = [](std::string& s) { for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); };
        lower(banner); lower(object); lower(format); lower(field); lower(symmetry);
        bool valid = banner == "%%matrixmarket" && object == "matrix" && format == "coordinate" &&
            (field == "real" || field == "integer" || field == "pattern" || field == "complex") &&
            (symmetry == "general" || symmetry == "symmetric" || symmetry == "skew-symmetric" || symmetry == "hermitian");
        do { if (!std::getline(in, line)) { valid = false; break; } } while (line.empty() || line[0] == '%');
        uint64_t rows = 0, cols = 0, declared = 0;
        std::istringstream dims(line);
        if (!(dims >> rows >> cols >> declared) || (symmetry != "general" && rows != cols)) valid = false;
        if (!valid) { std::cout << name << ",,,,,,,invalid_header\n"; continue; }
        try {
            std::vector<uint64_t> counts(rows, 0);
            uint64_t count = 0, nnz = 0;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '%') continue;
                const char* p = line.c_str(); char* end;
                auto row = std::strtoull(p, &end, 10);
                if (end == p) { valid = false; break; }
                p = end; auto col = std::strtoull(p, &end, 10);
                if (end == p || row == 0 || row > rows || col == 0 || col > cols) { valid = false; break; }
                ++counts[row - 1]; ++nnz; ++count;
                if (symmetry != "general" && row != col) { ++counts[col - 1]; ++nnz; }
            }
            if (!in.eof() || count != declared) valid = false;
            if (!valid) { std::cout << name << ",,,,,,,invalid_data\n"; continue; }
            auto it = std::max_element(counts.begin(), counts.end());
            uint64_t max_count = it == counts.end() ? 0 : *it;
            uint64_t max_row = it == counts.end() ? 0 : static_cast<uint64_t>(it - counts.begin()) + 1;
            std::cout.precision(12);
            std::cout << name << ',' << rows << ',' << cols << ',' << nnz << ',' << max_row << ',' << max_count << ',';
            if (nnz) std::cout << static_cast<double>(max_count) / nnz;
            std::cout << ",ok\n";
        } catch (const std::exception&) { std::cout << name << ",,,,,,,allocation_error\n"; }
        std::cout.flush();
    }
}
