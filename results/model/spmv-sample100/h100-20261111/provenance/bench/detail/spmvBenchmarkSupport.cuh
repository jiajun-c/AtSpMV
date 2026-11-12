#pragma once

#include "spmvPlan.h"
#include <cusparse.h>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace atspmv::bench {
// Restore temporary kernel settings even if Plan construction throws.
class ScopedEnvironment {
public:
    void set(const char* name, int value) {
        const char* old = std::getenv(name);
        saved_.emplace(name, std::make_pair(old != nullptr, old ? old : ""));
        setenv(name, std::to_string(value).c_str(), 1);
    }
    ~ScopedEnvironment() {
        for (const auto& entry : saved_) {
            if (entry.second.first) setenv(entry.first.c_str(), entry.second.second.c_str(), 1);
            else unsetenv(entry.first.c_str());
        }
    }
    ScopedEnvironment() = default;
    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;
private:
    std::map<std::string,std::pair<bool,std::string>> saved_;
};
inline void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline void cs(cusparseStatus_t s){if(s!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error("cuSPARSE status "+std::to_string(s));}
template<class T>T* upload(const std::vector<T>& v){T* p;ck(cudaMalloc(&p,std::max(size_t(1),v.size())*sizeof(T)));if(!v.empty())ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
struct Candidate {std::string name,config,graph_owner;std::unique_ptr<atspmv::SpmvPlan> plan;std::function<void()> launch;cudaGraph_t graph{};cudaGraphExec_t exec{};cudaEvent_t start{},stop{};std::vector<double> samples;};
} // namespace atspmv::bench
