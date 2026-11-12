#pragma once
#include "spmvPlan.h"
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace atspmv::bench {
// Labels belong to the experiment protocol, not the public library API.
inline SpmvPlanOptions benchmarkOptions(const std::string& kind,int tile,int bal,
                                       int parts,int threads,int group,int threshold) {
  atspmv::SpmvPlanOptions o;o.tile_index=tile;o.balanced=bal;o.long_rows={parts,threads,group,threshold};
  o.share_atbsr_tuning_storage=std::getenv("ATSPMV_BENCH_SHARE_ATBSR_PACKING")!=nullptr;
  o.retry_shared_storage_on_oom=std::getenv("ATSPMV_BENCH_RETRY_SHARED_ON_OOM")!=nullptr;
  if(kind=="atbsr"||kind=="atbsr_control"||kind=="atbsr_m8"||kind=="atbsr_m1_balanced"||kind=="atbsr_m32"||kind=="atbsr_historical"||kind=="atbsr_tuning"||kind=="atbsr_clear"||kind=="atbsr_i32")o.backend=atspmv::SpmvBackend::AtBsr;
  else if(kind=="mixed_tile") {o.backend=atspmv::SpmvBackend::MixedTile;o.mixed_tile_cap=parts;o.long_rows={8,128,1,1024};}
  else if(kind=="atbsr_compressed"||kind=="atbsr_grouped"||kind=="atbsr_grouped_model") {
   o.backend=atspmv::SpmvBackend::AtBsr;
   o.compress_tile_columns=kind=="atbsr_compressed";
   o.group_tile_columns=kind!="atbsr_compressed";
  }
  else if(kind=="csr_dsm"||kind=="csr_dsm_fixed")o.backend=atspmv::SpmvBackend::CsrDsm;
  else if(kind=="csr_two_stage"||kind=="csr_two_stage_fixed")o.backend=atspmv::SpmvBackend::CsrTwoStage;else if(kind=="hybrid_dsm_atbsr")o.backend=atspmv::SpmvBackend::HybridDsmAtBsr;
  else if(kind=="csr_vector"||kind.rfind("csr_vec",0)==0){o.backend=atspmv::SpmvBackend::CsrVector;o.vector_width=parts;o.long_rows={8,128,1,1024};}
  else if(kind.rfind("csr_dsm_vec",0)==0){o.backend=atspmv::SpmvBackend::CsrDsmVector;o.vector_width=std::stoi(kind.substr(11));}
  else if(kind=="auto_tuned"||kind=="auto_tuned_shared"){o.backend=atspmv::SpmvBackend::Auto;o.tune=true;if(kind=="auto_tuned_shared")o.share_atbsr_tuning_storage=true;}
  else if(kind=="auto_fast_tuned"){o.backend=atspmv::SpmvBackend::Auto;o.tune=true;o.fast_tune=true;}
  else if(kind=="auto_tuned_mixed"){o.backend=atspmv::SpmvBackend::Auto;o.tune=true;o.enable_mixed_tile=true;}
  else if(kind=="auto_model_refined"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;o.vector_model_select=true;o.model_refine=true;o.enable_mixed_tile=true;}
  else if(kind=="auto_model_refined_vector"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;o.vector_model_select=true;o.model_refine=true;}
  else if(kind=="auto_vector_model"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;o.vector_model_select=true;}
  else if(kind=="auto_vector_model_v2"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;o.vector_model_select=true;o.vector_model_v2=true;}
  else if(kind=="auto_medium_model"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;o.vector_model_select=true;o.vector_model_v2=true;o.medium_row_model=true;}
  else if(kind=="auto_vector_model_graph"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;o.vector_model_select=true;o.vector_model_graph=true;}
  else if(kind=="auto_vector_model_2pct"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;o.vector_model_select=true;o.vector_model_min_time_reduction=.02;}
  else if(kind=="auto_model_fallback_tuned"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;o.vector_model_select=true;o.model_fallback_tune=true;}
  else if(kind=="auto_model"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;}
  else if(kind=="auto_refined_v2"||kind=="auto_refined_v2_grouped") {
   o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;o.vector_model_select=true;
   o.vector_model_v2=true;o.model_refine=true;o.enable_mixed_tile=true;
   o.group_tile_columns=kind=="auto_refined_v2_grouped";
  }
  else throw std::runtime_error("unknown backend: "+kind);
  return o;
}
} // namespace atspmv::bench
