#include "spmvPlan.h"
#include "csrReader.h"
#include "detail/planStorage.cuh"
#include "kernelConfig.h"
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void cs(cusparseStatus_t s){if(s!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error("cuSPARSE status "+std::to_string(s));}
template<class T>T* upload(const std::vector<T>& v){T* p;ck(cudaMalloc(&p,std::max(size_t(1),v.size())*sizeof(T)));if(!v.empty())ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
struct Candidate {std::string name,config,graph_owner;std::unique_ptr<atspmv::SpmvPlan> plan;std::function<void()> launch;cudaGraph_t graph{};cudaGraphExec_t exec{};cudaEvent_t start{},stop{};std::vector<double> samples;};
// A controlled host packer for the row-order ablation. Both orders use exactly
// the same packing algorithm and production execution kernel; setup is untimed.
std::shared_ptr<atspmv::detail::PackedOwner> packRowOrder(
 const std::vector<int>& p,const std::vector<int>& col,const std::vector<double>& val,
 int columns,int tile,bool sort_rows) {
 auto owner=std::make_shared<atspmv::detail::PackedOwner>();auto& a=owner->value;
 a.nrows=p.size()-1;a.ncols=columns;a.nnz=val.size();a.TILE_M=1<<tile;a.TILE_K=32/a.TILE_M;
 a.useBalance=false;a.useColIdxInt16=columns<32767;a.kernelConfig=atspmv::kernelConfigFromEnvironment();
 a.brow=a.warpCount=(a.nrows+a.TILE_M-1)/a.TILE_M;
 std::vector<int> order(a.nrows),ptr(a.brow+1);std::iota(order.begin(),order.end(),0);
 if(sort_rows)std::stable_sort(order.begin(),order.end(),[&](int x,int y){return p[x+1]-p[x]<p[y+1]-p[y];});
 for(int g=0;g<a.brow;++g) {
  int longest=0;for(int r=g*a.TILE_M;r<std::min(a.nrows,(g+1)*a.TILE_M);++r)longest=std::max(longest,p[order[r]+1]-p[order[r]]);
  long long end=static_cast<long long>(ptr[g])+(longest+a.TILE_K-1)/a.TILE_K;
  if(end>std::numeric_limits<int>::max()/32)throw std::overflow_error("row-order packing too large");
  ptr[g+1]=end;
 }
 a.blockCount=ptr.back();std::vector<int> packed_col(size_t(a.blockCount)*32);std::vector<double> packed_val(packed_col.size());
 for(int g=0;g<a.brow;++g)for(int t=ptr[g];t<ptr[g+1];++t)for(int lane=0;lane<32;++lane) {
  int item=g*a.TILE_M+lane/a.TILE_K,pos=(t-ptr[g])*a.TILE_K+lane%a.TILE_K;
  if(item<a.nrows&&pos<p[order[item]+1]-p[order[item]]) {
   int j=p[order[item]]+pos;packed_col[size_t(t)*32+lane]=col[j];packed_val[size_t(t)*32+lane]=val[j];
  }
 }
 a.recorder=upload(order);a.blcPtr=upload(ptr);a.blcVal=upload(packed_val);
 if(a.useColIdxInt16)a.blcIdxInt16=upload(std::vector<int16_t>(packed_col.begin(),packed_col.end()));
 else a.blcIdx=upload(packed_col);
 return owner;
}
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("usage: spmv_hybrid_validation matrix.mtx configs.csv seed trials");
 unsigned seed=std::stoul(argv[3]);int trials=std::stoi(argv[4]);if(trials<1)throw std::runtime_error("invalid trials");
 auto a=read_mtx(argv[1]);int m=a.nrows,n=a.ncols;std::vector<int> p(a.row_ptr,a.row_ptr+m+1),col(a.col_idx,a.col_idx+a.nnz);std::vector<double> val(a.val,a.val+a.nnz);
 delete[] a.row_ptr;delete[] a.col_idx;delete[] a.val;
 std::vector<long double> references[2],magnitudes[2];
 for(int v=0;v<2;++v) {
  references[v].resize(m);magnitudes[v].resize(m);
  for(int r=0;r<m;++r)for(int j=p[r];j<p[r+1];++j) {
   long double term=(long double)val[j]*(((col[j]*(v?31LL:23LL)+7)%113-56)/67.);
   references[v][r]+=term;magnitudes[v][r]+=std::abs(term);
  }
 }
 CSRMatrix<double,int32_t> d{};d.nrows=m;d.ncols=n;d.nnz=val.size();d.row_ptr=upload(p);d.col_idx=upload(col);d.val=upload(val);
 cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 // Optional zero extension lets the narrow-index ablation declare unused columns.
 std::vector<double> x(std::getenv("ATSPMV_BENCH_PAD_X")?std::max(n,32767):n),y(m);double *dx=upload(x),*dy=upload(y);
 std::vector<Candidate> candidates;std::ifstream input(argv[2]);if(!input)throw std::runtime_error("cannot open configs");std::string line;
 while(std::getline(input,line)){
  if(line.empty()||line[0]=='#')continue;std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);
  std::string kind;int tile,bal,parts,threads,group,threshold;if(!(in>>kind>>tile>>bal>>parts>>threads>>group>>threshold))throw std::runtime_error("bad config");
  if(kind=="atbsr_rows_sorted"||kind=="atbsr_rows_original") {
   if(tile<0||tile>5||bal!=0)throw std::invalid_argument("row-order ablation requires fixed ordinary shape");
   auto packed=packRowOrder(p,col,val,n,tile,kind=="atbsr_rows_sorted");
   Candidate c;c.name=kind;c.config=std::to_string(tile)+"/0/8/128/1/1024";
   std::cerr<<"PLAN_LAYOUT,"<<kind<<','<<m<<','<<n<<','<<d.nnz<<','<<tile<<",0,0,0,0\n";
   std::cerr<<"PLAN_SLOTS,"<<kind<<','<<tile<<','<<size_t(packed->value.blockCount)*32<<'\n';
   c.launch=[=]{AtSpmvProduct(packed->value,dx,dy,1.,0.,stream);};candidates.push_back(std::move(c));continue;
  }
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
  else throw std::runtime_error("unknown backend");
  Candidate c;c.name=kind;c.config=std::to_string(tile)+"/"+std::to_string(bal)+"/"+std::to_string(parts)+"/"+std::to_string(threads)+"/"+std::to_string(group)+"/"+std::to_string(threshold);
  // Benchmark-only overrides are snapshotted by Plan construction, then restored.
  // atbsr_tuning uses the last four config fields as cap/warps/depth/merge.
  std::map<std::string,std::pair<bool,std::string>> saved_env;
  auto override_env=[&](const char* key,int value) {
   const char* old=std::getenv(key);saved_env[key]={old!=nullptr,old?old:""};
   setenv(key,std::to_string(value).c_str(),1);
  };
  if(kind=="atbsr_historical")override_env("ATSPMV_TUNE_PREFETCH",0);
  if(kind=="atbsr_tuning") {
   override_env("ATSPMV_TUNE_CAP",parts);override_env("ATSPMV_TUNE_WARPS",threads);
   override_env("ATSPMV_TUNE_PREFETCH",group);override_env("ATSPMV_TUNE_CTA_REDUCE",threshold);
   o.long_rows={8,128,1,1024};
  }
  auto plan_input=d;
  // Empty extra columns disable automatic int16 storage without changing any
  // actual gather address. Only meaningful for fixed-shape, narrow matrices.
  if(kind=="atbsr_i32") {
   if(tile<0||bal<0||n>=32767)throw std::invalid_argument("int32 ablation requires fixed config and narrow matrix");
   plan_input.ncols=32767;
  }
  c.plan=std::make_unique<atspmv::SpmvPlan>(atspmv::makeSpmvPlan(plan_input,o,stream));
  for(const auto& e:saved_env) {
   if(e.second.first)setenv(e.first.c_str(),e.second.second.c_str(),1);else unsetenv(e.first.c_str());
  }
  const auto& layout=c.plan->info();
  std::cerr<<"PLAN_LAYOUT,"<<kind<<','<<m<<','<<n<<','<<d.nnz<<','<<layout.tile_index<<','<<layout.balanced
           <<','<<layout.tile_columns_grouped<<','<<layout.tile_columns_compressed<<','<<layout.tile_column_bytes<<'\n';
  if(o.retry_shared_storage_on_oom)std::cerr<<"PLAN_OOM_RETRY,"<<kind<<','<<(c.plan->info().oom_retry_shared_storage_used?1:0)<<'\n';
  std::cerr<<std::setprecision(17)<<"PLAN_PREP,"<<kind<<','<<c.plan->info().setup_us<<'\n';
  if(std::getenv("ATSPMV_BENCH_LOG_MODEL_PREDICTIONS")&&c.plan->info().model_used) {
   const auto& info=c.plan->info();
   std::cerr<<std::setprecision(17)<<"MODEL_REFERENCE,"<<kind<<','<<info.model_reference_us<<','<<info.model_reference_calibrated<<'\n';
   for(const auto& p:info.model_predictions)
    std::cerr<<"MODEL_PREDICTION,"<<kind<<','<<atspmv::spmvBackendName(p.backend)<<','<<p.tile_index<<','<<p.balanced<<','<<p.long_rows.parts<<','<<p.long_rows.threads<<','<<p.long_rows.rows_per_group<<','<<p.long_rows.threshold<<','<<p.vector_width<<','<<p.time_ratio_to_default<<','<<p.estimated_us<<'\n';
  }
  if(kind=="auto_model_fallback_tuned")std::cerr<<"MODEL_FALLBACK_POLICY,"<<(c.plan->info().model_fallback_tune_used?"tuned":"model")<<'\n';
  if(kind=="auto_tuned"||kind=="auto_tuned_shared"||kind=="auto_fast_tuned"||kind=="auto_tuned_mixed"||kind=="auto_model_refined"||kind=="auto_model_refined_vector"||(kind=="auto_model_fallback_tuned"&&c.plan->info().model_fallback_tune_used)) {
   const auto& info=c.plan->info();
   std::cerr<<std::setprecision(17)<<"TUNING_BEGIN setup_us="<<info.setup_us<<" selected="<<atspmv::spmvBackendName(info.backend)<<" vector_width="<<info.vector_width<<'\n';
   for(const auto& t:info.timings)std::cerr<<"TUNING,"<<atspmv::spmvBackendName(t.backend)<<','<<t.tile_index<<','<<t.balanced<<','<<t.long_rows.parts<<','<<t.long_rows.threads<<','<<t.long_rows.rows_per_group<<','<<t.long_rows.threshold<<','<<t.vector_width<<','<<t.median_us<<','<<t.mixed_tile_cap<<'\n';
   std::cerr<<"TUNING_END\n";
  }
  if(kind=="atbsr_control") {const auto& info=c.plan->info();c.config="resolved/"+std::to_string(info.tile_index)+"/"+std::to_string(info.balanced);}
  if(kind=="auto_medium_model") {const auto& info=c.plan->info();if(!info.medium_row_model_used)throw std::runtime_error("medium row model not available for input");c.config=std::string(atspmv::spmvBackendName(info.backend))+"/"+std::to_string(info.tile_index)+"/"+std::to_string(info.balanced)+"/"+std::to_string(info.long_rows.parts)+"/"+std::to_string(info.long_rows.threads)+"/"+std::to_string(info.long_rows.rows_per_group)+"/"+std::to_string(info.long_rows.threshold)+"/vw="+std::to_string(info.vector_width);}
  if(kind=="auto_tuned"||kind=="auto_tuned_shared"||kind=="auto_fast_tuned"||kind=="auto_tuned_mixed"||kind=="auto_model_refined"||kind=="auto_model_refined_vector"||kind=="auto_model"||kind=="auto_vector_model"||kind=="auto_vector_model_v2"||kind=="auto_vector_model_graph"||kind=="auto_vector_model_2pct"||kind=="auto_model_fallback_tuned") {const auto& info=c.plan->info();if(kind=="auto_model"&&!info.model_used)throw std::runtime_error("model not available for input");if((kind=="auto_vector_model"||kind=="auto_vector_model_2pct")&&!info.vector_model_used)throw std::runtime_error("vector model not available for input");if(kind=="auto_vector_model_v2"&&!info.vector_model_v2_used)throw std::runtime_error("vector model v2 not available for input");if(kind=="auto_vector_model_graph"&&!info.vector_model_graph_used)throw std::runtime_error("graph vector model not available for input");if(kind=="auto_model_fallback_tuned"&&!info.model_fallback_tune_used&&!info.vector_model_used)throw std::runtime_error("neither model nor fallback tuning active");if(kind=="auto_fast_tuned"&&!info.fast_tuning_used)throw std::runtime_error("fast tuning not active");c.config=std::string(atspmv::spmvBackendName(info.backend))+"/"+std::to_string(info.tile_index)+"/"+std::to_string(info.balanced)+"/"+std::to_string(info.long_rows.parts)+"/"+std::to_string(info.long_rows.threads)+"/"+std::to_string(info.long_rows.rows_per_group)+"/"+std::to_string(info.long_rows.threshold)+"/vw="+std::to_string(info.vector_width);}
  if((kind=="auto_model_refined"||kind=="auto_model_refined_vector")&&!c.plan->info().model_refined)throw std::runtime_error("model refinement not available");
  if(kind=="auto_tuned_mixed"||kind=="auto_model_refined")c.config+="/mc="+std::to_string(c.plan->info().mixed_tile_cap);
  if(kind=="atbsr_clear"&&c.plan->info().balanced)throw std::invalid_argument("clear ablation requires ordinary mode");
  auto ptr=c.plan.get();c.launch=[=]{
   if(kind=="atbsr_clear")ck(cudaMemsetAsync(dy,0,size_t(m)*sizeof(double),stream));
   atspmv::executeSpmv(*ptr,dx,dy,stream);
  };candidates.push_back(std::move(c));
 }
 // Optional controlled comparison: if model decisions resolve to exactly the
 // same backend/config, execute the same Plan instance for both labels. This
 // isolates decision effects from separate device allocation histories.
 if(std::getenv("ATSPMV_BENCH_ALIAS_IDENTICAL_MODELS")) {
  std::map<std::string,std::pair<atspmv::SpmvPlan*,std::string>> canonical;
  for(const char* kind:{"auto_model","auto_vector_model","auto_vector_model_v2","auto_vector_model_graph","auto_vector_model_2pct"})
   for(auto& c:candidates)if(c.name==kind)canonical.emplace(c.config,std::make_pair(c.plan.get(),c.name));
  for(auto& c:candidates)if(c.name=="auto_model"||c.name=="auto_vector_model"||c.name=="auto_vector_model_v2"||c.name=="auto_vector_model_graph"||c.name=="auto_vector_model_2pct") {
   auto* ptr=canonical.at(c.config).first;c.graph_owner=canonical.at(c.config).second;
   c.launch=[=]{atspmv::executeSpmv(*ptr,dx,dy,stream);};
   std::cerr<<"PLAN_ALIAS,"<<c.name<<','<<(ptr==c.plan.get()?"owner":"shared")<<'\n';
  }
 }
 cusparseHandle_t handle;cs(cusparseCreate(&handle));cs(cusparseSetStream(handle,stream));cusparseSpMatDescr_t mat;cusparseDnVecDescr_t vx,vy;
 cs(cusparseCreateCsr(&mat,m,n,d.nnz,d.row_ptr,d.col_idx,d.val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
 cs(cusparseCreateDnVec(&vx,n,dx,CUDA_R_64F));cs(cusparseCreateDnVec(&vy,m,dy,CUDA_R_64F));double alpha=1,beta=0;size_t bytes=0;void* buffer;
 cs(cusparseSpMV_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,&bytes));ck(cudaMalloc(&buffer,std::max(size_t(1),bytes)));
 cs(cusparseSpMV_preprocess(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer));
 Candidate cc;cc.name="cusparse";cc.config="CSR_ALG1";cc.launch=[&]{cs(cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer));};candidates.push_back(std::move(cc));
 if(std::getenv("ATSPMV_BENCH_CUSPARSE_ALGORITHMS"))for(auto alg:{CUSPARSE_SPMV_ALG_DEFAULT,CUSPARSE_SPMV_CSR_ALG2}) {
  struct ExtraCsr {
   cusparseSpMatDescr_t matrix{};void* scratch{};
   ~ExtraCsr(){if(matrix)cusparseDestroySpMat(matrix);if(scratch)cudaFree(scratch);}
  };
  auto extra=std::make_shared<ExtraCsr>();size_t size=0;
  cs(cusparseCreateCsr(&extra->matrix,m,n,d.nnz,d.row_ptr,d.col_idx,d.val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
  cs(cusparseSpMV_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,extra->matrix,vx,&beta,vy,CUDA_R_64F,alg,&size));
  ck(cudaMalloc(&extra->scratch,std::max(size_t(1),size)));
  cs(cusparseSpMV_preprocess(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,extra->matrix,vx,&beta,vy,CUDA_R_64F,alg,extra->scratch));
  Candidate c;c.name=alg==CUSPARSE_SPMV_ALG_DEFAULT?"cusparse_default":"cusparse_alg2";c.config=c.name;
  c.launch=[&,extra,alg]{cs(cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,extra->matrix,vx,&beta,vy,CUDA_R_64F,alg,extra->scratch));};
  candidates.push_back(std::move(c));
 }
 const bool internal_events=std::getenv("ATSPMV_BENCH_INTERNAL_GRAPH_EVENTS")!=nullptr;
 const bool alias_graphs=std::getenv("ATSPMV_BENCH_ALIAS_IDENTICAL_GRAPHS")!=nullptr;
 if(alias_graphs&&!std::getenv("ATSPMV_BENCH_ALIAS_IDENTICAL_MODELS"))throw std::invalid_argument("Graph alias requires Plan alias");
 if(internal_events)std::cerr<<"TIMING_MODE,internal_graph_events\n";
 for(auto& c:candidates){
  for(int v=0;v<2;++v){for(int i=0;i<n;++i)x[i]=((i*(v?31LL:23LL)+7)%113-56)/67.;ck(cudaMemcpyAsync(dx,x.data(),n*sizeof(double),cudaMemcpyHostToDevice,stream));ck(cudaMemsetAsync(dy,0xff,m*sizeof(double),stream));c.launch();ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(y.data(),dy,m*sizeof(double),cudaMemcpyDeviceToHost));
   for(int r=0;r<m;++r){long double sum=references[v][r],mag=magnitudes[v][r];auto tol=8*std::numeric_limits<double>::epsilon()*(p[r+1]-p[r]+32.L)*mag+1e-12L;if(!std::isfinite(y[r])||std::abs(y[r]-sum)>tol)throw std::runtime_error("incorrect "+c.name+" "+c.config+" row="+std::to_string(r));}}
  if(alias_graphs&&!c.graph_owner.empty()&&c.graph_owner!=c.name)continue;
  if(internal_events){ck(cudaEventCreate(&c.start));ck(cudaEventCreate(&c.stop));}
  ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));if(internal_events)ck(cudaEventRecordWithFlags(c.start,stream,cudaEventRecordExternal));for(int i=0;i<8;++i)c.launch();if(internal_events)ck(cudaEventRecordWithFlags(c.stop,stream,cudaEventRecordExternal));ck(cudaStreamEndCapture(stream,&c.graph));ck(cudaGraphInstantiate(&c.exec,c.graph,nullptr,nullptr,0));
  ck(cudaMemsetAsync(dy,0xff,m*sizeof(double),stream));ck(cudaGraphLaunch(c.exec,stream));ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(y.data(),dy,m*sizeof(double),cudaMemcpyDeviceToHost));
  for(int r=0;r<m;++r){long double sum=references[1][r],mag=magnitudes[1][r];auto tol=8*std::numeric_limits<double>::epsilon()*(p[r+1]-p[r]+32.L)*mag+1e-12L;if(!std::isfinite(y[r])||std::abs(y[r]-sum)>tol)throw std::runtime_error("incorrect graph "+c.name+" row="+std::to_string(r));}
 }
 if(alias_graphs)for(auto& c:candidates)if(!c.graph_owner.empty()&&c.graph_owner!=c.name) {
  auto owner=std::find_if(candidates.begin(),candidates.end(),[&](const Candidate& x){return x.name==c.graph_owner;});
  if(owner==candidates.end()||!owner->exec)throw std::logic_error("missing shared Graph owner");
  c.exec=owner->exec;c.start=owner->start;c.stop=owner->stop;
  std::cerr<<"GRAPH_ALIAS,"<<c.name<<','<<c.graph_owner<<'\n';
 }
 cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));std::vector<int> order(candidates.size());std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed);
 for(int t=-3;t<trials;++t){std::shuffle(order.begin(),order.end(),rng);for(int i:order){auto& c=candidates[i];ck(cudaGraphLaunch(c.exec,stream));if(!internal_events)ck(cudaEventRecord(start,stream));ck(cudaGraphLaunch(c.exec,stream));if(!internal_events){ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));}else ck(cudaStreamSynchronize(stream));float ms;ck(cudaEventElapsedTime(&ms,internal_events?c.start:start,internal_events?c.stop:stop));if(t>=0)c.samples.push_back(ms*1000./8);}}
 std::cout<<"backend,config,trial,us\n"<<std::setprecision(10);for(auto& c:candidates)for(int t=0;t<trials;++t)std::cout<<c.name<<','<<c.config<<','<<t<<','<<c.samples[t]<<'\n';
 for(auto& c:candidates){if(alias_graphs&&!c.graph_owner.empty()&&c.graph_owner!=c.name)continue;ck(cudaGraphExecDestroy(c.exec));ck(cudaGraphDestroy(c.graph));if(c.start)ck(cudaEventDestroy(c.start));if(c.stop)ck(cudaEventDestroy(c.stop));}candidates.clear();ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
 cs(cusparseDestroyDnVec(vx));cs(cusparseDestroyDnVec(vy));cs(cusparseDestroySpMat(mat));cs(cusparseDestroy(handle));ck(cudaFree(buffer));ck(cudaFree(dx));ck(cudaFree(dy));ck(cudaFree(d.row_ptr));ck(cudaFree(d.col_idx));ck(cudaFree(d.val));ck(cudaStreamDestroy(stream));
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
