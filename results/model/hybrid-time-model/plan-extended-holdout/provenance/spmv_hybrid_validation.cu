#include "spmvPlan.h"
#include "csrReader.h"
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void cs(cusparseStatus_t s){if(s!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error("cuSPARSE status "+std::to_string(s));}
template<class T>T* upload(const std::vector<T>& v){T* p;ck(cudaMalloc(&p,std::max(size_t(1),v.size())*sizeof(T)));if(!v.empty())ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
struct Candidate {std::string name,config;std::unique_ptr<atspmv::SpmvPlan> plan;std::function<void()> launch;cudaGraph_t graph{};cudaGraphExec_t exec{};std::vector<double> samples;};
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("usage: spmv_hybrid_validation matrix.mtx configs.csv seed trials");
 unsigned seed=std::stoul(argv[3]);int trials=std::stoi(argv[4]);if(trials<1)throw std::runtime_error("invalid trials");
 auto a=read_mtx(argv[1]);int m=a.nrows,n=a.ncols;std::vector<int> p(a.row_ptr,a.row_ptr+m+1),col(a.col_idx,a.col_idx+a.nnz);std::vector<double> val(a.val,a.val+a.nnz);
 delete[] a.row_ptr;delete[] a.col_idx;delete[] a.val;
 CSRMatrix<double,int32_t> d{};d.nrows=m;d.ncols=n;d.nnz=val.size();d.row_ptr=upload(p);d.col_idx=upload(col);d.val=upload(val);
 cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 std::vector<double> x(n),y(m);double *dx=upload(x),*dy=upload(y);
 std::vector<Candidate> candidates;std::ifstream input(argv[2]);if(!input)throw std::runtime_error("cannot open configs");std::string line;
 while(std::getline(input,line)){
  if(line.empty()||line[0]=='#')continue;std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);
  std::string kind;int tile,bal,parts,threads,group,threshold;if(!(in>>kind>>tile>>bal>>parts>>threads>>group>>threshold))throw std::runtime_error("bad config");
  atspmv::SpmvPlanOptions o;o.tile_index=tile;o.balanced=bal;o.long_rows={parts,threads,group,threshold};
  if(kind=="atbsr"||kind=="atbsr_control"||kind=="atbsr_m8"||kind=="atbsr_m1_balanced")o.backend=atspmv::SpmvBackend::AtBsr;else if(kind=="csr_dsm"||kind=="csr_dsm_fixed")o.backend=atspmv::SpmvBackend::CsrDsm;
  else if(kind=="csr_two_stage"||kind=="csr_two_stage_fixed")o.backend=atspmv::SpmvBackend::CsrTwoStage;else if(kind=="hybrid_dsm_atbsr")o.backend=atspmv::SpmvBackend::HybridDsmAtBsr;
  else if(kind=="auto_tuned"){o.backend=atspmv::SpmvBackend::Auto;o.tune=true;}
  else if(kind=="auto_model"){o.backend=atspmv::SpmvBackend::Auto;o.model_select=true;}
  else throw std::runtime_error("unknown backend");
  Candidate c;c.name=kind;c.config=std::to_string(tile)+"/"+std::to_string(bal)+"/"+std::to_string(parts)+"/"+std::to_string(threads)+"/"+std::to_string(group)+"/"+std::to_string(threshold);
  c.plan=std::make_unique<atspmv::SpmvPlan>(atspmv::makeSpmvPlan(d,o,stream));
  if(kind=="atbsr_control") {const auto& info=c.plan->info();c.config="resolved/"+std::to_string(info.tile_index)+"/"+std::to_string(info.balanced);}
  if(kind=="auto_tuned"||kind=="auto_model") {const auto& info=c.plan->info();if(kind=="auto_model"&&!info.model_used)throw std::runtime_error("model not available for input");c.config=std::string(atspmv::spmvBackendName(info.backend))+"/"+std::to_string(info.tile_index)+"/"+std::to_string(info.balanced)+"/"+std::to_string(info.long_rows.parts)+"/"+std::to_string(info.long_rows.threads)+"/"+std::to_string(info.long_rows.rows_per_group)+"/"+std::to_string(info.long_rows.threshold);}
  auto ptr=c.plan.get();c.launch=[=]{atspmv::executeSpmv(*ptr,dx,dy,stream);};candidates.push_back(std::move(c));
 }
 cusparseHandle_t handle;cs(cusparseCreate(&handle));cs(cusparseSetStream(handle,stream));cusparseSpMatDescr_t mat;cusparseDnVecDescr_t vx,vy;
 cs(cusparseCreateCsr(&mat,m,n,d.nnz,d.row_ptr,d.col_idx,d.val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
 cs(cusparseCreateDnVec(&vx,n,dx,CUDA_R_64F));cs(cusparseCreateDnVec(&vy,m,dy,CUDA_R_64F));double alpha=1,beta=0;size_t bytes=0;void* buffer;
 cs(cusparseSpMV_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,&bytes));ck(cudaMalloc(&buffer,std::max(size_t(1),bytes)));
 cs(cusparseSpMV_preprocess(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer));
 Candidate cc;cc.name="cusparse";cc.config="CSR_ALG1";cc.launch=[&]{cs(cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,mat,vx,&beta,vy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG1,buffer));};candidates.push_back(std::move(cc));
 for(auto& c:candidates){
  for(int v=0;v<2;++v){for(int i=0;i<n;++i)x[i]=((i*(v?31LL:23LL)+7)%113-56)/67.;ck(cudaMemcpyAsync(dx,x.data(),n*sizeof(double),cudaMemcpyHostToDevice,stream));ck(cudaMemsetAsync(dy,0xff,m*sizeof(double),stream));c.launch();ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(y.data(),dy,m*sizeof(double),cudaMemcpyDeviceToHost));
   for(int r=0;r<m;++r){long double sum=0,mag=0;for(int j=p[r];j<p[r+1];++j){long double v=(long double)val[j]*x[col[j]];sum+=v;mag+=std::abs(v);}auto tol=8*std::numeric_limits<double>::epsilon()*(p[r+1]-p[r]+32.L)*mag+1e-12L;if(!std::isfinite(y[r])||std::abs(y[r]-sum)>tol)throw std::runtime_error("incorrect "+c.name+" "+c.config+" row="+std::to_string(r));}}
  ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int i=0;i<8;++i)c.launch();ck(cudaStreamEndCapture(stream,&c.graph));ck(cudaGraphInstantiate(&c.exec,c.graph,nullptr,nullptr,0));
  ck(cudaMemsetAsync(dy,0xff,m*sizeof(double),stream));ck(cudaGraphLaunch(c.exec,stream));ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(y.data(),dy,m*sizeof(double),cudaMemcpyDeviceToHost));
  for(int r=0;r<m;++r){long double sum=0,mag=0;for(int j=p[r];j<p[r+1];++j){long double v=(long double)val[j]*x[col[j]];sum+=v;mag+=std::abs(v);}auto tol=8*std::numeric_limits<double>::epsilon()*(p[r+1]-p[r]+32.L)*mag+1e-12L;if(!std::isfinite(y[r])||std::abs(y[r]-sum)>tol)throw std::runtime_error("incorrect graph "+c.name+" row="+std::to_string(r));}
 }
 cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));std::vector<int> order(candidates.size());std::iota(order.begin(),order.end(),0);std::mt19937 rng(seed);
 for(int t=-3;t<trials;++t){std::shuffle(order.begin(),order.end(),rng);for(int i:order){auto& c=candidates[i];ck(cudaGraphLaunch(c.exec,stream));ck(cudaEventRecord(start,stream));ck(cudaGraphLaunch(c.exec,stream));ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));if(t>=0)c.samples.push_back(ms*1000./8);}}
 std::cout<<"backend,config,trial,us\n"<<std::setprecision(10);for(auto& c:candidates)for(int t=0;t<trials;++t)std::cout<<c.name<<','<<c.config<<','<<t<<','<<c.samples[t]<<'\n';
 for(auto& c:candidates){ck(cudaGraphExecDestroy(c.exec));ck(cudaGraphDestroy(c.graph));}candidates.clear();ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
 cs(cusparseDestroyDnVec(vx));cs(cusparseDestroyDnVec(vy));cs(cusparseDestroySpMat(mat));cs(cusparseDestroy(handle));ck(cudaFree(buffer));ck(cudaFree(dx));ck(cudaFree(dy));ck(cudaFree(d.row_ptr));ck(cudaFree(d.col_idx));ck(cudaFree(d.val));ck(cudaStreamDestroy(stream));
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
