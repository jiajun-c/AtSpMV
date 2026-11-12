// Replay a real application's first implicit CSR, independently of its solver.
#include "spmvPlan.h"
#include "atspmvprod.h"
#include "tunedSpmv.cuh"
#include "detail/planStorage.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cstring>

using atspmv::detail::Buffer;
static cudaStream_t bench_stream;
static void check(cudaError_t e) { if(e!=cudaSuccess)throw atspmv::CudaError(e); }
template<class T> static void upload(Buffer& b,const std::vector<T>& v) {
    b.allocate(v.size()*sizeof(T));
    if(!v.empty())check(cudaMemcpy(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
}

// Diagnostic variants isolate matrix cache policy and full-batch predication.
template<int Depth,bool Stream,bool Full>
__global__ void packedRows(const int* __restrict__ ptr,const int* __restrict__ col,
    const double* __restrict__ val,const int* __restrict__ order,int groups,int rows,
    const double* __restrict__ x,double* __restrict__ y) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(warp>=groups)return;
    int end=ptr[warp+1],tile=ptr[warp];double sums[Depth]={};
    if constexpr(Full) {
        for(;tile+Depth<=end;tile+=Depth) {
            int c[Depth];double a[Depth];
            #pragma unroll
            for(int j=0;j<Depth;++j) {
                int p=(tile+j)*32+lane;
                c[j]=Stream?__ldcg(col+p):col[p];
                a[j]=Stream?__ldcg(val+p):val[p];
            }
            #pragma unroll
            for(int j=0;j<Depth;++j) if(c[j]>=0)sums[j]+=a[j]*x[c[j]];
        }
    }
    for(;tile<end;tile+=Depth) {
        int c[Depth];double a[Depth];
        #pragma unroll
        for(int j=0;j<Depth;++j) {
            int p=(tile+j)*32+lane;
            c[j]=tile+j<end?(Stream?__ldcg(col+p):col[p]):-1;
            a[j]=tile+j<end?(Stream?__ldcg(val+p):val[p]):0.;
        }
        #pragma unroll
        for(int j=0;j<Depth;++j) if(c[j]>=0)sums[j]+=a[j]*x[c[j]];
    }
    double sum=0;
    #pragma unroll
    for(int j=0;j<Depth;++j)sum+=sums[j];
    if(warp*32+lane<rows)y[order[warp*32+lane]]=sum;
}

template<int Depth>
__global__ void compressedRows(const int* __restrict__ ptr,const int2* __restrict__ meta,
    const unsigned char* __restrict__ code,const double* __restrict__ val,
    const int* __restrict__ order,int groups,int rows,const double* __restrict__ x,double* __restrict__ y) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(warp>=groups)return;
    double sums[Depth]={};int end=ptr[warp+1];
    for(int t=ptr[warp];t<end;t+=Depth) {
        int c[Depth];double a[Depth];
        #pragma unroll
        for(int j=0;j<Depth;++j) {
            c[j]=-1;a[j]=0;
            if(t+j<end) {
                int2 m=meta[t+j];const unsigned char* p=code+m.y;
                if(m.x==-1)c[j]=reinterpret_cast<const int*>(p)[lane];
                else if(m.x>=0) {int u=reinterpret_cast<const unsigned short*>(p)[lane];c[j]=u==65535?-1:m.x+u;}
                else {int u=p[lane];c[j]=u==255?-1:-m.x-2+u;}
                a[j]=val[(t+j)*32+lane];
            }
        }
        #pragma unroll
        for(int j=0;j<Depth;++j)if(c[j]>=0)sums[j]+=a[j]*x[c[j]];
    }
    double sum=0;
    #pragma unroll
    for(int j=0;j<Depth;++j)sum+=sums[j];
    if(warp*32+lane<rows)y[order[warp*32+lane]]=sum;
}

template<class Launch> static double measure(Launch launch) {
    for(int i=0;i<4;++i)launch();
    check(cudaDeviceSynchronize());
    cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(bench_stream,cudaStreamCaptureModeGlobal));
    for(int i=0;i<8;++i)launch();
    check(cudaStreamEndCapture(bench_stream,&graph));
    check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
    std::vector<float> times;
    for(int i=0;i<9;++i) {
        check(cudaEventRecord(start,bench_stream));check(cudaGraphLaunch(executable,bench_stream));
        check(cudaEventRecord(stop,bench_stream));check(cudaEventSynchronize(stop));
        float ms;check(cudaEventElapsedTime(&ms,start,stop));times.push_back(ms*1000/8);
    }
    std::sort(times.begin(),times.end());
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
    check(cudaEventDestroy(start));check(cudaEventDestroy(stop));return times[times.size()/2];
}

int main(int argc,char** argv) {try {
    if(argc<2||argc>3||(argc==3&&std::string(argv[2])!="--persist-x"))
        throw std::invalid_argument("usage: packed_spmv_bench implicit.csr [--persist-x]");
    std::ifstream in(argv[1],std::ios::binary);int h[3];in.read(reinterpret_cast<char*>(h),sizeof(h));
    if(!in||h[0]<=0||h[1]<=0||h[2]<0)throw std::runtime_error("invalid CSR header");
    std::vector<int> ptr(size_t(h[0])+1),col(h[2]);std::vector<double> val(h[2]);
    in.read(reinterpret_cast<char*>(ptr.data()),ptr.size()*sizeof(int));
    in.read(reinterpret_cast<char*>(col.data()),col.size()*sizeof(int));
    in.read(reinterpret_cast<char*>(val.data()),val.size()*sizeof(double));
    if(!in||ptr.front()!=0||ptr.back()!=h[2]||!std::is_sorted(ptr.begin(),ptr.end()))throw std::runtime_error("invalid CSR data");
    std::vector<double> x(h[1]),reference(h[0]),absolute(h[0]),got(h[0]);
    for(int i=0;i<h[1];++i)x[i]=std::sin(i*.13)+.1;
    for(int r=0;r<h[0];++r)for(int p=ptr[r];p<ptr[r+1];++p) {
        if(col[p]<0||col[p]>=h[1])throw std::runtime_error("invalid column");
        reference[r]+=val[p]*x[col[p]];absolute[r]+=std::abs(val[p]*x[col[p]]);
    }
    check(cudaStreamCreateWithFlags(&bench_stream,cudaStreamNonBlocking));
    Buffer dp,dc,dv,dx,dy;upload(dp,ptr);upload(dc,col);upload(dv,val);upload(dx,x);upload(dy,got);
    if(argc==3) {
        int device;cudaDeviceProp prop{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
        size_t bytes=x.size()*sizeof(double);
        if(bytes>size_t(prop.persistingL2CacheMaxSize)||bytes>size_t(prop.accessPolicyMaxWindowSize))
            throw std::runtime_error("X exceeds persisting L2 limits");
        check(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,bytes));
        cudaStreamAttrValue attr{};attr.accessPolicyWindow.base_ptr=dx.p;attr.accessPolicyWindow.num_bytes=bytes;
        attr.accessPolicyWindow.hitRatio=1;attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
        check(cudaStreamSetAttribute(bench_stream,cudaStreamAttributeAccessPolicyWindow,&attr));
    }
    CSRMatrix<double,int32_t> csr;csr.nrows=h[0];csr.ncols=h[1];csr.nnz=h[2];
    csr.row_ptr=dp.get<int>();csr.col_idx=dc.get<int>();csr.val=dv.get<double>();
    auto report=[&](const std::string& name,auto launch) {
        launch();check(cudaStreamSynchronize(bench_stream));check(cudaMemcpy(got.data(),dy.p,got.size()*sizeof(double),cudaMemcpyDeviceToHost));
        double error=0;for(int r=0;r<h[0];++r) {
            if(!std::isfinite(got[r]))throw std::runtime_error("nonfinite output");
            error=std::max(error,std::abs(got[r]-reference[r])/std::max(1e-300,absolute[r]));
        }
        if(error>1e-12)throw std::runtime_error("incorrect output: "+name);
        std::cout<<name<<','<<measure(launch)<<','<<error<<std::endl;
    };
    std::cout<<std::setprecision(10)<<"candidate,median_us,max_backward_error\n";
    atspmv::SpmvPlanOptions options;options.backend=atspmv::SpmvBackend::CuSparse;
    auto cu=atspmv::makeSpmvPlan(csr,options);
    report("cusparse",[&]{atspmv::executeSpmv(cu,dx.get<double>(),dy.get<double>(),bench_stream);});
    options.backend=atspmv::SpmvBackend::CsrVector;
    for(int width:{1,2,4,8,16,32}) {
        options.vector_width=width;auto plan=atspmv::makeSpmvPlan(csr,options);
        report("csr-"+std::to_string(width),[&]{atspmv::executeSpmv(plan,dx.get<double>(),dy.get<double>(),bench_stream);});
    }
    atspmv::detail::PackedOwner packed;auto& a=packed.value;
    CSRToAtBSRProduct(csr,a,5,0);
    std::vector<int> ap(a.brow+1),ac(size_t(a.blockCount)*32);
    check(cudaMemcpy(ap.data(),a.blcPtr,ap.size()*sizeof(int),cudaMemcpyDeviceToHost));
    if(!a.useColIdxInt16) {
        check(cudaMemcpy(ac.data(),a.blcIdx,ac.size()*sizeof(int),cudaMemcpyDeviceToHost));
        size_t narrow=0;
        for(int t=0;t<a.blockCount;++t) {
            int lo=h[1],hi=-1;for(int l=0;l<32;++l)if(ac[t*32+l]>=0){lo=std::min(lo,ac[t*32+l]);hi=std::max(hi,ac[t*32+l]);}
            if(hi-lo<65535)++narrow;
        }
        std::cerr<<"rows="<<h[0]<<" nnz="<<h[2]<<" slots="<<ac.size()<<" narrow_tiles="<<narrow<<'/'<<a.blockCount<<'\n';
    }
    for(int warps:{2,4,8})for(int depth:{0,1,2,4,8}) {
        a.kernelConfig.warps_per_block=warps;a.kernelConfig.prefetch=depth;
        report("atbsr-w"+std::to_string(warps)+"-d"+std::to_string(depth),[&]{AtSpmvProduct(a,dx.get<double>(),dy.get<double>(),1,0,bench_stream);});
    }
    if(!a.useColIdxInt16) {
        for(int warps:{2,4,8})for(int policy=0;policy<4;++policy) {
            auto call=[&](auto depth) {
                constexpr int D=decltype(depth)::value;
                int blocks=(a.brow+warps-1)/warps;
                auto launch=[&] {
#define RUN(S,F) packedRows<D,S,F><<<blocks,warps*32,0,bench_stream>>>(a.blcPtr,a.blcIdx,a.blcVal,a.recorder,a.brow,a.nrows,dx.get<double>(),dy.get<double>())
                    if(policy==0){RUN(false,false);}else if(policy==1){RUN(true,false);}
                    else if(policy==2){RUN(false,true);}else {RUN(true,true);}
#undef RUN
                };
                report("probe-w"+std::to_string(warps)+"-d"+std::to_string(D)+"-p"+std::to_string(policy),launch);
            };
            call(std::integral_constant<int,2>{});call(std::integral_constant<int,4>{});call(std::integral_constant<int,8>{});
        }
        for(bool byte:{false,true}) {
            std::vector<int2> meta(a.blockCount);std::vector<unsigned char> code;
            int count8=0,count16=0;
            for(int t=0;t<a.blockCount;++t) {
                int lo=h[1],hi=-1;
                for(int l=0;l<32;++l)if(ac[t*32+l]>=0){lo=std::min(lo,ac[t*32+l]);hi=std::max(hi,ac[t*32+l]);}
                int size=byte&&hi-lo<255?1:hi-lo<65535?2:4;
                int base=size==1?-lo-2:size==2?lo:-1;
                size_t offset=code.size();meta[t]=make_int2(base,int(offset));code.resize(offset+32*size);
                count8+=size==1;count16+=size==2;
                for(int l=0;l<32;++l) {
                    int c=ac[t*32+l];unsigned v=c>=0?unsigned(c-lo):size==1?255:65535;
                    if(size==4)std::memcpy(code.data()+offset+l*size,&c,size);
                    else std::memcpy(code.data()+offset+l*size,&v,size);
                }
            }
            Buffer dm,dc;upload(dm,meta);upload(dc,code);
            std::cerr<<"compressed byte="<<byte<<" bytes="<<code.size()+meta.size()*sizeof(int2)<<" tiles8="<<count8<<" tiles16="<<count16<<'\n';
            for(int warps:{2,4,8}) {
                auto call=[&](auto depth) {
                    constexpr int D=decltype(depth)::value;
                    report("compressed-b"+std::to_string(byte)+"-w"+std::to_string(warps)+"-d"+std::to_string(D),[&] {
                        compressedRows<D><<<(a.brow+warps-1)/warps,warps*32,0,bench_stream>>>(a.blcPtr,dm.get<int2>(),dc.get<unsigned char>(),a.blcVal,a.recorder,a.brow,a.nrows,dx.get<double>(),dy.get<double>());
                    });
                };
                call(std::integral_constant<int,1>{});call(std::integral_constant<int,2>{});call(std::integral_constant<int,4>{});
            }
        }
    }
    report("cusparse-end",[&]{atspmv::executeSpmv(cu,dx.get<double>(),dy.get<double>(),bench_stream);});
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
