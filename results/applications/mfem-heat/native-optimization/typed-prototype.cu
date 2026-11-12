// Replay a real application's first implicit CSR, independently of its solver.
#include "spmvPlan.h"
#include "atspmvprod.h"
#include "detail/planStorage.cuh"
#include "detail/compressedTileColumns.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using atspmv::detail::Buffer;
static cudaStream_t bench_stream;
static void check(cudaError_t e) { if(e!=cudaSuccess)throw atspmv::CudaError(e); }
template<class T> static void upload(Buffer& b,const std::vector<T>& v) {
    b.allocate(v.size()*sizeof(T));
    if(!v.empty())check(cudaMemcpy(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
}

template<int Length,int Depth>
__device__ __forceinline__ double decodeRow(const int2* __restrict__ meta,
    const unsigned char* __restrict__ code,const double* __restrict__ values,
    const double* __restrict__ x,int first,int length,int lane) {
    double sum[Depth]={};
    const int count=Length?Length:length;
    #pragma unroll
    for(int t=0;t<count;t+=Depth) {
        int cols[Depth];double a[Depth];
        #pragma unroll
        for(int j=0;j<Depth;++j) {
            cols[j]=0;a[j]=0;
            if(t+j<count) {
                int2 m=meta[first+t+j];const auto* p=code+m.y;
                if(m.x==-1)cols[j]=reinterpret_cast<const int*>(p)[lane];
                else if(m.x>=0)cols[j]=m.x+reinterpret_cast<const unsigned short*>(p)[lane];
                else cols[j]=-(m.x+2)+p[lane];
                a[j]=values[(first+t+j)*32+lane];
            }
        }
        #pragma unroll
        for(int j=0;j<Depth;++j) {
            double b=t+j<count?x[cols[j]]:0.;
            sum[j]+=a[j]*b;
        }
    }
    double total=0;
    #pragma unroll
    for(int j=0;j<Depth;++j)total+=sum[j];
    return total;
}
template<int Depth,bool Static>
__global__ void decodeRows(const int* __restrict__ ptr,const int2* __restrict__ meta,
    const unsigned char* __restrict__ code,const double* __restrict__ val,
    const int* __restrict__ order,int groups,int rows,const double* __restrict__ x,double* __restrict__ y) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(warp>=groups)return;
    int first=ptr[warp],length=ptr[warp+1]-first;double sum;
    if constexpr(Static) {
        if(length==9)sum=decodeRow<9,Depth>(meta,code,val,x,first,length,lane);
        else if(length==15)sum=decodeRow<15,Depth>(meta,code,val,x,first,length,lane);
        else if(length==25)sum=decodeRow<25,Depth>(meta,code,val,x,first,length,lane);
        else sum=decodeRow<0,Depth>(meta,code,val,x,first,length,lane);
    } else sum=decodeRow<0,Depth>(meta,code,val,x,first,length,lane);
    if(warp*32+lane<rows)y[order[warp*32+lane]]=sum;
}

template<class Index,int Depth,bool Base>
__device__ __forceinline__ double typedSum(int2 range,const Index* __restrict__ columns,
    const int* __restrict__ bases,const double* __restrict__ values,const double* __restrict__ x,int lane) {
    double accum[Depth]={};
    for(int tile=range.x;tile<range.y;tile+=Depth) {
        int c[Depth];double a[Depth];
        #pragma unroll
        for(int j=0;j<Depth;++j) {
            bool active=tile+j<range.y;
            c[j]=active?int(columns[(tile+j)*32+lane])+(Base?bases[tile+j]:0):0;
            a[j]=active?values[(tile+j)*32+lane]:0.;
        }
        #pragma unroll
        for(int j=0;j<Depth;++j)accum[j]+=a[j]*(tile+j<range.y?x[c[j]]:0.);
    }
    double sum=0;
    #pragma unroll
    for(int j=0;j<Depth;++j)sum+=accum[j];
    return sum;
}
template<int Depth>
__global__ void typedRows(const int2* __restrict__ ranges,const unsigned char* c8,const int* b8,const double* a8,
    const unsigned short* c16,const int* b16,const double* a16,const int* c32,const double* a32,
    const int* __restrict__ order,int rows,const double* __restrict__ x,double* __restrict__ y) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(warp*32>=rows)return;
    double sum=typedSum<unsigned char,Depth,true>(ranges[warp*3],c8,b8,a8,x,lane);
    sum+=typedSum<unsigned short,Depth,true>(ranges[warp*3+1],c16,b16,a16,x,lane);
    sum+=typedSum<int,Depth,false>(ranges[warp*3+2],c32,nullptr,a32,x,lane);
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
    options.backend=atspmv::SpmvBackend::AtBsr;
    options.tile_index=5;options.balanced=0;options.compress_tile_columns=true;
    for(int warps:{2,4,8}) {
        setenv("ATSPMV_TUNE_WARPS",std::to_string(warps).c_str(),1);
        auto plan=atspmv::makeSpmvPlan(csr,options,bench_stream);
        std::cerr<<"compressed column bytes="<<plan.info().tile_column_bytes<<'\n';
        report("compressed-w"+std::to_string(warps),[&]{atspmv::executeSpmv(plan,dx.get<double>(),dy.get<double>(),bench_stream);});
    }
    auto compressed=atspmv::detail::compressTileColumns(a,bench_stream);
    if(compressed)for(int warps:{2,4,8}) {
        auto run=[&](auto depth,auto stat) {
            constexpr int D=decltype(depth)::value;constexpr bool S=decltype(stat)::value;
            report("decode-w"+std::to_string(warps)+"-d"+std::to_string(D)+"-s"+std::to_string(S),[&]{
                decodeRows<D,S><<<(a.brow+warps-1)/warps,warps*32,0,bench_stream>>>(a.blcPtr,
                    compressed->meta.get<int2>(),compressed->code.get<unsigned char>(),a.blcVal,a.recorder,a.brow,a.nrows,dx.get<double>(),dy.get<double>());
            });
        };
        run(std::integral_constant<int,2>{},std::false_type{});
        run(std::integral_constant<int,4>{},std::false_type{});
        run(std::integral_constant<int,2>{},std::true_type{});
        run(std::integral_constant<int,4>{},std::true_type{});
    }
    if(!a.useColIdxInt16) {
        std::vector<int> order(a.nrows),base(a.blockCount),kind(a.blockCount);
        check(cudaMemcpy(order.data(),a.recorder,order.size()*sizeof(int),cudaMemcpyDeviceToHost));
        for(int t=0;t<a.blockCount;++t) {
            int lo=h[1],hi=0;
            for(int l=0;l<32;++l){lo=std::min(lo,ac[t*32+l]);hi=std::max(hi,ac[t*32+l]);}
            base[t]=lo;kind[t]=hi-lo<255?0:hi-lo<65535?1:2;
        }
        std::vector<int2> ranges(a.brow*3);
        std::vector<unsigned char> c8;std::vector<unsigned short> c16;std::vector<int> c32,bases[2];
        std::vector<double> vals[3];
        for(int g=0;g<a.brow;++g)for(int k=0;k<3;++k) {
            int begin=vals[k].size()/32;
            for(int t=ap[g];t<ap[g+1];++t)if(kind[t]==k) {
                if(k<2)bases[k].push_back(base[t]);
                for(int lane=0;lane<32;++lane) {
                    int c=ac[t*32+lane];
                    if(k==0)c8.push_back(c-base[t]);else if(k==1)c16.push_back(c-base[t]);else c32.push_back(c);
                    int item=g*32+lane,row=item<a.nrows?order[item]:-1,j=t-ap[g];
                    vals[k].push_back(row>=0&&j<ptr[row+1]-ptr[row]?val[ptr[row]+j]:0.);
                }
            }
            ranges[g*3+k]=make_int2(begin,vals[k].size()/32);
        }
        Buffer dr,dc8,dc16,dc32,db8,db16,da8,da16,da32;
        upload(dr,ranges);upload(dc8,c8);upload(dc16,c16);upload(dc32,c32);upload(db8,bases[0]);upload(db16,bases[1]);
        upload(da8,vals[0]);upload(da16,vals[1]);upload(da32,vals[2]);
        for(int warps:{2,4,8}) {
            auto run=[&](auto depth) {
                constexpr int D=decltype(depth)::value;
                report("typed-w"+std::to_string(warps)+"-d"+std::to_string(D),[&]{
                    typedRows<D><<<(a.brow+warps-1)/warps,warps*32,0,bench_stream>>>(dr.get<int2>(),dc8.get<unsigned char>(),db8.get<int>(),da8.get<double>(),
                        dc16.get<unsigned short>(),db16.get<int>(),da16.get<double>(),dc32.get<int>(),da32.get<double>(),a.recorder,a.nrows,dx.get<double>(),dy.get<double>());
                });
            };
            run(std::integral_constant<int,1>{});run(std::integral_constant<int,2>{});run(std::integral_constant<int,4>{});run(std::integral_constant<int,8>{});
        }
    }
    report("cusparse-end",[&]{atspmv::executeSpmv(cu,dx.get<double>(),dy.get<double>(),bench_stream);});
    check(cudaStreamDestroy(bench_stream));
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
