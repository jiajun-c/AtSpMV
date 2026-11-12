// Extract row-partition features using the same Matrix Market expansion as
// read_mtx. Matrix list contains dataset names, one per line.
#include "hybridModelFeatures.h"
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    if(argc!=3) { std::cerr<<"Usage: hybrid_model_features MATRIX_LIST DATA_ROOT\n"; return 2; }
    std::ifstream list(argv[1]);if(!list)return 2;
    std::cout<<std::setprecision(17)<<"dataset,rows,cols,nnz,threshold,long_rows,short_rows,long_nnz,short_nnz,longest,longest_short,long_length_squared";
    for(int i=0;i<6;++i)std::cout<<",short_tiles_"<<i<<",short_max_tiles_"<<i;
    std::cout<<'\n';
    std::string name;
    while(std::getline(list,name)) {
        if(!name.empty()&&name.back()=='\r')name.pop_back();
        if(name.empty()||name[0]=='#')continue;
        std::ifstream in(std::string(argv[2])+"/"+name+"/"+name+".mtx");
        if(!in) {std::cerr<<name<<": missing\n";continue;}
        std::string line,banner,object,format,field,symmetry;
        std::getline(in,line);std::istringstream header(line);header>>banner>>object>>format>>field>>symmetry;
        for(auto* s:{&banner,&object,&format,&field,&symmetry})for(char& c:*s)c=std::tolower(static_cast<unsigned char>(c));
        bool valid=banner=="%%matrixmarket"&&object=="matrix"&&format=="coordinate"&&
            (field=="real"||field=="integer"||field=="pattern"||field=="complex")&&
            (symmetry=="general"||symmetry=="symmetric"||symmetry=="skew-symmetric"||symmetry=="hermitian");
        do {if(!std::getline(in,line)){valid=false;break;}}while(line.empty()||line[0]=='%');
        long long rows=0,cols=0,declared=0;std::istringstream dims(line);
        if(!(dims>>rows>>cols>>declared)||rows<0||cols<0||rows>INT_MAX||cols>INT_MAX||declared<0||(symmetry!="general"&&rows!=cols))valid=false;
        if(!valid){std::cerr<<name<<": invalid header\n";continue;}
        std::vector<int> counts(rows,0);long long count=0,nnz=0;
        while(std::getline(in,line)) {
            if(line.empty()||line[0]=='%')continue;
            const char* p=line.c_str();char* end;auto row=std::strtoull(p,&end,10);
            if(end==p){valid=false;break;}p=end;auto col=std::strtoull(p,&end,10);
            if(end==p||row==0||row>static_cast<unsigned long long>(rows)||col==0||col>static_cast<unsigned long long>(cols)||nnz>=INT_MAX-1){valid=false;break;}
            ++counts[row-1];++nnz;++count;
            if(symmetry!="general"&&row!=col){++counts[col-1];++nnz;}
        }
        if(!in.eof()||count!=declared||!valid){std::cerr<<name<<": invalid entries\n";continue;}
        std::vector<int> offsets(rows+1,0);
        for(int i=0;i<rows;++i)offsets[i+1]=offsets[i]+counts[i];
        for(int threshold:{128,256,1024,4096}) {
            auto f=atspmv::collectHybridModelFeatures(offsets,cols,threshold);
            std::cout<<name<<','<<f.rows<<','<<f.cols<<','<<f.nnz<<','<<f.threshold<<','<<f.long_rows<<','<<f.short_rows<<','<<f.long_nnz<<','<<f.short_nnz<<','<<f.longest<<','<<f.longest_short<<','<<f.long_length_squared;
            for(int i=0;i<6;++i)std::cout<<','<<f.short_tiles[i]<<','<<f.short_max_tiles[i];
            std::cout<<'\n';
        }
    }
}
