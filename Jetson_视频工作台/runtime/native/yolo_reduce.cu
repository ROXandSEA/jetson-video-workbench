/* Lossless per-anchor reduction before host transfer. No score truncation/NMS. */
#include <cuda_runtime.h>
#include <math.h>
extern "C" int yolo_reduce_abi() { return 1; }
__global__ void reduce_channels(const float *p,float *out,int n,int nc) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    bool finite=true;
    for(int c=0;c<4;++c){float v=p[c*n+i];out[c*n+i]=v;finite=finite&&isfinite(v);}
    float score=p[4*n+i];int label=0;
    for(int c=0;c<nc;++c){float v=p[(4+c)*n+i];finite=finite&&isfinite(v);if(v>score){score=v;label=c;}}
    out[4*n+i]=finite?score:-1.0f;out[5*n+i]=(float)label;
}
extern "C" int yolo_reduce(const float *p,float *out,int n,int nc,void *stream) {
    reduce_channels<<<(n+255)/256,256,0,(cudaStream_t)stream>>>(p,out,n,nc);
    return (int)cudaGetLastError();
}
