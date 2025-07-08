#include <mma.h>
#include <vector>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <functional>
#include <cuda_runtime.h> 
#include <algorithm>
#include <cublas_v2.h>
#include <cmath>

using namespace nvcuda;

#define PAD(X,Y) (X % Y ? (X/Y+1)*Y : X)
#define BLOCK_DIM_DEFAULT 512
#define WARP_SIZE 32
#define TIMES 5


template <typename TIN,typename TOUT,int M_TILE,int N_TILE,int K_TILE>
__global__ void wmma_kernel(TIN *a, TIN *b, TOUT *c,int M_PAD,int N_PAD,int K_PAD) {
   int idx,midx,nidx,ndim,kdim;
   ndim = N_PAD / N_TILE;
   kdim = K_PAD / K_TILE;
   idx = (blockIdx.x*blockDim.x+threadIdx.x)/WARP_SIZE;
   nidx = idx%ndim;
   midx = idx/ndim;

   wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, TOUT> c_frag;

   wmma::fill_fragment(c_frag, 0.0f);

   TOUT *c_unique = c + nidx*N_TILE + midx*M_TILE*ndim*N_TILE;

   for(int kidx=0;kidx<kdim;kidx++){

      TIN *a_unique = a + kidx*K_TILE + midx*M_TILE*kdim*K_TILE;
      TIN *b_unique = b + nidx*N_TILE + kidx*K_TILE*ndim*N_TILE;

      wmma::load_matrix_sync(a_frag, a_unique, K_PAD);
      wmma::load_matrix_sync(b_frag, b_unique, N_PAD);

      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   }
   wmma::store_matrix_sync(c_unique, c_frag, N_PAD, wmma::mem_row_major);
}

template <typename T> struct cuda_data {
  T *data;
  cuda_data(size_t n) {
    cudaMallocManaged(&data, sizeof(T) * n);
    for(long i=0;i<n;i++){
      data[i] = 0;
    }
  }
  ~cuda_data() {
    cudaFree(data);
  }
};

enum DIR {ARR2CUARR,CUARR2ARR};

template <typename TARR,typename TCUARR,DIR dir>
void copy(int ARR_M,int ARR_N,TARR *arr,
      int CUARR_M,int CUARR_N,cuda_data<TCUARR> &cuarr){
   assert(CUARR_M>=ARR_M && CUARR_N>=ARR_N);
   if(dir==ARR2CUARR){
      for(int i=0;i<ARR_M;i++)
      for(int j=0;j<ARR_N;j++){
         cuarr.data[i*CUARR_N+j] = arr[i*ARR_N+j];
      }
   }else if(dir==CUARR2ARR){
      for(int i=0;i<ARR_M;i++){
         for(int j=0;j<ARR_N;j++){
            arr[i*ARR_N+j] = cuarr.data[i*CUARR_N+j];
         }
      }
   }else assert(0);
}

void Timer(const char *tag, const std::function<void()> &kernel,int test_time = TIMES) {
  float min_time = 9e99;
  for (int i = 0; i < test_time; ++i) {
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    kernel();
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    min_time = std::min(min_time, elapsed_time);
    std::printf("[%s] iter %d: %f ms elapsed, %f ms min.\n", tag, i,elapsed_time, min_time);
  }
}

float compute_rel_error(const std::vector<float>& wmma_result, const std::vector<float>& cublas_result) {
    float diff_sum = 0.0f;
    float ref_sum = 0.0f;
    for (size_t i = 0; i < wmma_result.size(); ++i) {
        diff_sum += std::abs(wmma_result[i] - cublas_result[i]);
        ref_sum += std::abs(cublas_result[i]);
    }
    return diff_sum / ref_sum;
}

void cublas_gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, 
                d_B, N, 
                d_A, K, 
                &beta, 
                d_C, N);

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

template <typename TIN, typename TOUT,typename TGEMMIN=half, typename TGEMMOUT=float,int M_TILE=16,int N_TILE=16,int K_TILE=16>
void GEMM_wmma(int M,int N,int K,TIN *a_in,TIN *b_in,TOUT *c_out){
   assert(M!=0 && N!=0 && K!=0);

   const int M_PAD = PAD(M,M_TILE) ;
   const int N_PAD = PAD(N,N_TILE) ;
   const int K_PAD = PAD(K,K_TILE) ;

   cuda_data<TGEMMIN> a(M_PAD*K_PAD),b(K_PAD*N_PAD);
   cuda_data<TGEMMOUT> c(M_PAD*N_PAD);

   copy<TIN,TGEMMIN,ARR2CUARR>(M,K,a_in,M_PAD,K_PAD,a);
   copy<TIN,TGEMMIN,ARR2CUARR>(K,N,b_in,K_PAD,N_PAD,b);

   int GRID_DIM,BLOCK_DIM,nwarp;
   nwarp = (M_PAD/M_TILE) * (N_PAD/N_TILE);
   if(nwarp*WARP_SIZE < BLOCK_DIM_DEFAULT){
      GRID_DIM = 1;
      BLOCK_DIM = nwarp*WARP_SIZE;
   }else{
      GRID_DIM = (nwarp*WARP_SIZE)%BLOCK_DIM_DEFAULT ? nwarp*WARP_SIZE/BLOCK_DIM_DEFAULT+1 : nwarp*WARP_SIZE/BLOCK_DIM_DEFAULT ;
      BLOCK_DIM = BLOCK_DIM_DEFAULT;
   }
   printf("GRID_DIM:%d BLOCK_DIM:%d\n",GRID_DIM,BLOCK_DIM);


   Timer("gemm_gty_wmma", [&]{
   wmma_kernel<TGEMMIN,TGEMMOUT,M_TILE,N_TILE,K_TILE>
      <<<GRID_DIM,BLOCK_DIM>>>(a.data,b.data,c.data,
            M_PAD,N_PAD,K_PAD);});

   cudaDeviceSynchronize();
   copy<TOUT,TGEMMOUT,CUARR2ARR>(M,N,c_out,M_PAD,N_PAD,c);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);

    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    std::vector<float> h_c_wmma(M * N);
    std::vector<float> h_c_cublas(M * N);

    for (int i = 0; i < M * K; ++i) {
        h_a[i] = static_cast<float>(i % 10); 
    }
    for (int i = 0; i < K * N; ++i) {
        h_b[i] = static_cast<float>(i % 5); 
    }

    GEMM_wmma<float, float>(M, N, K, h_a.data(), h_b.data(), h_c_wmma.data());

    cublas_gemm(M, N, K, h_a.data(), h_b.data(), h_c_cublas.data());

    float rel_error = compute_rel_error(h_c_wmma, h_c_cublas);
    std::cout << "Relative error between WMMA and cuBLAS results: " << rel_error << std::endl;

    if (rel_error < 0.05f) {
        std::cout << "Validation PASSED (rel_error < 0.05)" << std::endl;
    } else {
        std::cout << "Validation FAILED (rel_error >= 0.05)" << std::endl;
    }

    return 0;
}
