#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <cmath>
#include <cublas_v2.h>

//C(m,n) = A(m,k) * B(k,n)
void gemm_gpu(const float *A, const float *B, float *C, const int m, const int k, const int n) 
{
    int lda = k, ldb = k, ldc = m;
    const float alf = 1, bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle; cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);
}



int main() {
    int m = 3; 
    int n = 2; 
    int k = 4; 

    std::vector<float> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::vector<float> B = {13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> C(m * n, 0.0);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, A.size() * sizeof(float));
    cudaMalloc((void**)&d_B, B.size() * sizeof(float));
    cudaMalloc((void**)&d_C, C.size() * sizeof(float));

    cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);

    gemm_gpu(d_A, d_B, d_C, m, k, n);

    cudaMemcpy(C.data(), d_C, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    
    std::cout << "output matrix: C:" << std::endl;
    for (int i = 0; i < n*m; ++i) {
        std::cout << C[i] << " ";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
