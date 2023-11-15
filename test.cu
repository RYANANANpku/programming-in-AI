#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <cmath>
#include <cublas_v2.h>
#include <curand.h>

// Use 512 threads per block
const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N) {
   return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDA_KERNEL_LOOP(i, n)                         \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;  \
       i < (n);                                        \
       i += blockDim.x * gridDim.x)

/* in this file, we are going to test the correctness of our functions */

/* matrix generator,we use curand to generate a matrix */
void matrix_init(float*A, int rows, int cols){
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, rows * cols);
    curandDestroyGenerator(prng);
}

/* first: fully connected layer */

/* before the impletation of fully connected layer,we have to make some assumptions
   we assume that the input X is in_features * batch_size
   we assume that the weight W is out_features * in_features
   we assume that the output Y is out_features * batch_size
*/
//C(m,n) = A(m,k) * B(k,n)
void gemm_gpu(cublasOperation_t op1, cublasOperation_t op2, const float *A, const float *B, float *C, const int m, const int k, const int n, float p1, float p2) 
{
    int lda,ldb,ldc = m;
    if(op1 == CUBLAS_OP_N) lda = m;
    else lda = k;
    if(op2 == CUBLAS_OP_N) ldb = k;
    else ldb = n;
    const float alf = p1, bet = p2;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle; cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, op1, op2, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);
}

void forward_fc(float* input, float* output, float* weights, float* bias, int batch_size, int in_features, int out_features) 
{
    // matrix product with gemm
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, weights, input, output, out_features, in_features, batch_size, 1.0 , 0.0);
    
    // add bias
    std::vector<float> _ones(batch_size, 1.0);
    float *d_ones;
    cudaMalloc((void**)&d_ones, _ones.size() * sizeof(float));
    cudaMemcpy(d_ones, _ones.data(), _ones.size() * sizeof(float), cudaMemcpyHostToDevice);
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, bias, d_ones, output, out_features, 1, batch_size, 1.0 , 1.0);
    
    return;
}

/* grad_output -- y (out_features*batch_size)  grad_input -- x (in_features * batch_size) 
   grad_weights -- out_features * in_features*/
void backward_fc(float* input, float* output, float* weights, float* bias, int batch_size, int in_features, int out_features,
                 float* grad_output, float* grad_input, float* grad_weights, float* grad_bias)
{
    // compute grad_input
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, weights, grad_output, grad_input, in_features, out_features, batch_size, 1.0 , 0.0);
    // compute grad_weight
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, grad_output, input, grad_weights, out_features, batch_size, in_features, 1.0 , 0.0);
    // compute grad_bias?
    cudaMemcpy(grad_bias, grad_output, out_features*batch_size*sizeof(float), cudaMemcpyDeviceToDevice);
    return;
}

int main()
{
    /* batchsize:3 , C_in:2 , C_out:4 . So we can construct our input X (2,3) and weight matrix W (4,2) */
    int space_x = 6*sizeof(float);
    int space_w = 8*sizeof(float);
    std::vector<float> X_cpu = {1.0,-2.0, 4.0,-3.0, 5.0,7.0};
    std::vector<float> W_cpu = {-2.0,1.0,3.0,4.0,-2.0,3.0,4.0,6.0};
    std::vector<float> bias = {1.0,-1.0,0,1.0};
    float *X;
    float *W;
    float *B;
    cudaMalloc(&X,space_x);
    cudaMalloc(&W,space_w);
    cudaMalloc(&B,4*sizeof(float));

    cudaMemcpy(X,X_cpu.data(),space_x,cudaMemcpyHostToDevice);
    cudaMemcpy(W,W_cpu.data(),space_w,cudaMemcpyHostToDevice);
    cudaMemcpy(B,bias.data(),4*sizeof(float),cudaMemcpyHostToDevice);

    float* output = nullptr;
    cudaMalloc(&output, 12*sizeof(float));

    forward_fc(X,output,W,B,3,2,4);

    float* Y_cpu = (float*)malloc(12*sizeof(float));
    cudaMemcpy(Y_cpu,output,12*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0; i < 12; i ++)
    {
        std::cout << Y_cpu[i] << " ";
    }
    std::cout << std::endl;

    /* we let the output to be the grad_output, so that we can test our backward function */
    float *grad_input, *grad_weights, *grad_bias;
    cudaMalloc(&grad_input, 6*sizeof(float));
    cudaMalloc(&grad_weights, 8*sizeof(float));
    cudaMalloc(&grad_bias,12*sizeof(float));

    backward_fc(X,output,W,B,3,2,4,output,grad_input,grad_weights,grad_bias);

    float *grad_input_cpu, *grad_weights_cpu;
    grad_input_cpu = (float*)malloc(6*sizeof(float));
    grad_weights_cpu = (float*)malloc(8*sizeof(float));
    cudaMemcpy(grad_input_cpu,grad_input,6*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_weights_cpu,grad_weights,8*sizeof(float),cudaMemcpyDeviceToHost);

    for(int i=0; i < 6; i ++)
    {
        std::cout << grad_input_cpu[i] << " ";
    }
    std::cout << std::endl;

    for(int i=0; i < 8; i ++)
    {
        std::cout << grad_weights_cpu[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}