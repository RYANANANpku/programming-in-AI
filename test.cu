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

void matrix_print(float* data, int size, std::string mode)
{
    if(mode == "cpu")
    {
        for(int i = 0; i < size; i++)
        {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    else if(mode == "gpu")
    {
        float* data_cpu = (float*)malloc(size*sizeof(float));
        cudaMemcpy(data_cpu,data,size*sizeof(float),cudaMemcpyDeviceToHost);
        for(int i = 0; i < size; i++)
        {
            //if(i % 18 == 0) std::cout << std::endl;
            std::cout << data_cpu[i] << " ";
        }
        std::cout << std::endl;
    }
    return;
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

__global__ void im2col(const float* data_im, float* data_col, int H, int W, int C, int N)
{
    int k_size = 3;
    int length = H*W*N;
    int width = C*k_size*k_size;
    int nthreads = length*width;
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int col_i = index / width;
        int col_j = index % width;
        int batch_num = col_i/(H*W);
        int w_i = (col_i%(H*W))/W;
        int w_j = (col_i%(H*W))%W;
        int c_num = col_j/(k_size*k_size);
        int d_i = (col_j%(k_size*k_size))/3 - 1;
        int d_j = (col_j%(k_size*k_size))%3 - 1;
        int im_i = w_i + d_i;
        int im_j = w_j + d_j;
        if(im_i < 0 || im_j < 0 || im_i >= H || im_j >= W) data_col[index] = 0;
        else data_col[index] = data_im[batch_num*H*W*C + c_num*H*W + im_i*W + im_j];
    }
    return;
}

__global__ void reshape(float* input, float* output, int batch_size, int C_in, int H, int W, int C_out)
{
    /* this function is used to assist the impletation of convolution forward */
    /* through gemm we have a output matrix, which is (batch_size*H*W,C_out). we want to reshape it to the (N,C,H,W) form */
    int nthreads = batch_size*C_out*H*W;
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int batch_num = index/(H*W*C_out);
        int C_num = (index % (H*W*C_out))/(H*W);
        int i = (index % (H*W)) / W;
        int j = (index % (H*W)) % W;
        output[index] = input[C_num*H*W*batch_size + batch_num*H*W + i*W + j];
    }
    return;
}

__global__ void grad_reshape(float* input, float* output, int batch_size, int C_in, int H, int W, int C_out)
{
    /* this is the reverse operation of reshape */
    /* our input is grad_y (N*C*H*W), we want to reshape it to (N*H*W,C), which helps us do the gemm operation */
    int nthreads = batch_size*C_out*H*W;
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int C_num = index / (H*W*batch_size);
        int batch_num = (index % (H*W*batch_size))/(H*W);
        int i = (index % (H*W)) / W;
        int j = (index % (H*W)) % W;
        output[index] = input[batch_num*H*W*C_out + C_num*H*W + i*W + j];
    }
    return;
}

void convolution_forward(const float* data_im, const float* data_kernel, float* data_col, float* output_gpu, float* bias, int batch_size, int C_in, int H, int W, int C_out)
{
    /* first step: compute the im2col matrix */
    int k = 3;
    im2col <<<CudaGetBlocks(H*W*batch_size*C_in*k*k), kCudaThreadsNum>>> (data_im, data_col, H, W, C_in, batch_size);
    //matrix_print(data_col_gpu,H*W*batch_size*C_in*k*k,"gpu");

    /* gemm computation */
    float* output = nullptr;
    cudaMalloc(&output,batch_size*H*W*C_out*sizeof(float));
    gemm_gpu(CUBLAS_OP_T,CUBLAS_OP_N,data_col,data_kernel,output,batch_size*H*W,C_in*k*k,C_out,1.0,0.0);
    matrix_print(output,batch_size*H*W*C_out,"gpu");
    reshape <<<CudaGetBlocks(batch_size*H*W*C_out),kCudaThreadsNum >>> (output,output_gpu,batch_size,C_in,H,W,C_out);
    matrix_print(output_gpu,batch_size*H*W*C_out,"gpu");
    grad_reshape <<<CudaGetBlocks(batch_size*H*W*C_out),kCudaThreadsNum >>> (output_gpu,output,batch_size,C_in,H,W,C_out);
    matrix_print(output,batch_size*H*W*C_out,"gpu");

    /* add bias */
    std::vector<float> _ones(H*W, 1.0);
    float *d_ones;
    cudaMalloc((void**)&d_ones, _ones.size() * sizeof(float));
    cudaMemcpy(d_ones, _ones.data(), _ones.size() * sizeof(float), cudaMemcpyHostToDevice);
    gemm_gpu(CUBLAS_OP_N,CUBLAS_OP_T,d_ones,bias,output_gpu,H*W,1,C_out*batch_size,1.0,1.0);
    matrix_print(output_gpu,batch_size*H*W*C_out,"gpu");
    
}

__global__ void col2im(float* input, float*output, int batch_size, int C_in, int H, int W, int C_out)
{
    int k = 3;
    int nthreads = H*W*batch_size*C_in*k*k;
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int col_i = index / (C_in*k*k);
        int col_j = index % (C_in*k*k);
        int batch_num = col_i / (H*W);
        int win_i = (col_i % (H*W)) / W;
        int win_j = (col_i % (H*W)) % W;
        int c_num = col_j /(k*k);
        int d_i = (col_j % (k*k))/3 - 1;
        int d_j = (col_j % (k*k))%3 - 1; 
        int i = win_i + d_i;
        int j = win_j + d_j;
        if(i > 0 && i < H && j > 0 && j < W)
        {
            atomicAdd(&output[batch_num*H*W*C_in + c_num*H*W + i*W + j],input[index]);
            //output[batch_num*H*W*C_in + c_num*H*W + i*W + j] += input[index];
        } 
    }
    return;
}

void convolution_backward(const float* data_im, const float* data_kernel, float* data_col, float* output, int batch_size, int C_in, int H, int W, int C_out,
                          float* grad_output, float* grad_in_col, float* grad_weights, float* grad_in_im, float* grad_bias)
{
    int k = 3;
    /* reshape the grad_output */
    float* grad_y = nullptr;
    cudaMalloc(&grad_y,batch_size*H*W*C_out*sizeof(float));
    grad_reshape <<<CudaGetBlocks(batch_size*H*W*C_out),kCudaThreadsNum >>> (grad_output,grad_y,batch_size,C_in,H,W,C_out);

   /* compute the grad_weigths */
   gemm_gpu(CUBLAS_OP_N,CUBLAS_OP_N,data_col,grad_y,grad_weights,C_in*k*k,batch_size*H*W,C_out,1.0,0.0);

   /*compute the grad_in_col */
   gemm_gpu(CUBLAS_OP_N,CUBLAS_OP_T,grad_y,data_kernel,grad_in_col,batch_size*H*W,C_out,C_in*k*k,1.0,0.0);
}

int main()
{
    /* batchsize:3 , C_in:2 , C_out:4 . So we can construct our input X (2,3) and weight matrix W (4,2) */
    std::vector<float> X_cpu = {1.0,-2.0, 4.0,-3.0, 5.0,7.0};
    std::vector<float> W_cpu = {-2.0,1.0,3.0,4.0,-2.0,3.0,4.0,6.0};
    std::vector<float> bias = {1.0,-1.0,0,1.0};
    float *X, *W, *B;
    cudaMalloc(&X,6*sizeof(float));
    cudaMalloc(&W,8*sizeof(float));
    cudaMalloc(&B,4*sizeof(float));

    cudaMemcpy(X,X_cpu.data(),6*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(W,W_cpu.data(),8*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B,bias.data(),4*sizeof(float),cudaMemcpyHostToDevice);

    float* output = nullptr;
    cudaMalloc(&output, 12*sizeof(float));

    forward_fc(X,output,W,B,3,2,4);

    float* Y_cpu = (float*)malloc(12*sizeof(float));
    cudaMemcpy(Y_cpu,output,12*sizeof(float),cudaMemcpyDeviceToHost);
    //matrix_print(Y_cpu,12);

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

    //matrix_print(grad_input_cpu,6);
    //matrix_print(grad_weights_cpu,8);

    /* next: test the convolution layer */
    /* test the im2col function first, we assume that the batch_size is 2,the channel is 2,the (H,W) is (3,2) */
    float *data = (float*)malloc(3*2*2*2*sizeof(float));
    for(int i = 0; i < 24; i++)
    {
        data[i] = i;
    }
    float* data_gpu = nullptr;
    cudaMalloc(&data_gpu,3*2*2*2*sizeof(float));
    cudaMemcpy(data_gpu,data,3*2*2*2*sizeof(float),cudaMemcpyHostToDevice);
    /* we construct our kernel data, the channel is 2, the kernel num is 2 */
    float* data_kernel = (float*)malloc(36*sizeof(float));
    for(int i = 0; i < 36; i ++)
    {
        data_kernel[i] = i - 18;
    }
    float* kernel_gpu = nullptr;
    cudaMalloc(&kernel_gpu,18*2*sizeof(float));
    cudaMemcpy(kernel_gpu,data_kernel,18*2*sizeof(float),cudaMemcpyHostToDevice);
    /* we can do the forward test */
    float* output_gpu = nullptr;
    cudaMalloc(&output_gpu,24*sizeof(float));
    float* data_col = nullptr;
    cudaMalloc(&data_col,3*2*2*2*3*3*sizeof(float));
    convolution_forward(data_gpu,kernel_gpu,data_col,output_gpu,B,2,2,3,2,2);



    return 0;
}