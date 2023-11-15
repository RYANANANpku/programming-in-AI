#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <cmath>
#include <cublas_v2.h>


// Use 512 threads per block
const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N) {
   return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDA_KERNEL_LOOP(i, n)                         \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;  \
       i < (n);                                        \
       i += blockDim.x * gridDim.x)


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
    grad_bias = grad_output;
    return;
}

/* to simplyfy our code, we assume that the kernel size is 3*3, the stride is 1, and the padding strategy is zero padding
   so we don't have to set parameters for them seperately */
void im2col(const float* data_im, float* data_col, int H, int W, int C, int N)
{
    int k_size = 3;
    int stride = 1;
    int padding = 1;
    /* we consider the first situation: input is just H*W */
    int space = H*W*k_size*k_size*sizeof(float);
    data_col = (float*)malloc(sizeof(space));
    int length = H*W;
    for(int col_i = 0; col_i < length; col_i ++)
    {
        for(int col_j = 0; col_j < k_size*k_size; col_j ++)
        {
            int w_i = col_i/W;
            int w_j = col_i%W;
            int dx = col_j/3 - 1;
            int dy = col_j%3 - 1;
            int im_i = w_i + dx;
            int im_j = w_j + dy;
            if(im_i < 0 || im_j < 0 || im_i >= H || im_j >= W) data_col[col_i*length + col_j] = 0;
            else data_col[col_i*length + col_j] = data_im[im_i*W + im_j];
        }
    }
    /* we consider the second situation: input is C*H*W */
    int space = C*H*W*k_size*k_size*sizeof(float);
    data_col = (float*)malloc(sizeof(space));
    int length = H*W;
    int width = C*k_size*k_size;
    for(int col_i = 0; col_i < length; col_i ++)
    {
        for(int col_j = 0; col_j < width; col_j ++)
        {
            int w_i = col_i/W;
            int w_j = col_i%W;
            int c_num = col_j/C;
            int d_i = (col_j%C)/3 - 1;
            int d_j = (col_j%C)%3 - 1;
            int im_i = w_i + d_i;
            int im_j = w_j + d_j;
            if(im_i < 0 || im_j < 0 || im_i >= H || im_j >= W) data_col[col_i*length + col_j] = 0;
            else data_col[col_i*length + col_j] = data_im[c_num*H*W + im_i*W + im_j];
        }
    }
    /* we consider the last situation: input is N*C*H*W */
    int space = N*C*H*W*sizeof(float);
    data_col = (float*)malloc(sizeof(space));
    int length = H*W*N;
    int width = C*k_size*k_size;
    for(int col_i = 0; col_i < length; col_i ++)
    {
        for(int col_j = 0; col_j < width; col_j ++)
        {
            int batch_num = col_i/(H*W);
            int w_i = (col_i%(H*W))/W;
            int w_j = (col_j%(H*W))%W;
            int c_num = col_j/C;
            int d_i = (col_j%C)/3 - 1;
            int d_j = (col_j%C)%3 - 1;
            int im_i = w_i + d_i;
            int im_j = w_j + d_j;
            if(im_i < 0 || im_j < 0 || im_i >= H || im_j >= W) data_col[col_i*length + col_j] = 0;
            else data_col[col_i*length + col_j] = data_im[batch_num*H*W*C + c_num*H*W + im_i*W + im_j];
        }
    }
    return;
}

void convolution_forward(const float* data_im, const float* data_kernel, float* output, int batch_size, int C_in, int H, int W, int C_out)
{
    /* first step: compute the im2col matrix */
    float* data_col;
    int k = 3;
    im2col(data_im, data_col, H, W, C_in, batch_size);
    /* col_matrix (N*H*W,c_in*k*k) kernel_matrix (c_in*k*k,c_out) */
    int space = C_in*H*W*batch_size*sizeof(int);
    float* output_matrix = (float*)malloc(sizeof(space));
    gemm_gpu(CUBLAS_OP_T,CUBLAS_OP_N,data_col,data_kernel,output_matrix,batch_size*H*W,C_in*k*k,C_out,1.0,0.0);
    // /* we get the output matrix (batch_size*H*W,c_out), we have to transfer the data from matrix to output*/
    // output = (float*)malloc(sizeof(space));
    // for(int batch_num = 0; batch_num < batch_size; batch_num ++)
    // {
    //     for(int c_num = 0; c_num < C_out, c_num ++)
    //     {
    //         for(int im_num = 0; im_num < H*W; im_num ++)
    //         {
    //             output[batch_num*H*W*C_out + c_num*H*W + im_num] = output_matrix[c_num*H*W*batch_size + batch_num*H*W + im_num];
    //         }
    //     }
    // }
    return;
}

void convolution_backward(const float* data_im, const float* data_kernel, float* output, int batch_size, int C_in, int H, int W, int C_out,
                          const float* grad_output, float* grad_in_col, float* grad_weights, float* grad_in_im)
{
    /* we assume that the grad_output dimension is (N*H*W,c_out) */
    float* data_col;
    int k = 3;
    im2col(data_im, data_col, H, W, C_in, batch_size);
    /* compute grad_weights */
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, grad_in_col, grad_output, grad_weights, C_in*k*k, batch_size*H*W, C_out, 1.0, 0.0);
    /* compute grad_in_col */
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, grad_output, data_kernel, grad_in_col, batch_size*H*W, C_out, C_in*k*k, 1.0, 0.0);
    /* compute grad_in_im from grad_in_col */
    for(int ckk_num = 0; ckk_num < C_in*k*k; ckk_num ++)
    {
        for(int HW_num = 0; HW_num < batch_size*H*W; HW_num ++)
        {
            int batch_num = HW_num / batch_size*H*W;
            int window_num = batch_num % batch_size*H*W;
            int c_num = ckk_num / k*k;
            int dx = (ckk_num%(k*k))/3 - 1;
            int dy = (ckk_num%(k*k))%3 - 1;
            int i = window_num / W + dx;
            int j = window_num % W + dy;
            grad_in_im[batch_num*H*W*C_in + c_num*H*W + i*W + j] += grad_in_col[ckk_num*H*W*batch_size + HW_num];
        }
    }
    return;
}

/* in this function, we try to implement max pooling layer, to simplify our code, we set kernel size is 2*2 
   the stride is also 2, and we consider no padding. so we don't have to set the parameters seperately */
__global__ void max_pool_forward(float* in_data, int nthreads, int batch_size, int channels, int H, int W, int out_H, int out_W,
                                 float* out_data, float* out_mask)
{
    CUDA_KERNEL_LOOP(index, nthreads){
        int n = index / out_W / out_H / channels;
        int c = (index / out_W / out_H) % channels;
        int ph = (index / out_W) % out_H;
        int pw = index % out_W;
        int im_h = ph*2;
        int im_w = pw*2;
        int location = n*H*W*channels + c*H*W + im_h*W + im_w;
        int max_value = in_data[location];
        int max_mask =  location;
        if(in_data[location + 1] > max_value)
        {
            max_value = in_data[location + 1];
            max_mask = location + 1;
        }
        if(in_data[location + W] > max_value)
        {
            max_value = in_data[location + W];
            max_mask = location + W;
        }
        if(in_data[location + W + 1] > max_value)
        {
            max_value = in_data[location + W + 1];
            max_mask = location + W + 1;
        }
        out_data[n*out_H*out_W*channels + c*out_H*out_W + ph*out_W + out_H] = max_value;
        out_mask[n*out_H*out_W*channels + c*out_H*out_W + ph*out_W + out_H] = max_mask;
    }
    return;
}

__global__ void max_pool_backward(float* in_data, int nthreads, int batch_size, int channels, int H, int W, int out_H, int out_W,
                                 float* out_data, float* out_mask, float* out_gradient, float* input_gradient)
{
    CUDA_KERNEL_LOOP(index, nthreads){
        int n = index / out_W / out_H / channels;
        int c = (index / out_W / out_H) % channels;
        int ph = (index / out_W) % out_H;
        int pw = index % out_W;
        int location = n*out_W*out_H*channels + c*out_W*out_H + ph*out_W + pw;
        input_gradient[int(out_mask[location])] = out_gradient[location]*out_data[location];
    }
    return;
}

__global__ void thread_row(float* in_data, int batch_size, int c, float* out_data, std::string mode)
{
    int nthreads = batch_size;
    CUDA_KERNEL_LOOP(index, nthreads){
        //float out_data[index] = in_data[index*c];
        for(int i=0; i < c; i ++)
        {
            if(mode == "max")
            {
                if(in_data[index*c + i] > out_data[index])
                {
                    out_data[index] = in_data[index*c + i];
                }
            }
            else if(mode == "sum")
            {
                out_data[index] += in_data[index*c + i];
            }
        }
    }
    return;
}

__global__ void map_operation(float* in_data, int batch_size, int c, float* value_data, std::string mode)
{
    int nthreads = batch_size*c;
    CUDA_KERNEL_LOOP(index, nthreads){
        if(mode == "substract")
        {
            int batch_num = index / c;
            int c_num = index % c;
            in_data[batch_num*c + c_num] -= value_data[batch_num];
        }
        else if(mode == "exp")
        {
            in_data[index] = expf(in_data[index]);
        }
        else if(mode == "normalize")
        {
            int batch_num = index / c;
            int c_num = index % c;
            in_data[batch_num*c + c_num] /= value_data[batch_num];
        }
    }
    return;
}

void softmax(float* in_data, int batch_size, int c)
{
    // first step: compute the max element
    float* out_data;
    cudaMalloc(&out_data, batch_size*sizeof(float));
    thread_row <<<CudaGetBlocks(batch_size),kCudaThreadsNum>>> (in_data, batch_size, c, out_data, "max");
    __syncthreads();

    // second step: substract the max value for each row
    map_operation <<<CudaGetBlocks(batch_size),kCudaThreadsNum>>> (in_data, batch_size, c, out_data, "substract");
    __syncthreads();

    // third step: Compute the exponent for each element
    map_operation <<<CudaGetBlocks(batch_size),kCudaThreadsNum>>> (in_data, batch_size, c, out_data, "exp");
    __syncthreads();

    // Sum over each row to compute the normalization factor
    thread_row <<<CudaGetBlocks(batch_size),kCudaThreadsNum>>> (in_data, batch_size, c, out_data, "sum");
    __syncthreads();

    // Normalize the results
    map_operation <<<CudaGetBlocks(batch_size),kCudaThreadsNum>>> (in_data, batch_size, c, out_data, "normalize");
    __syncthreads();

    return;
}


__global__ void CE_loss_forward(float* in_data, float* label, float* loss, int batch_size, int c)
{
    int nthreads = batch_size;
    CUDA_KERNEL_LOOP(index, nthreads){
        for(int i = 0; i < c; i ++)
        {
            loss[index] += -label[index*c + i]*log(in_data[index*c + i]);
        }
        loss[index] /= c;
    }
    __syncthreads();

    return;
}

__global__ void CE_loss_backward(float* in_data, float* label, float* loss, int batch_size, int c, float* in_grad)
{
    int nthreads = batch_size*c;
    CUDA_KERNEL_LOOP(index, nthreads){
        int batch_num = index / c;
        int c_num = index % c;
        in_grad[batch_num*c + c_num] = (in_data[batch_num*c + c_num] - label[batch_num*c + c_num])/c;
    }
    __syncthreads();

    return;
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

    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, d_A, d_B, d_C, m, k, n, 1.0 , 0.0);

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