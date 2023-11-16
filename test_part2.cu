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
        free(data_cpu);
    }
    return;
}
/* in this function, we try to implement max pooling layer, to simplify our code, we set kernel size is 2*2 
   the stride is also 2, and we consider no padding. so we don't have to set the parameters seperately */
__global__ void max_pool_forward(float* in_data, int batch_size, int channels, int H, int W, int out_H, int out_W,
                                 float* out_data, float* out_mask)
{
    int nthreads = out_W*out_H*batch_size*channels;
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
        out_data[n*out_H*out_W*channels + c*out_H*out_W + ph*out_W + pw] = max_value;
        out_mask[n*out_H*out_W*channels + c*out_H*out_W + ph*out_W + pw] = max_mask;
    }
    return;
}

__global__ void max_pool_backward(float* in_data, int batch_size, int channels, int H, int W, int out_H, int out_W,
                                 float* out_data, float* out_mask, float* out_gradient, float* input_gradient)
{
    int nthreads = out_H*out_W*batch_size*channels;
    CUDA_KERNEL_LOOP(index, nthreads){
        int n = index / out_W / out_H / channels;
        int c = (index / out_W / out_H) % channels;
        int ph = (index / out_W) % out_H;
        int pw = index % out_W;
        int location = n*out_W*out_H*channels + c*out_W*out_H + ph*out_W + pw;
        input_gradient[int(out_mask[location])] = out_gradient[location]*in_data[int(out_mask[location])];
    }
    return;
}

int main()
{
    float* data = (float*)malloc(24*sizeof(float));
    for(int i = 0; i < 24; i++)
    {
        data[i] = i;
    }
    float* data_gpu = nullptr;
    cudaMalloc(&data_gpu,24*sizeof(float));
    cudaMemcpy(data_gpu,data,24*sizeof(float),cudaMemcpyHostToDevice);
    float* out_data,*out_mask;
    cudaMalloc(&out_data,4*sizeof(float));
    cudaMalloc(&out_mask,4*sizeof(float));
    max_pool_forward <<< CudaGetBlocks(4),kCudaThreadsNum >>>(data_gpu,2,2,3,2,1,1,out_data,out_mask);
    std::cout << "data_gpu" << std::endl; 
    matrix_print(data_gpu,24,"gpu");
    std::cout << "out_data" << std::endl;
    matrix_print(out_data,4,"gpu");
    std::cout << "out_mask" << std::endl;
    matrix_print(out_mask,4,"gpu");

    float* grad_data = (float*)malloc(4*sizeof(float));
    for(int i = 0; i < 4; i ++)
    {
        grad_data[i] = i - 2;
    }
    float* grad_data_gpu = nullptr;
    cudaMalloc(&grad_data_gpu,4*sizeof(float));
    cudaMemcpy(grad_data_gpu,grad_data,4*sizeof(float),cudaMemcpyHostToDevice);
    float* input_gradient = nullptr;
    cudaMalloc(&input_gradient,24*sizeof(float));
    max_pool_backward <<< CudaGetBlocks(4),kCudaThreadsNum >>>(data_gpu,2,2,3,2,1,1,out_data,out_mask,grad_data_gpu,input_gradient);
    matrix_print(input_gradient,24,"gpu");

    return 0;
}