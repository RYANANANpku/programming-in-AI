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


__global__ void find_max(float* in_data, int batch_size, int c, float* out_data)
{
    int nthreads = batch_size;
    CUDA_KERNEL_LOOP(index, nthreads){
        for(int i = 0; i < c; i ++)
        {
            if(in_data[c*index + i] > out_data[index])
            {
                out_data[index] = in_data[c*index + i];
            }
        }
    }
    return;
}

__global__ void substract_operation(float* in_data, int batch_size, int c, float* value_data)
{
    int nthreads = batch_size*c;
    CUDA_KERNEL_LOOP(index, nthreads){
        int batch_num = index / c;
        int c_num = index % c;
        in_data[batch_num*c + c_num] -= value_data[batch_num];
    }
    return;
}

__global__ void exponent_operation(float* in_data, int batch_size, int c, float* value_data)
{
    int nthreads = batch_size*c;
    CUDA_KERNEL_LOOP(index, nthreads){
        in_data[index] = expf(in_data[index]);
    }
    return;
}

__global__ void sum_row(float* in_data, int batch_size, int c, float* out_data)
{
    int nthreads = batch_size;
    CUDA_KERNEL_LOOP(index, nthreads){
        for(int i = 0; i < c; i ++)
        {
            out_data[index] += in_data[c*index + i];
        }
    }
    return;
}

__global__ void normalize_operation(float* in_data, int batch_size, int c, float* value_data)
{
    int nthreads = batch_size*c;
    CUDA_KERNEL_LOOP(index, nthreads){
        int batch_num = index / c;
        int c_num = index % c;
        in_data[batch_num*c + c_num] /= value_data[batch_num];
    }
    return;
}

void softmax(float* in_data, int batch_size, int c)
{
    // first step: compute the max element
    float* out_data;
    cudaMalloc(&out_data, batch_size*sizeof(float));
    find_max <<<CudaGetBlocks(batch_size),kCudaThreadsNum>>> (in_data, batch_size, c, out_data);
    matrix_print(out_data,batch_size,"gpu");

    // second step: substract the max value for each row
    substract_operation <<<CudaGetBlocks(batch_size*c),kCudaThreadsNum>>> (in_data, batch_size, c, out_data);
    matrix_print(in_data,batch_size*c,"gpu");

    // third step: Compute the exponent for each element
    exponent_operation <<<CudaGetBlocks(batch_size*c),kCudaThreadsNum>>> (in_data, batch_size, c, out_data);
    matrix_print(in_data,batch_size*c,"gpu");

    // Sum over each row to compute the normalization factor
    sum_row <<<CudaGetBlocks(batch_size),kCudaThreadsNum>>> (in_data, batch_size, c, out_data);
    matrix_print(out_data,batch_size,"gpu");

    // Normalize the results
    normalize_operation <<<CudaGetBlocks(batch_size*c),kCudaThreadsNum>>> (in_data, batch_size, c, out_data);
    matrix_print(in_data,batch_size*c,"gpu");

    return;
}

int main()
{
    float* data = (float*)malloc(20*sizeof(float));
    for(int i = 0; i < 20; i++)
    {
        data[i] = i;
    }
    float* data_gpu = nullptr;
    cudaMalloc(&data_gpu,24*sizeof(float));
    cudaMemcpy(data_gpu,data,24*sizeof(float),cudaMemcpyHostToDevice);
    softmax(data_gpu,2,10);
    //matrix_print(data_gpu,20,"gpu");

    return 0;
    
}