#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

#define BLOCK_SIZE 256

//reduce_native,无并行性
__global__ void devide_reduce_v0(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx <  N)
    atomicAdd(output, input[idx]);
}




//使用全局内存,N要求是BLOCK_SIZE整数倍
__global__ void device_reduce_v1(float* d_x, float* d_y)
{
    const int tid = threadIdx.x;
    float *x = &d_x[blockDim.x * blockIdx.x];
    
    for(int offset = blockDim.x >>1; offset > 0; offset >>=1)
    {
        if(tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}

template <const int BLOCK_SIZE>
void call_reduce_v1(float *d_x, float *d_y, float *h_y, const int N, float *sum)
{
    const int GRID_SIZE = (N + BLOCK_SIZE -1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    device_reduce_v1<<<block_size, grid_size>>>(d_x,d_y);
    cudaMemcpy(h_y, d_y, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    *sum = 0.0;
    for(int i=0; i < GRID_SIZE; i++)
    {
        *sum + = h_y[i];
    }
}



//使用共享内存优化,规约成块
template <const int BLOCK_SIZE>
__global__ void device_reduce_v2(float *d_x, float *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + threadIdx.x;
    __shared__ float s_y[BLOCK_SIZE];


    if(n >= N)
    s_y[n] = 0.0;
    else
    s_y[n] = d_x[n];

    __syncthreads();
    
    for(int offset = blockDim.x >>1 ; offset>0; offset >>=1)
    {
        if(tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0)
    d_x[bid] = s_y[tid];
}


template <constxpr int BLOCK_SIZE>
void call_reduce_v2(float *d_x, float *d_y, float *h_y, const int N, float *sum)
{
    const int GRID_SIZE = (N + BLOCK_SIZE -1) / BLOCK_SIZE;
    dim3 grid_size(GRID_SIZE);
    dim3 block_size(BLOCK_SIZE);
    device_reduce_v2<BLOCK_SIZE><<<grid_size, block_size>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
    *sum = 0;
    for(int i = 0; i < GRID_SIZE; i++)
    {
        *sum += h_y[i];
    }
}


//动态共享内存
__global__ void device_reduce_v3(float *d_x, float *d_y, int *sum, int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ float s_y[];//动态共享内存
    if(n >= N)
    s_y[tid] = 0.0;
    else
    s_y[tid] = d_x[n];

    __syncthreads();

    for(int offset = blockDim.x >>1; offset > 0; offset>>=1)
    {
        if(tid < offset)
        s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if(tid == 0)
    d_y[bid] = s_y[tid];
}

void call_reduce_v3(float *d_x, float *d_y, float *h_y, int N, float *sum)
{
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    device_reduce_v3<BLOCK_SIZE><<<grid_size, block_size, sizeof(float) * BLOCK_SIZE>>>
    (d_x, d_y, sum, N);

    cudaMemcpy(h_y, d_y, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    *sum = 0.0;
    for(int i=0; i<GRID_SIZE; i++)
    {
        *sum += h_y[i];
    }
}


//原子操作
__global__ void device_reduce_v4(float *d_x, float *d_y, int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n =bid * blockDim.x + tid;
    extern __shared__ float s_y[];
    
    if(n >= N)
    s_y[n] = 0.0;
    else
    s_y[n] = d_x[n];

    for(int offset = blockDim.x >> 1; offset > 0; offset >>=1)
    {
        if(tid < offset)
        s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    if(tid==0)
    atomicAdd(d_y, s_y[tid]);
}

template <const int BLOCK_SIZE>
void call_reduce_v4(float *d_x, float *d_y, const int N)
{
    const int GRID_SIZE = (N + BLOCK_SIZE -1) / BLOCK_SIZE;
    dim3 grid_size(GRID_SIZE);
    dim3 block_size(BLOCK_SIZE);
    *h_y = 0.0;
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);
    device_reduce_v4<<<grid_size, block_size, sizeof(float) * BLOCK_SIZE>>>
    (d_x, d_y, N);
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}



//warp shuffle   BLOCK_SIZE需要时32的整数倍
__global__ void device_reduce_v5(float *d_x, float *d_y, const int N)
{
    __shared__ float s_y[32];//最多需要32个 32*32=1024
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;//warp的位置
    int laneId = threadIdx.x % warpSize;

    float val;
    if(idx >= N)//搬运到寄存器
    val = 0.0;
    else
    val = d_x[idx];

#pragma unroll
    for(int offset = warpSize >> 1; offset > 0; offset >>=1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);//warp里折半规约
        //warp同步掩码， 交换的数据， 线程索引偏移量（必须是2的幂）
    }

    if(laneId == 0)//warp里第一个写回warp规约结果到共享内存
    s_y[warpId] = val;

    __syncthreads();

    if(warpId == 0)//第一个warp再进行一次warp内规约
    {
        int warpNum = blockDim.x / warpSize;
        if(laneId >= warpNum)//从共享内存读到本地内存
        val = 0.0;
        else
        val = s_y[laneId];
    
        for(int offset = warpSize >> 1; offset > 0; offset >>=1)//warp内规约
        {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if(laneId == 0)//写回
        atomicAdd(d_y, val);
    }
}

void call_reduce_v5(float *d_x, float *d_y, const int N)
{
    const int GRID_SIZE = (N + BLOCK_SIZE -1) / BLOCK_SIZE;
    dim3 grid_size(GRID_SIZE);
    dim3 block_size(BLOCK_SIZE);
    *h_y=0;
    cudaMemcpy(d_y, h_y, cudaMemcpyDeviceToHost);
    device_reduce_v5<<<grid_size, block_size>>> (d_x, d_y, N);
    cudaMemcpy(d_y, h_y, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}


#define FLOAT4(value) (*(float4*)(&(value)))
//向量化+warp_shafi
__global__ void device_reduce_v6(float *d_x, float *d_y, int N)
{
    __shared__ s_y[32];
    float val = 0.0;
    int idx = 4 * (blockIdx.x * blockDim.x  + threadIdx.x);//向量化
    int warpId = blockIdx.x / warpSize;
    int laneId = blockIdx.x % warpSize;
    if(idx < N)
    {
        float4 tmp_x = FLOAT4(d_x[idx]);
        val += tmp_x.x;
        val += tmp_x.y;
        val += tmp_x.z;
        val += tmp_x.w;
    }
#pragma unroll
    for(int offset = warpSize >> 1; offset > 0; offset>>=1)
    {
        val +=__shfl_down_sync(0XFFFFFFFF, val, offset);
    }

    if(laneId == 0)
    s_y[warpId] = val;

    __syncthreads();

    if(warpId == 0)//对每个warp的规约结果进一步规约
    {
        int warp_num = blockDim.x / warpSize;
        if(laneId < warp_num)
        val = s_y[laneId];
        else
        val = 0;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if(laneId == 0)//每个blcok一个线程把块内的规约结果写回全局
        //必须原子操作，*d_y = val;
        atomicAdd(d_y, val);
    }
    
}

void call_reduce_v6(float *d_x, float*d_y, float* h_y, const int N)
{
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GRID_SIZE = (GRID_SIZE + 4 - 1) / 4;
    dim3 grid_size(GRID_SIZE);
    dim3 block_size(BLOCK_SIZE);
    *h_y = 0;
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);
    device_reduce_v6<<<grid_size, block_size>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
}