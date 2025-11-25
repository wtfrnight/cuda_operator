#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>


// ---------------------------- 常量与类型别名宏 ----------------------------
#define WARP_SIZE 32  // 每个 warp 的线程数
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])       
// 将变量 reinterpret_cast 为 int4 类型引用（一次处理4个int）
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])   
// 将变量 reinterpret_cast 为 float4 类型引用（一次处理4个float）
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])     
// 将变量 reinterpret_cast 为 half2 类型引用（一次处理2个half）
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0]) 
// 同理用于 bfloat16
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])     
// 通过float4实现一次128bit加载/存储



// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_kernel(float* a, float* b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
     c[idx] = a[idx] + b[idx];
}


// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32x4_kernel(float* a, float* b, float* c, int N)
{
    int idx = 4 * (blockIdx * blockDim.x + threadIdx.x);
    //展开
    if(idx < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.x + reg_b.y; 
        reg_c.z = reg_a.x + reg_b.z; 
        reg_c.w = reg_a.x + reg_b.w;
        FLOAT4(c[idx]) = reg_c; 
    }
}


// FP16
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);
}

// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x2_kernel(half * a, half* b, half* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx;
    if(idx < N)
    {
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c;
        reg_c.x = __hadd(reg_a.x, reg_b.x);
        reg_c.y = __hadd(reg_a.y, reg_b.y);
        HALF2(c[idx]) = reg_c;  
    }
}


__global__ void elementwise_add_f16x8_kernel(half* a, half* b, half* c, int N)
{
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_a_0 = HALF2(a[idx]);
    half2 reg_a_1 = HALF2(a[idx+2]);
    half2 reg_a_2 = HALF2(a[idx+4]);
    half2 reg_a_3 = HALF2(a[idx+6]);
    half2 reg_b_0 = HALF2(b[idx]);
    half2 reg_b_1 = HALF2(b[idx+2]);
    half2 reg_b_2 = HALF2(b[idx+4]);
    half2 reg_b_3 = HALF2(b[idx+6]);
    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
    rer_c_0.x = __hadd(reg_a_0.x + reg_b_0.x);
    rer_c_0.y = __hadd(reg_a_0.y + reg_b_0.y);
    rer_c_1.x = __hadd(reg_a_1.x + reg_b_1.x);
    rer_c_1.y = __hadd(reg_a_1.y + reg_b_1.y);
    rer_c_2.x = __hadd(reg_a_2.x + reg_b_2.x);
    rer_c_2.y = __hadd(reg_a_2.y + reg_b_2.y);
    rer_c_3.x = __hadd(reg_a_3.x + reg_b_3.x);
    rer_c_3.y = __hadd(reg_a_3.y + reg_b_3.y);
    if ((idx + 0) < N) {
        HALF2(c[idx + 0]) = reg_c_0;
    }
    if ((idx + 2) < N) {
        HALF2(c[idx + 2]) = reg_c_1;
    }
    if ((idx + 4) < N) {
        HALF2(c[idx + 4]) = reg_c_2;
    }
    if ((idx + 6) < N) {
        HALF2(c[idx + 6]) = reg_c_3;
    }    
}


__global__ void elementwise_add_f16x8_pack_kernel(half* a, half* b, half* c, int N)
{
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    //local register
    
    if(idx + 7 < N)//board check
    {
        half pack_a[8], pack_b[8], pack_c[8];

        LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
        LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

        #pragma unroll//展开
        for(int i =0; idx + i < N; i+=2)
        {
            HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]),HALF2(pack_b[i]));
        }
        //写回
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    }
    else
    {
        #pragma unroll
        for(int i=0; idx + i < N; i++)
        {
            c[idx + i] = __hadd(a[idx + i],b[idx + i]); 
        }
    }
}