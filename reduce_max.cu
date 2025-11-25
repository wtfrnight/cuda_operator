//atomicCAS(addr, cmp_val, new_val)
//原子性检查 addr 指向的目标值，是否与 cmp_val（前两个参数）相等；
//若相等：将 new_val（第三个参数）写入 addr，返回「原来的 cmp_val」；
//若不相等：不写入任何数据，返回「addr 当前的最新真实值」；
//atomixCAS只能比较整形的



//读旧值-->原子比较新制是否一致-->一致则没有其他线程更改-->写入
//                          -->不一致说明有其他线程更改--->更新旧值--->进入循环
__device__ static float atomicMax(float* address, float val)
{
    int *address_as_i = (int*) address;//address转换成int*类型指针
    int old = *address_as_i;//address的旧值用int解码
    int assumed;
    do{
        assumed = old;
        old = atomixCAS(address_as_i, assumed, 
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    }while(assumed!=old);//写入失败进入循环


    return __int_as_float(old);
}

__global__ void reduce_max(float* d_x, float *d_y, const int N)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    __shared__ float s_y[32];//BLOCKsize最大1024
    float val;
    if(idx >= N)
    {
        val = -FLT_MAX;
    }
    else
        val = d_x[idx];

    for(int offset = warpSize >>1; offset >0; offset>>=1)//warp内规约
    {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    if(laneId == 0)//一个warp中一个thread写回共享内存
    s_y[warpId] = val;
    
    __syncthreads();

    if(warpId==0)//对所有warp的结果进行规约
    {
        int warpNum = blockDim.x / warpSize;
        if(laneId >= warp_num)
        val = -FLT_MAX;
        else
        val = s_y[laneId];
        for(int offset = warpSize; offset>0; offset>>=1)
        {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if(laneId == 0)//每个block一个thread负责写回blockmax
        atomicMax(d_y, val);
    }

}