#include <cuda_fp16.h>
#include <stdint.h>
#define ROWS_PER_BLOCK 8
#define WARP_SIZE 32
#define NTHREADS 256

extern "C" __global__ __launch_bounds__(256)
void decode_dispatch_20_matmul_1024x1x3072_f16(
    const uint8_t* __restrict__ q8_stacked,
    const char* __restrict__ b1_raw,
    char* __restrict__ b2_raw,
    uint32_t c0, uint32_t c1
) {
    const int N_DIM = 1024, K_DIM = 3072;
    const int Q8_BLOCK = 32, Q8_BYTES = 34;
    const int BYTES_PER_ROW = 3264;
    const long long BPL = 3342336LL;

    long long layer = (long long)c0 | ((long long)c1 << 32);
    const __half* input = (const __half*)(b1_raw + 18432);
    __half* output = (__half*)(b2_raw + 0);
    const uint8_t* layer_data = q8_stacked + layer * BPL;

    __shared__ __half shmem[K_DIM];
    for (int i = threadIdx.x; i < K_DIM; i += NTHREADS) shmem[i] = input[i];
    __syncthreads();

    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= N_DIM) return;

    const uint8_t* row_data = layer_data + (long long)row * BYTES_PER_ROW;
    float acc = 0.0f;
    for (int b = lane; b < K_DIM/Q8_BLOCK; b += WARP_SIZE) {
        int boff = b * Q8_BYTES;
        uint16_t sbits = row_data[boff] | ((uint16_t)row_data[boff+1] << 8);
        float scale = __half2float(*reinterpret_cast<const __half*>(&sbits));
        int k_base = b * Q8_BLOCK;
        #pragma unroll
        for (int j = 0; j < Q8_BLOCK; j++) {
            int8_t qv = ((const int8_t*)row_data)[boff + 2 + j];
            acc += scale * (float)qv * __half2float(shmem[k_base + j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    if (lane == 0) output[row] = __float2half(acc);
}
