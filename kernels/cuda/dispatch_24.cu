// Vocab projection [151936, 1024] @ [1024, 1] — f16 GEMV
// 1 push constant (output offset), 3 bindings
// This is the LM head — f16 weights (not Q8_0), runs once per token
#include <cuda_fp16.h>
#include <stdint.h>
#define ROWS_PER_BLOCK 8
#define WARP_SIZE 32
#define NTHREADS 256

extern "C" __global__ __launch_bounds__(256)
void decode_dispatch_24_matmul_151936x1x1024_f16(
    const __half* __restrict__ weight,   // binding(0): [151936, 1024] f16
    const char* __restrict__ b1_raw,     // binding(1): input [1024, 1] at offset 64
    char* __restrict__ b2_raw,           // binding(2): output [151936, 1] at offset c0
    uint32_t c0                          // output byte offset
) {
    const int N = 151936, K = 1024;

    const __half* input = (const __half*)(b1_raw + 64);
    __half* output = (__half*)(b2_raw + (int)c0);

    // Load input vector to shared memory
    __shared__ __half shmem[K];
    for (int i = threadIdx.x; i < K; i += NTHREADS)
        shmem[i] = input[i];
    __syncthreads();

    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= N) return;

    // Each lane handles K/32 = 32 elements
    const __half* row_data = weight + (long long)row * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += WARP_SIZE) {
        acc += __half2float(row_data[k]) * __half2float(shmem[k]);
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        output[row] = __float2half(acc);
}
