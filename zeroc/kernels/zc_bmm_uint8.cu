#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32 // Tile width, can be tuned for different scenarios

__global__ void bmm_uint8_kernel(const uint8_t *A, const uint8_t *B, uint8_t *C, int bs, int a, int b, int c)
{
    __shared__ uint8_t As[TILE_WIDTH][TILE_WIDTH];
    __shared__ uint8_t Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    uint8_t Cvalue = 0;

    for (int t = 0; t < (b - 1) / TILE_WIDTH + 1; ++t)
    {
        if (Row < bs * a && t * TILE_WIDTH + tx < b)
            As[ty][tx] = A[Row * b + t * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0;

        if (Col < c && t * TILE_WIDTH + ty < b)
            Bs[ty][tx] = B[Row / a * b * c + (t * TILE_WIDTH + ty) * c + Col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Cvalue += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (Row < bs * a && Col < c)
        C[Row * c + Col] = Cvalue;
}

torch::Tensor bmm_uint8(torch::Tensor A, torch::Tensor B)
{
    const auto bs = A.size(0);
    const auto a = A.size(1);
    const auto b = A.size(2);
    const auto c = B.size(2);

    auto C = torch::zeros({bs, a, c}, torch::dtype(torch::kUInt8).device(torch::kCUDA));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((c + TILE_WIDTH - 1) / TILE_WIDTH, (bs * a + TILE_WIDTH - 1) / TILE_WIDTH);

    bmm_uint8_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(), C.data_ptr<uint8_t>(), bs, a, b, c);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("call", &bmm_uint8, "Tiled batch matrix multiplication for uint8_t with CUDA");
}
