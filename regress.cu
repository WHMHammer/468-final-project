#include <cstdio>
#include <ctime>
#include <curand_kernel.h>

constexpr int model_count = 100;

// Hyperparameters
constexpr float huber_loss_threashold = 10;
constexpr float z_score_trimming_threashold = 2;
constexpr float epsilon = 0.49;
constexpr float learning_rate = 0.01;
constexpr int batch_size = 128;
constexpr int max_iter = 10000;

constexpr int sample_size = 1000;
constexpr int dimension = 6;

template<int block_size> __global__ void kernel(float* const _X, float* const _y, float* const _w, const clock_t seed) {
    float* const X = _X + blockIdx.x * sample_size * dimension;
    float* const y = _y + blockIdx.x * sample_size;
    __shared__ float w[dimension];
    __shared__ int indices[batch_size];
    int indices_copy[batch_size];
    __shared__ float residuals[batch_size];
    __shared__ float residuals_copy[batch_size];
    __shared__ float gradient[dimension];
    __shared__ float prev_loss;
    __shared__ float loss;
    __shared__ int index_low;
    __shared__ int index_high;
    __shared__ bool z_score_trimming_flag_converged;
    curandState_t state;
    curand_init(seed, 0, 0, &state);

    // Initialization
    if (threadIdx.x == 0) {
        for (int i = 0; i < dimension; i++) {
            w[i] = 1;
        }
        prev_loss = 0;
    }

    for (int _ = -1; _ < max_iter; _++) {
        // Sample a consecutive batch with random starting index
        indices[threadIdx.x] = ((int)(curand_uniform(&state) * sample_size) + threadIdx.x) % sample_size;

        // Calculate residuals
        residuals[threadIdx.x] = -y[indices[threadIdx.x]];
        for (int j = 0; j < dimension; j++) {
            residuals[threadIdx.x] += X[j * sample_size + indices[threadIdx.x]] * w[j];
        }

        // Merge sort residuals and permute the indices accordingly
        for (int i = 1; i < batch_size; i += 2) {
            if (residuals[i] < residuals[i - 1]) {
                const float tmp_float = residuals[i];
                residuals[i] = residuals[i - 1];
                residuals[i - 1] = tmp_float;
                const int tmp_int = indices[i];
                indices[i] = indices[i - 1];
                indices[i - 1] = tmp_int;
            }
        }
        for (int stride = 2; stride < batch_size; stride *= 2) {
            __syncthreads();
            if (threadIdx.x % (stride * 2) == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + stride;
                if (j_end >= batch_size) {
                    break;
                }
                int k = j_end;
                const int k_end = ((threadIdx.x + stride * 2) > batch_size) ? batch_size : (threadIdx.x + stride * 2);
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (residuals[j] < residuals[k]) {
                        residuals_copy[l] = residuals[j];
                        indices_copy[l] = indices[j];
                        j++;
                    }
                    else {
                        residuals_copy[l] = residuals[k];
                        indices_copy[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = residuals_copy[j];
                        indices[j] = indices_copy[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = residuals_copy[k];
                        indices[k] = indices_copy[k];
                    }
                }
            }
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            // Epsilon-trimming
            index_low = 0;
            float abs_residual_low = std::abs(residuals[0]);
            index_high = batch_size - 1;
            float abs_residual_high = std::abs(residuals[batch_size - 1]);
            for (int i = 0; i < (int)(batch_size * epsilon); i++) {
                if (abs_residual_low < abs_residual_high) {
                    residuals[index_high] = 0;
                    index_high--;
                    abs_residual_high = std::abs(residuals[index_high]);
                }
                else {
                    residuals[index_low] = 0;
                    index_low++;
                    abs_residual_low = std::abs(residuals[index_low]);
                }
            }
        }

        // Z-score-trimming
        __syncthreads();
        while (true) {
            residuals_copy[threadIdx.x] = residuals[threadIdx.x];
            __syncthreads();
            if (block_size > 512 && threadIdx.x < 512) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 512];
            }
            __syncthreads();
            if (block_size > 256 && threadIdx.x < 256) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 256];
            }
            __syncthreads();
            if (block_size > 128 && threadIdx.x < 128) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 128];
            }
            __syncthreads();
            if (block_size > 64 && threadIdx.x < 64) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 64];
            }
            __syncthreads();
            if (block_size > 32 && threadIdx.x < 32) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 32];
            }
            __syncthreads();
            if (block_size > 16 && threadIdx.x < 16) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 16];
            }
            __syncthreads();
            if (block_size > 8 && threadIdx.x < 8) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 8];
            }
            __syncthreads();
            if (block_size > 4 && threadIdx.x < 4) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 4];
            }
            __syncthreads();
            if (block_size > 2 && threadIdx.x < 2) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 2];
            }
            __syncthreads();
            if (block_size > 1 && threadIdx.x < 1) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 1];
            }
            __syncthreads();
            const float mean = residuals_copy[0] / (index_high - index_low);
            const float diff = residuals[threadIdx.x] - mean;
            __syncthreads();
            residuals_copy[threadIdx.x] = (threadIdx.x >= index_low && threadIdx.x <= index_high) ? diff * diff : 0;
            __syncthreads();
            if (block_size > 512 && threadIdx.x < 512) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 512];
            }
            __syncthreads();
            if (block_size > 256 && threadIdx.x < 256) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 256];
            }
            __syncthreads();
            if (block_size > 128 && threadIdx.x < 128) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 128];
            }
            __syncthreads();
            if (block_size > 64 && threadIdx.x < 64) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 64];
            }
            __syncthreads();
            if (block_size > 32 && threadIdx.x < 32) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 32];
            }
            __syncthreads();
            if (block_size > 16 && threadIdx.x < 16) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 16];
            }
            __syncthreads();
            if (block_size > 8 && threadIdx.x < 8) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 8];
            }
            __syncthreads();
            if (block_size > 4 && threadIdx.x < 4) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 4];
            }
            __syncthreads();
            if (block_size > 2 && threadIdx.x < 2) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 2];
            }
            __syncthreads();
            if (block_size > 1 && threadIdx.x < 1) {
                residuals_copy[threadIdx.x] += residuals_copy[threadIdx.x + 1];
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                float stdev = sqrt(residuals_copy[0] / (index_high - index_low));
                z_score_trimming_flag_converged = true;
                const float threashold_low = mean - stdev * z_score_trimming_threashold;
                while (residuals[index_low] < threashold_low) {
                    residuals[index_low] = 0;
                    index_low++;
                    z_score_trimming_flag_converged = false;
                }
                const float threashold_high = mean + stdev * z_score_trimming_threashold;
                while (residuals[index_high] > threashold_high) {
                    residuals[index_high] = 0;
                    index_high--;
                    z_score_trimming_flag_converged = false;
                }
            }
            __syncthreads();
            if (z_score_trimming_flag_converged) {
                break;
            }
        }

        // Calculate Huber Loss and gradient
        if (threadIdx.x == 0) {
            loss = 0;
            for (int i = 0; i < dimension; i++) {
                gradient[i] = 0;
            }
        }
        __syncthreads();
        const float residual = residuals[threadIdx.x];
        const float abs_residual = std::abs(residual);
        if (abs_residual <= huber_loss_threashold) {
            atomicAdd(&loss, residual * residual / 2);
            for (int j = 0; j < dimension; j++) {
                atomicAdd(gradient + j, residual * X[j * sample_size + indices[threadIdx.x]]);
            }
        }
        else {
            atomicAdd(&loss, abs_residual * huber_loss_threashold - huber_loss_threashold * huber_loss_threashold / 2);
            for (int j = 0; j < dimension; j++) {
                atomicAdd(gradient + j, ((residual > 0) - (residual < 0)) * X[j * sample_size + indices[threadIdx.x]] * huber_loss_threashold);
            }
        }

        // Update weights
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 0; i < dimension; i++) {
                w[i] -= learning_rate * gradient[i] / (index_high - index_low + 1);
            }
        }

        // Check convergence
        __syncthreads();
        if (std::abs((loss - prev_loss) / prev_loss) < 1e-4) {
            //break;
        }
        prev_loss = loss;
    }

    // Write to global memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < dimension; i++) {
            _w[blockIdx.x * dimension + i] = w[i];
        }
    }
}

int main(void) {
    float X[sample_size * dimension * model_count];
    float y[sample_size * model_count];
    float w[dimension * model_count];
    srand(clock());

    // Read training data
    FILE* f = fopen("in.txt", "r");
    for (int i = 0; i < sample_size; i++) {
        X[i] = 1;
        for (int j = 1; j < dimension; j++) {
            fscanf(f, "%f", X + j * sample_size + i);
        }
        fscanf(f, "%f", y + i);
    }
    for (int i = 0; i < model_count; i++) {
        memcpy(X + i * sample_size * dimension, X, sample_size * dimension * sizeof(float));
        memcpy(y + i * sample_size, y, sample_size * sizeof(float));
    }
    fclose(f);

    // Allocate device memory
    float* device_X, * device_y, * device_w;
    cudaMalloc(&device_X, sample_size * dimension * model_count * sizeof(float));
    cudaMalloc(&device_y, sample_size * model_count * sizeof(float));
    cudaMalloc(&device_w, dimension * model_count * sizeof(float));

    // Copy input to device memory
    cudaMemcpy(device_X, X, sample_size * dimension * model_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, y, sample_size * model_count * sizeof(float), cudaMemcpyHostToDevice);

    // Start timing
    clock_t clk = clock();

    kernel<batch_size> << <model_count, batch_size >> > (device_X, device_y, device_w, clk);

    // Stop timing
    cudaDeviceSynchronize();
    clk = clock() - clk;
    printf("CUDA running time:\t%.3fms\n", (double)clk / CLOCKS_PER_SEC * 1000);

    // Copy output to host memory
    cudaMemcpy(w, device_w, dimension * model_count * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_X);
    cudaFree(device_y);
    cudaFree(device_w);

    // Write the trained weights
    f = fopen("out.txt", "w");
    fprintf(f, "%f", w[(model_count - 1) * dimension]);
    for (int i = 1; i < dimension; i++) {
        fprintf(f, " %f", w[(model_count - 1) * dimension + i]);
    }
    fclose(f);

    return 0;
}
