#include <cstdio>
#include <ctime>
#include <curand_kernel.h>

// Hyperparameters
constexpr float huber_loss_threashold = 10;
constexpr float z_score_trimming_threashold = 2;
constexpr float epsilon = 0.49;
constexpr float learning_rate = 0.01;
constexpr int batch_size = 128;
constexpr int max_iter = 100000;

constexpr int sample_size = 1000;
constexpr int dimension = 6;

template<int block_size> __global__ void kernel(float* const X, float* const y, float* const w, const clock_t seed) {
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
    curand_init(seed, threadIdx.x, 0, &state);

    if (threadIdx.x == 0) {
        // Initialize weights
        for (int i = 0; i < dimension; i++) {
            w[i] = 1;
        }
        prev_loss = 0;
    }

    for (int _ = -1; _ < max_iter; _++) {
        __syncthreads();
        // Sample with replacement
        indices[threadIdx.x] = curand_uniform(&state) * sample_size;

        // Calculate residuals
        float residual = -y[indices[threadIdx.x]];
        for (int j = 0; j < dimension; j++) {
            residual += X[indices[threadIdx.x] * dimension + j] * w[j];
        }
        residuals[threadIdx.x] = residual;

        __syncthreads();
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
                    for (k = j_end - 1;k >= j; k--) {
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

        // Z-score trimming
        __syncthreads();
        while (true) {
            residuals_copy[threadIdx.x] = residuals[threadIdx.x];
            __syncthreads();
            if (block_size > 512 && threadIdx.x < 512) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 512];
            }
            __syncthreads();
            if (block_size > 256 && threadIdx.x < 256) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 256];
            }
            __syncthreads();
            if (block_size > 128 && threadIdx.x < 128) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 128];
            }
            __syncthreads();
            if (block_size > 64 && threadIdx.x < 64) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 64];
            }
            __syncthreads();
            if (block_size > 32 && threadIdx.x < 32) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 32];
            }
            __syncthreads();
            if (block_size > 16 && threadIdx.x < 16) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 16];
            }
            __syncthreads();
            if (block_size > 8 && threadIdx.x < 8) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 8];
            }
            __syncthreads();
            if (block_size > 4 && threadIdx.x < 4) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 4];
            }
            __syncthreads();
            if (block_size > 2 && threadIdx.x < 2) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 2];
            }
            __syncthreads();
            if (block_size > 1 && threadIdx.x < 1) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 1];
            }
            __syncthreads();
            const float mean = residuals[0];
            const float diff = residuals[threadIdx.x] - mean;
            residuals_copy[threadIdx.x] = diff * diff;
            __syncthreads();
            if (block_size > 512 && threadIdx.x < 512) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 512];
            }
            __syncthreads();
            if (block_size > 256 && threadIdx.x < 256) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 256];
            }
            __syncthreads();
            if (block_size > 128 && threadIdx.x < 128) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 128];
            }
            __syncthreads();
            if (block_size > 64 && threadIdx.x < 64) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 64];
            }
            __syncthreads();
            if (block_size > 32 && threadIdx.x < 32) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 32];
            }
            __syncthreads();
            if (block_size > 16 && threadIdx.x < 16) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 16];
            }
            __syncthreads();
            if (block_size > 8 && threadIdx.x < 8) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 8];
            }
            __syncthreads();
            if (block_size > 4 && threadIdx.x < 4) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 4];
            }
            __syncthreads();
            if (block_size > 2 && threadIdx.x < 2) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 2];
            }
            __syncthreads();
            if (block_size > 1 && threadIdx.x < 1) {
                residuals_copy[threadIdx.x] += residuals[threadIdx.x + 1];
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const float stdev = sqrt(residuals_copy[0] / (index_high - index_low));
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

        // Huber loss
        __syncthreads();
        if (threadIdx.x == 0) {
            loss = 0;
            for (int i = 0; i < dimension; i++) {
                gradient[i] = 0;
            }
        }
        __syncthreads();
        residual = residuals[threadIdx.x];
        const float abs_residual = std::abs(residual);
        if (abs_residual <= huber_loss_threashold) {
            atomicAdd(&loss, residual * residual / 2);
            for (int j = 0; j < dimension; j++) {
                atomicAdd(gradient + j, residual * X[indices[threadIdx.x] * dimension + j]);
            }
        }
        else {
            atomicAdd(&loss, abs_residual * huber_loss_threashold - huber_loss_threashold * huber_loss_threashold / 2);
            for (int j = 0; j < dimension; j++) {
                atomicAdd(gradient + j, ((residual > 0) - (residual < 0)) * X[indices[threadIdx.x] * dimension + j] * huber_loss_threashold);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 0; i < dimension; i++) {
                gradient[i] /= batch_size;
            }
            loss /= batch_size;
        }

        // Update weights
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 0; i < dimension; i++) {
                w[i] -= learning_rate * gradient[i];
            }
        }

        // Check convergence
        __syncthreads();
        if (std::abs((loss - prev_loss) / prev_loss) < 1e-5) {
            return;
        }

        if (threadIdx.x == 0) {
            prev_loss = loss;
        }
    }
}

int main(void) {
    float X[sample_size * dimension];
    float y[sample_size];
    float w[dimension];
    srand(clock());

    // Read training data
    FILE* f = fopen("in.txt", "r");
    for (int i = 0; i < sample_size; i++) {
        X[i * dimension] = 1;
        for (int j = 1; j < dimension; j++) {
            fscanf(f, "%f", X + i * dimension + j);
        }
        fscanf(f, "%f", y + i);
    }
    fclose(f);

    // Allocate device memory
    float* device_X, * device_y, * device_w;
    cudaMalloc(&device_X, sample_size * dimension * sizeof(float));
    cudaMalloc(&device_y, sample_size * sizeof(float));
    cudaMalloc(&device_w, dimension * sizeof(float));

    // Copy input to device memory
    cudaMemcpy(device_X, X, sample_size * dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, y, sample_size * sizeof(float), cudaMemcpyHostToDevice);

    // Start timing
    clock_t clk = clock();

    kernel<batch_size> << <1, batch_size >> > (device_X, device_y, device_w, clk);

    // Stop timing
    cudaDeviceSynchronize();
    clk = clock() - clk;
    printf("CUDA running time:\t%.3fms\n", (double)clk / CLOCKS_PER_SEC * 1000);

    // Copy output to host memory
    cudaMemcpy(w, device_w, dimension * sizeof(float), cudaMemcpyDeviceToHost);

    // Write the trained weights
    f = fopen("out.txt", "w");
    fprintf(f, "%f", *w);
    for (int i = 1; i < dimension; i++) {
        fprintf(f, " %f", w[i]);
    }
    fclose(f);

    return 0;
}
