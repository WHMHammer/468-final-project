#include <cstdio>
#include <ctime>

constexpr int model_count = 1000;

// Hyperparameters
constexpr float huber_loss_threashold = 10;
constexpr float z_score_trimming_threashold = 2;
constexpr float epsilon = 0.49;
constexpr float learning_rate = 0.01;
constexpr int batch_size = 128;
constexpr int max_iter = 10000;

constexpr int sample_size = 1024;
constexpr int dimension = 6;

template<int block_size> __global__ void kernel(float* const global_X, float* const global_y, float* const global_w) {
    __shared__ float X[dimension * batch_size];
    __shared__ float y[batch_size];
    __shared__ float w[dimension];
    __shared__ int indices[batch_size];
    __shared__ float residuals[batch_size];
    __shared__ float gradient[dimension * batch_size];
    __shared__ float shared_float_batch_size_buffer[batch_size];
    __shared__ int shared_int_batch_size_buffer[batch_size];
    __shared__ bool flag;

    // Initialization
    if (threadIdx.x == 0) {
        for (int i = 0; i < dimension; i++) {
            w[i] = 1;
        }
    }
    float prev_loss = 0;
    int sample_index_base = 0;

    for (int _ = -1; _ < max_iter; _++) {

        // Sample consecutive batches in a Round Robin manner
        for (int i = 0; i < dimension; i++) {
            X[i * batch_size + threadIdx.x] = global_X[blockIdx.x * sample_size * dimension + i * sample_size + (sample_index_base + threadIdx.x) % sample_size];
            X[i * batch_size + block_size + threadIdx.x] = global_X[blockIdx.x * sample_size * dimension + i * sample_size + (sample_index_base + block_size + threadIdx.x) % sample_size];
        }
        y[threadIdx.x] = global_y[blockIdx.x * sample_size + (sample_index_base + threadIdx.x) % sample_size];
        y[block_size + threadIdx.x] = global_y[blockIdx.x * sample_size + (sample_index_base + block_size + threadIdx.x) % sample_size];
        indices[threadIdx.x] = threadIdx.x;
        indices[block_size + threadIdx.x] = block_size + threadIdx.x;
        sample_index_base += batch_size;

        // Calculate residuals
        __syncthreads();
        residuals[threadIdx.x] = -y[threadIdx.x];
        residuals[block_size + threadIdx.x] = -y[block_size + threadIdx.x];
        for (int i = 0; i < dimension; i++) {
            residuals[threadIdx.x] += X[i * batch_size + threadIdx.x] * w[i];
            residuals[block_size + threadIdx.x] += X[i * batch_size + block_size + threadIdx.x] * w[i];
        }

        // Sort (absolute) residuals and permute the indices accordingly
        __syncthreads();
        int count0 = 0;
        int count1 = 0;
        for (int i = 0; i < batch_size; i++) {
            count0 += abs(residuals[threadIdx.x]) > abs(residuals[i]) || (abs(residuals[threadIdx.x]) == abs(residuals[i]) && threadIdx.x > i);
            count1 += abs(residuals[threadIdx.x + block_size]) > abs(residuals[i]) || (abs(residuals[threadIdx.x + block_size]) == abs(residuals[i]) && (threadIdx.x + block_size) > i);
        }
        shared_float_batch_size_buffer[count0] = residuals[threadIdx.x];
        shared_int_batch_size_buffer[count0] = indices[threadIdx.x];
        shared_float_batch_size_buffer[count1] = residuals[threadIdx.x + block_size];
        shared_int_batch_size_buffer[count1] = indices[threadIdx.x + block_size];
        __syncthreads();
        residuals[threadIdx.x] = shared_float_batch_size_buffer[threadIdx.x];
        indices[threadIdx.x] = shared_int_batch_size_buffer[threadIdx.x];
        residuals[threadIdx.x + block_size] = shared_float_batch_size_buffer[threadIdx.x + block_size];
        indices[threadIdx.x + block_size] = shared_int_batch_size_buffer[threadIdx.x + block_size];

        // Epsilon-trimming
        __syncthreads();
        residuals[threadIdx.x] *= threadIdx.x < batch_size * (1 - epsilon);
        residuals[block_size + threadIdx.x] *= block_size + threadIdx.x < batch_size* (1 - epsilon);

        // Z-score-trimming
        __syncthreads();
        while (true) {
            shared_float_batch_size_buffer[threadIdx.x] = residuals[threadIdx.x];
            shared_int_batch_size_buffer[threadIdx.x] = residuals[threadIdx.x] != 0;
            shared_float_batch_size_buffer[threadIdx.x + block_size] = residuals[threadIdx.x + block_size];
            shared_int_batch_size_buffer[threadIdx.x + block_size] = residuals[threadIdx.x + block_size] != 0;
            if (block_size == 1024) {
                __syncthreads();
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 1024];
                shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 1024];
            }
            if (block_size >= 512) {
                __syncthreads();
                if (threadIdx.x < 512) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 512];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 512];
                }
            }
            if (block_size >= 256) {
                __syncthreads();
                if (threadIdx.x < 256) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 256];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 256];
                }
            }
            if (block_size >= 128) {
                __syncthreads();
                if (threadIdx.x < 128) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 128];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 128];
                }
            }
            if (block_size >= 64) {
                __syncthreads();
                if (threadIdx.x < 64) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 64];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 64];
                }
            }
            if (block_size >= 32) {
                __syncthreads();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 32];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 32];
                }
            }
            if (block_size >= 16) {
                __syncwarp();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 16];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 16];
                }
            }
            if (block_size >= 8) {
                __syncwarp();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 8];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 8];
                }
            }
            if (block_size >= 4) {
                __syncwarp();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 4];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 4];
                }
            }
            if (block_size >= 2) {
                __syncwarp();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 2];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 2];
                }
            }
            if (block_size >= 1) {
                __syncwarp();
                if (threadIdx.x == 0) {
                    shared_float_batch_size_buffer[0] += shared_float_batch_size_buffer[1];
                    shared_int_batch_size_buffer[0] += shared_int_batch_size_buffer[1];
                }
            }
            __syncthreads();
            const float mean = shared_float_batch_size_buffer[0] / shared_int_batch_size_buffer[0];
            const float diff0 = residuals[threadIdx.x] - mean;
            const float diff1 = residuals[block_size + threadIdx.x] - mean;
            __syncthreads();
            shared_float_batch_size_buffer[threadIdx.x] = (residuals[threadIdx.x] != 0) * diff0 * diff0 / shared_int_batch_size_buffer[0];
            shared_float_batch_size_buffer[block_size + threadIdx.x] = (residuals[block_size + threadIdx.x] != 0) * diff1 * diff1 / shared_int_batch_size_buffer[0];
            if (block_size == 1024) {
                __syncthreads();
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 1024];
            }
            if (block_size >= 512) {
                __syncthreads();
                if (threadIdx.x < 512) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 512];
                }
            }
            if (block_size >= 256) {
                __syncthreads();
                if (threadIdx.x < 256) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 256];
                }
            }
            if (block_size >= 128) {
                __syncthreads();
                if (threadIdx.x < 128) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 128];
                }
            }
            if (block_size >= 64) {
                __syncthreads();
                if (threadIdx.x < 64) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 64];
                }
            }
            if (block_size >= 32) {
                __syncthreads();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 32];
                }
            }
            if (block_size >= 16) {
                __syncwarp();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 16];
                }
            }
            if (block_size >= 8) {
                __syncwarp();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 8];
                }
            }
            if (block_size >= 4) {
                __syncwarp();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 4];
                }
            }
            if (block_size >= 2) {
                __syncwarp();
                if (threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 2];
                }
            }
            if (block_size >= 1) {
                __syncwarp();
                if (threadIdx.x == 0) {
                    shared_float_batch_size_buffer[0] = sqrt(shared_float_batch_size_buffer[0] + shared_float_batch_size_buffer[1]);
                    flag = true;
                }
            }
            __syncthreads();
            const float stdev = shared_float_batch_size_buffer[0];
            const float threashold_low = mean - stdev * z_score_trimming_threashold;
            const float threashold_high = mean + stdev * z_score_trimming_threashold;
            if (residuals[threadIdx.x] != 0 && (residuals[threadIdx.x] < threashold_low || residuals[threadIdx.x] > threashold_high)) {
                residuals[threadIdx.x] = 0;
                flag = false;
            }
            if (residuals[block_size + threadIdx.x] != 0 && (residuals[block_size + threadIdx.x] < threashold_low || residuals[block_size + threadIdx.x] > threashold_high)) {
                residuals[block_size + threadIdx.x] = 0;
                flag = false;
            }
            __syncthreads();
            if (flag) {
                break;
            }
        }

        // Calculate Huber Loss and gradient
        const float residual0 = residuals[threadIdx.x];
        const float abs_residual0 = abs(residual0);
        const float flag_squared_loss0 = abs_residual0 < huber_loss_threashold;
        const float residual1 = residuals[block_size + threadIdx.x];
        const float abs_residual1 = abs(residual1);
        const float flag_squared_loss1 = abs_residual1 < huber_loss_threashold;
        shared_float_batch_size_buffer[threadIdx.x] = flag_squared_loss0 * residual0 * residual0 / 2 + !flag_squared_loss0 * (abs_residual0 * huber_loss_threashold - huber_loss_threashold * huber_loss_threashold / 2);
        shared_float_batch_size_buffer[block_size + threadIdx.x] = flag_squared_loss1 * residual1 * residual1 / 2 + !flag_squared_loss1 * (abs_residual1 * huber_loss_threashold - huber_loss_threashold * huber_loss_threashold / 2);
        for (int i = 0; i < dimension; i++) {
            gradient[i * batch_size + threadIdx.x] = flag_squared_loss0 * residual0 * X[i * batch_size + indices[threadIdx.x]] + !flag_squared_loss0 * ((residual0 > 0) - (residual0 < 0)) * X[i * batch_size + indices[threadIdx.x]] * huber_loss_threashold;
            gradient[i * batch_size + block_size + threadIdx.x] = flag_squared_loss1 * residual1 * X[i * batch_size + indices[block_size + threadIdx.x]] + !flag_squared_loss1 * ((residual1 > 0) - (residual1 < 0)) * X[i * batch_size + indices[block_size + threadIdx.x]] * huber_loss_threashold;
        }
        if (block_size == 1024) {
            __syncthreads();
            shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 1024];
            for (int i = 0; i < dimension; i++) {
                gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 1024];
            }
        }
        if (block_size >= 512) {
            __syncthreads();
            if (threadIdx.x < 512) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 512];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 512];
                }
            }
        }
        if (block_size >= 256) {
            __syncthreads();
            if (threadIdx.x < 256) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 256];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 256];
                }
            }
        }
        if (block_size >= 128) {
            __syncthreads();
            if (threadIdx.x < 128) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 128];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 128];
                }
            }
        }
        if (block_size >= 64) {
            __syncthreads();
            if (threadIdx.x < 64) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 64];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 64];
                }
            }
        }
        if (block_size >= 32) {
            __syncthreads();
            if (threadIdx.x < 32) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 32];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 32];
                }
            }
        }
        if (block_size >= 16) {
            __syncwarp();
            if (threadIdx.x < 32) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 16];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 16];
                }
            }
        }
        if (block_size >= 8) {
            __syncwarp();
            if (threadIdx.x < 32) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 8];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 8];
                }
            }
        }
        if (block_size >= 4) {
            __syncwarp();
            if (threadIdx.x < 32) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 4];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 4];
                }
            }
        }
        if (block_size >= 2) {
            __syncwarp();
            if (threadIdx.x < 32) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 2];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 2];
                }
            }
        }
        if (block_size >= 1) {
            __syncwarp();
            if (threadIdx.x == 0) {
                shared_float_batch_size_buffer[0] += shared_float_batch_size_buffer[0];

                // Update weights
                for (int i = 0; i < dimension; i++) {
                    w[i] -= learning_rate * (gradient[i * batch_size] + gradient[i * batch_size + 1]) / shared_int_batch_size_buffer[0];
                }
            }
        }

        // Check convergence
        __syncthreads();
        if (abs((shared_float_batch_size_buffer[0] - prev_loss) / prev_loss) < 1e-4) {
            //break;
        }
        prev_loss = shared_float_batch_size_buffer[0];
    }

    // Write to global memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < dimension; i++) {
            global_w[blockIdx.x * dimension + i] = w[i];
        }
    }
}

int main(void) {
    float* const X = new float[sample_size * dimension * model_count];
    float* const y = new float[sample_size * model_count];
    float* const w = new float[dimension * model_count];
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

    kernel<batch_size / 2><<<model_count, batch_size / 2>>>(device_X, device_y, device_w);

    // Stop timing
    cudaDeviceSynchronize();
    clk = clock() - clk;
    printf("CUDA running time:\t%.3fms\n", (double)clk / CLOCKS_PER_SEC * 1000);

    // Copy output to host memory
    cudaMemcpy(w, device_w, dimension * model_count * sizeof(float), cudaMemcpyDeviceToHost);

    // Write the trained weights
    f = fopen("out.txt", "w");
    fprintf(f, "%f", w[(model_count - 1) * dimension]);
    for (int i = 1; i < dimension; i++) {
        fprintf(f, " %f", w[(model_count - 1) * dimension + i]);
    }
    fclose(f);

    // Free resources
    cudaFree(device_X);
    cudaFree(device_y);
    cudaFree(device_w);
    delete[] X;
    delete[] y;
    delete[] w;

    return 0;
}
