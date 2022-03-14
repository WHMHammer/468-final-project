#include <cstdio>
#include <ctime>

constexpr int model_count = 100;

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
    float prev_loss;
    int sample_index_base = 0;

    for (int _ = -1; _ < max_iter; _++) {
        // Sample consecutive batches in a Round Robin manner
        const int sample_index = (sample_index_base + threadIdx.x) % sample_size;
        sample_index_base += batch_size;
        for (int i = 0; i < dimension; i++) {
            X[i * batch_size + threadIdx.x] = global_X[blockIdx.x * sample_size * dimension + i * sample_size + sample_index];
        }
        y[threadIdx.x] = global_y[blockIdx.x * sample_size + sample_index];
        indices[threadIdx.x] = threadIdx.x;

        // Calculate residuals
        __syncthreads();
        residuals[threadIdx.x] = -y[threadIdx.x];
        for (int i = 0; i < dimension; i++) {
            residuals[threadIdx.x] += X[i * batch_size + threadIdx.x] * w[i];
        }

        // Odd-even sort (absolute) residuals and permute the indices accordingly
        __syncthreads();
        for (int i = 0; i < batch_size / 2; i++) {
            if (threadIdx.x % 2 == 0 && threadIdx.x != 0 && abs(residuals[threadIdx.x - 1]) > abs(residuals[threadIdx.x])) {
                const float tmp_float = residuals[threadIdx.x];
                residuals[threadIdx.x] = residuals[threadIdx.x - 1];
                residuals[threadIdx.x - 1] = tmp_float;
                const int tmp_int = indices[threadIdx.x];
                indices[threadIdx.x] = indices[threadIdx.x - 1];
                indices[threadIdx.x - 1] = tmp_int;
            }
            __syncthreads();
            if (threadIdx.x % 2 == 0 && abs(residuals[threadIdx.x]) > abs(residuals[threadIdx.x + 1])) {
                const float tmp_float = residuals[threadIdx.x];
                residuals[threadIdx.x] = residuals[threadIdx.x + 1];
                residuals[threadIdx.x + 1] = tmp_float;
                const int tmp_int = indices[threadIdx.x];
                indices[threadIdx.x] = indices[threadIdx.x + 1];
                indices[threadIdx.x + 1] = tmp_int;
            }
            __syncthreads();
        }

        // Epsilon-trimming
        __syncthreads();
        residuals[threadIdx.x] *= threadIdx.x < batch_size * (1 - epsilon);

        // Z-score-trimming
        __syncthreads();
        while (true) {
            {
                shared_float_batch_size_buffer[threadIdx.x] = residuals[threadIdx.x];
                shared_int_batch_size_buffer[threadIdx.x] = residuals[threadIdx.x] != 0;
                __syncthreads();
                if (block_size > 512 && threadIdx.x < 512) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 512];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 512];
                }
                __syncthreads();
                if (block_size > 256 && threadIdx.x < 256) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 256];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 256];
                }
                __syncthreads();
                if (block_size > 128 && threadIdx.x < 128) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 128];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 128];
                }
                __syncthreads();
                if (block_size > 64 && threadIdx.x < 64) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 64];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 64];
                }
                __syncthreads();
                if (block_size > 32 && threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 32];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 32];
                }
                __syncwarp();
                if (block_size > 16 && threadIdx.x < 16) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 16];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 16];
                }
                __syncwarp();
                if (block_size > 8 && threadIdx.x < 8) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 8];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 8];
                }
                __syncwarp();
                if (block_size > 4 && threadIdx.x < 4) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 4];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 4];
                }
                __syncwarp();
                if (block_size > 2 && threadIdx.x < 2) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 2];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 2];
                }
                __syncwarp();
                if (block_size > 1 && threadIdx.x == 0) {
                    shared_int_batch_size_buffer[0] += shared_int_batch_size_buffer[1];
                }
            }
            __syncthreads();
            const float mean = (shared_float_batch_size_buffer[0] + shared_float_batch_size_buffer[1]) / shared_int_batch_size_buffer[0];
            const float diff = residuals[threadIdx.x] - mean;
            __syncthreads();
            shared_float_batch_size_buffer[threadIdx.x] = (residuals[threadIdx.x] != 0) * diff * diff / shared_int_batch_size_buffer[0];
            {
                __syncthreads();
                if (block_size > 512 && threadIdx.x < 512) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 512];
                }
                __syncthreads();
                if (block_size > 256 && threadIdx.x < 256) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 256];
                }
                __syncthreads();
                if (block_size > 128 && threadIdx.x < 128) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 128];
                }
                __syncthreads();
                if (block_size > 64 && threadIdx.x < 64) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 64];
                }
                __syncthreads();
                if (block_size > 32 && threadIdx.x < 32) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 32];
                }
                __syncwarp();
                if (block_size > 16 && threadIdx.x < 16) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 16];
                }
                __syncwarp();
                if (block_size > 8 && threadIdx.x < 8) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 8];
                }
                __syncwarp();
                if (block_size > 4 && threadIdx.x < 4) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 4];
                }
                __syncwarp();
                if (block_size > 2 && threadIdx.x < 2) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 2];
                }
            }
            __syncwarp();
            if (block_size > 1 && threadIdx.x == 0) {
                shared_float_batch_size_buffer[0] = sqrt(shared_float_batch_size_buffer[0] + shared_float_batch_size_buffer[1]);
                flag = true;
            }
            __syncthreads();
            const float stdev = shared_float_batch_size_buffer[0];
            if (residuals[threadIdx.x] != 0 && (residuals[threadIdx.x] < mean - stdev * z_score_trimming_threashold || residuals[threadIdx.x] >  mean + stdev * z_score_trimming_threashold)) {
                residuals[threadIdx.x] = 0;
                flag = false;
            }
            __syncthreads();
            if (flag) {
                break;
            }
        }

        // Calculate Huber Loss and gradient
        const float residual = residuals[threadIdx.x];
        const float abs_residual = abs(residual);
        if (abs(residuals[threadIdx.x / 32 * 32]) <= huber_loss_threashold) {
            shared_float_batch_size_buffer[threadIdx.x] = residual * residual / 2;
            for (int i = 0; i < dimension; i++) {
                gradient[i * batch_size + threadIdx.x] = residual * X[i * batch_size + indices[threadIdx.x]];
            }
        }
        else {
            shared_float_batch_size_buffer[threadIdx.x] = abs_residual * huber_loss_threashold - huber_loss_threashold * huber_loss_threashold / 2;
            for (int i = 0; i < dimension; i++) {
                gradient[i * batch_size + threadIdx.x] = ((residual > 0) - (residual < 0)) * X[i * batch_size + indices[threadIdx.x]] * huber_loss_threashold;
            }
        }
        {
            __syncthreads();
            if (block_size > 512 && threadIdx.x < 512) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 512];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 512];
                }
            }
            __syncthreads();
            if (block_size > 256 && threadIdx.x < 256) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 256];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 256];
                }
            }
            __syncthreads();
            if (block_size > 128 && threadIdx.x < 128) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 128];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 128];
                }
            }
            __syncthreads();
            if (block_size > 64 && threadIdx.x < 64) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 64];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 64];
                }
            }
            __syncthreads();
            if (block_size > 32 && threadIdx.x < 32) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 32];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 32];
                }
            }
            __syncwarp();
            if (block_size > 16 && threadIdx.x < 16) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 16];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 16];
                }
            }
            __syncwarp();
            if (block_size > 8 && threadIdx.x < 8) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 8];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 8];
                }
            }
            __syncwarp();
            if (block_size > 4 && threadIdx.x < 4) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 4];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 4];
                }
            }
            __syncwarp();
            if (block_size > 2 && threadIdx.x < 2) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 2];
                for (int i = 0; i < dimension; i++) {
                    gradient[i * batch_size + threadIdx.x] += gradient[i * batch_size + threadIdx.x + 2];
                }
            }
        }
        __syncwarp();
        if (block_size > 1 && threadIdx.x == 0) {
            shared_float_batch_size_buffer[0] += shared_float_batch_size_buffer[0];
            // Update weights
            for (int i = 0; i < dimension; i++) {
                w[i] -= learning_rate * (gradient[i * batch_size] + gradient[i * batch_size + 1]) / shared_int_batch_size_buffer[0];
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
    kernel<batch_size><<<1, batch_size>>>(device_X, device_y, device_w);
    cudaDeviceSynchronize();
    clock_t clk = clock();

    kernel<batch_size><<<model_count, batch_size>>>(device_X, device_y, device_w);

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
