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
    __shared__ int shared_int_batch_size_buffer[batch_size];
    __shared__ float residuals[batch_size];
    __shared__ float shared_float_batch_size_buffer[batch_size];
    __shared__ float gradient[dimension * batch_size];
    __shared__ float prev_loss;
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

        // Merge sort (absolute) residuals and permute the indices accordingly
        {
            if (block_size > 1 && threadIdx.x % 2 == 1) {
                if (abs(residuals[threadIdx.x]) < abs(residuals[threadIdx.x - 1])) {
                    const float tmp_float = residuals[threadIdx.x];
                    residuals[threadIdx.x] = residuals[threadIdx.x - 1];
                    residuals[threadIdx.x - 1] = tmp_float;
                    const int tmp_int = indices[threadIdx.x];
                    indices[threadIdx.x] = indices[threadIdx.x - 1];
                    indices[threadIdx.x - 1] = tmp_int;
                }
            }
            __syncthreads();
            if (block_size > 2 && threadIdx.x % 4 == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + 2;
                int k = j_end;
                const int k_end = threadIdx.x + 4;
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (abs(residuals[j]) < abs(residuals[k])) {
                        shared_float_batch_size_buffer[l] = residuals[j];
                        shared_int_batch_size_buffer[l] = indices[j];
                        j++;
                    }
                    else {
                        shared_float_batch_size_buffer[l] = residuals[k];
                        shared_int_batch_size_buffer[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = shared_float_batch_size_buffer[j];
                        indices[j] = shared_int_batch_size_buffer[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = shared_float_batch_size_buffer[k];
                        indices[k] = shared_int_batch_size_buffer[k];
                    }
                }
            }
            __syncthreads();
            if (block_size > 4 && threadIdx.x % 8 == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + 4;
                int k = j_end;
                const int k_end = threadIdx.x + 8;
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (abs(residuals[j]) < abs(residuals[k])) {
                        shared_float_batch_size_buffer[l] = residuals[j];
                        shared_int_batch_size_buffer[l] = indices[j];
                        j++;
                    }
                    else {
                        shared_float_batch_size_buffer[l] = residuals[k];
                        shared_int_batch_size_buffer[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = shared_float_batch_size_buffer[j];
                        indices[j] = shared_int_batch_size_buffer[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = shared_float_batch_size_buffer[k];
                        indices[k] = shared_int_batch_size_buffer[k];
                    }
                }
            }
            __syncthreads();
            if (block_size > 8 && threadIdx.x % 16 == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + 8;
                int k = j_end;
                const int k_end = threadIdx.x + 16;
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (abs(residuals[j]) < abs(residuals[k])) {
                        shared_float_batch_size_buffer[l] = residuals[j];
                        shared_int_batch_size_buffer[l] = indices[j];
                        j++;
                    }
                    else {
                        shared_float_batch_size_buffer[l] = residuals[k];
                        shared_int_batch_size_buffer[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = shared_float_batch_size_buffer[j];
                        indices[j] = shared_int_batch_size_buffer[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = shared_float_batch_size_buffer[k];
                        indices[k] = shared_int_batch_size_buffer[k];
                    }
                }
            }
            __syncthreads();
            if (block_size > 16 && threadIdx.x % 32 == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + 16;
                int k = j_end;
                const int k_end = threadIdx.x + 32;
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (abs(residuals[j]) < abs(residuals[k])) {
                        shared_float_batch_size_buffer[l] = residuals[j];
                        shared_int_batch_size_buffer[l] = indices[j];
                        j++;
                    }
                    else {
                        shared_float_batch_size_buffer[l] = residuals[k];
                        shared_int_batch_size_buffer[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = shared_float_batch_size_buffer[j];
                        indices[j] = shared_int_batch_size_buffer[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = shared_float_batch_size_buffer[k];
                        indices[k] = shared_int_batch_size_buffer[k];
                    }
                }
            }
            __syncthreads();
            if (block_size > 32 && threadIdx.x % 64 == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + 32;
                int k = j_end;
                const int k_end = threadIdx.x + 64;
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (abs(residuals[j]) < abs(residuals[k])) {
                        shared_float_batch_size_buffer[l] = residuals[j];
                        shared_int_batch_size_buffer[l] = indices[j];
                        j++;
                    }
                    else {
                        shared_float_batch_size_buffer[l] = residuals[k];
                        shared_int_batch_size_buffer[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = shared_float_batch_size_buffer[j];
                        indices[j] = shared_int_batch_size_buffer[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = shared_float_batch_size_buffer[k];
                        indices[k] = shared_int_batch_size_buffer[k];
                    }
                }
            }
            __syncthreads();
            if (block_size > 64 && threadIdx.x % 128 == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + 64;
                int k = j_end;
                const int k_end = threadIdx.x + 128;
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (abs(residuals[j]) < abs(residuals[k])) {
                        shared_float_batch_size_buffer[l] = residuals[j];
                        shared_int_batch_size_buffer[l] = indices[j];
                        j++;
                    }
                    else {
                        shared_float_batch_size_buffer[l] = residuals[k];
                        shared_int_batch_size_buffer[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = shared_float_batch_size_buffer[j];
                        indices[j] = shared_int_batch_size_buffer[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = shared_float_batch_size_buffer[k];
                        indices[k] = shared_int_batch_size_buffer[k];
                    }
                }
            }
            __syncthreads();
            if (block_size > 128 && threadIdx.x % 256 == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + 128;
                int k = j_end;
                const int k_end = threadIdx.x + 256;
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (abs(residuals[j]) < abs(residuals[k])) {
                        shared_float_batch_size_buffer[l] = residuals[j];
                        shared_int_batch_size_buffer[l] = indices[j];
                        j++;
                    }
                    else {
                        shared_float_batch_size_buffer[l] = residuals[k];
                        shared_int_batch_size_buffer[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = shared_float_batch_size_buffer[j];
                        indices[j] = shared_int_batch_size_buffer[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = shared_float_batch_size_buffer[k];
                        indices[k] = shared_int_batch_size_buffer[k];
                    }
                }
            }
            __syncthreads();
            if (block_size > 256 && threadIdx.x % 512 == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + 256;
                int k = j_end;
                const int k_end = threadIdx.x + 512;
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (abs(residuals[j]) < abs(residuals[k])) {
                        shared_float_batch_size_buffer[l] = residuals[j];
                        shared_int_batch_size_buffer[l] = indices[j];
                        j++;
                    }
                    else {
                        shared_float_batch_size_buffer[l] = residuals[k];
                        shared_int_batch_size_buffer[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = shared_float_batch_size_buffer[j];
                        indices[j] = shared_int_batch_size_buffer[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = shared_float_batch_size_buffer[k];
                        indices[k] = shared_int_batch_size_buffer[k];
                    }
                }
            }
            __syncthreads();
            if (block_size > 512 && threadIdx.x % 1024 == 0) {
                int j = threadIdx.x;
                const int j_end = threadIdx.x + 512;
                int k = j_end;
                const int k_end = threadIdx.x + 1024;
                int l = threadIdx.x;
                while (j != j_end && k != k_end) {
                    if (abs(residuals[j]) < abs(residuals[k])) {
                        shared_float_batch_size_buffer[l] = residuals[j];
                        shared_int_batch_size_buffer[l] = indices[j];
                        j++;
                    }
                    else {
                        shared_float_batch_size_buffer[l] = residuals[k];
                        shared_int_batch_size_buffer[l] = indices[k];
                        k++;
                    }
                    l++;
                }
                if (j == j_end) {
                    for (j = threadIdx.x; j < l; j++) {
                        residuals[j] = shared_float_batch_size_buffer[j];
                        indices[j] = shared_int_batch_size_buffer[j];
                    }
                }
                else {
                    for (k = j_end - 1; k >= j; k--) {
                        residuals[k + k_end - j_end] = residuals[k];
                        indices[k + k_end - j_end] = indices[k];
                    }
                    for (k = threadIdx.x; k < l; k++) {
                        residuals[k] = shared_float_batch_size_buffer[k];
                        indices[k] = shared_int_batch_size_buffer[k];
                    }
                }
            }
        }

        // Epsilon-trimming
        __syncthreads();
        if (threadIdx.x >= batch_size * (1 - epsilon)) {
            residuals[threadIdx.x] = 0;
        }

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
                __syncthreads();
                if (block_size > 16 && threadIdx.x < 16) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 16];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 16];
                }
                __syncthreads();
                if (block_size > 8 && threadIdx.x < 8) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 8];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 8];
                }
                __syncthreads();
                if (block_size > 4 && threadIdx.x < 4) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 4];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 4];
                }
                __syncthreads();
                if (block_size > 2 && threadIdx.x < 2) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 2];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 2];
                }
                __syncthreads();
                if (block_size > 1 && threadIdx.x < 1) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 1];
                    shared_int_batch_size_buffer[threadIdx.x] += shared_int_batch_size_buffer[threadIdx.x + 1];
                }
                __syncthreads();
            }
            const float mean = shared_float_batch_size_buffer[0] / shared_int_batch_size_buffer[0];
            const float diff = residuals[threadIdx.x] - mean;
            {
                __syncthreads();
                shared_float_batch_size_buffer[threadIdx.x] = (residuals[threadIdx.x] != 0) * diff * diff / shared_int_batch_size_buffer[0];
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
                __syncthreads();
                if (block_size > 16 && threadIdx.x < 16) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 16];
                }
                __syncthreads();
                if (block_size > 8 && threadIdx.x < 8) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 8];
                }
                __syncthreads();
                if (block_size > 4 && threadIdx.x < 4) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 4];
                }
                __syncthreads();
                if (block_size > 2 && threadIdx.x < 2) {
                    shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 2];
                }
                __syncthreads();
            }
            if (block_size > 1 && threadIdx.x == 0) {
                shared_float_batch_size_buffer[threadIdx.x] = sqrt(shared_float_batch_size_buffer[0] + shared_float_batch_size_buffer[1]);
                z_score_trimming_flag_converged = true;
            }
            __syncthreads();
            const float stdev = shared_float_batch_size_buffer[0];
            if (residuals[threadIdx.x] != 0 && (residuals[threadIdx.x] < mean - stdev * z_score_trimming_threashold || residuals[threadIdx.x] >  mean + stdev * z_score_trimming_threashold)) {
                residuals[threadIdx.x] = 0;
                z_score_trimming_flag_converged = false;
            }
            __syncthreads();
            if (z_score_trimming_flag_converged) {
                break;
            }
        }

        // Calculate Huber Loss and gradient
        const float residual = residuals[threadIdx.x];
        const float abs_residual = abs(residual);
        if (residuals[threadIdx.x / 32 * 32] <= huber_loss_threashold) {
            shared_float_batch_size_buffer[threadIdx.x] = residual * residual / 2;
            for (int j = 0; j < dimension; j++) {
                gradient[j * batch_size + threadIdx.x] = residual * X[j * sample_size + indices[threadIdx.x]];
            }
        }
        else {
            shared_float_batch_size_buffer[threadIdx.x] = abs_residual * huber_loss_threashold - huber_loss_threashold * huber_loss_threashold / 2;
            for (int j = 0; j < dimension; j++) {
                gradient[j * batch_size + threadIdx.x] = ((residual > 0) - (residual < 0)) * X[j * sample_size + indices[threadIdx.x]] * huber_loss_threashold;
            }
        }
        {
            __syncthreads();
            if (block_size > 512 && threadIdx.x < 512) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 512];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 512];
                }
            }
            __syncthreads();
            if (block_size > 256 && threadIdx.x < 256) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 256];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 256];
                }
            }
            __syncthreads();
            if (block_size > 128 && threadIdx.x < 128) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 128];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 128];
                }
            }
            __syncthreads();
            if (block_size > 64 && threadIdx.x < 64) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 64];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 64];
                }
            }
            __syncthreads();
            if (block_size > 32 && threadIdx.x < 32) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 32];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 32];
                }
            }
            __syncthreads();
            if (block_size > 16 && threadIdx.x < 16) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 16];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 16];
                }
            }
            __syncthreads();
            if (block_size > 8 && threadIdx.x < 8) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 8];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 8];
                }
            }
            __syncthreads();
            if (block_size > 4 && threadIdx.x < 4) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 4];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 4];
                }
            }
            __syncthreads();
            if (block_size > 2 && threadIdx.x < 2) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 2];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 2];
                }
            }
            __syncthreads();
            if (block_size > 1 && threadIdx.x < 1) {
                shared_float_batch_size_buffer[threadIdx.x] += shared_float_batch_size_buffer[threadIdx.x + 1];
                for (int j = 0; j < dimension; j++) {
                    gradient[j * batch_size + threadIdx.x] += gradient[j * batch_size + threadIdx.x + 1];
                }
            }
        }

        // Update weights
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 0; i < dimension; i++) {
                w[i] -= learning_rate * gradient[i * batch_size] / shared_int_batch_size_buffer[0];
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

    kernel<batch_size><<<model_count, batch_size>>>(device_X, device_y, device_w, clk);

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
