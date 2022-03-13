#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int main(void) {
    srand(clock());

    // Read training data
    FILE* f = fopen("in.txt", "r");
    float* const _X = new float[sample_size * dimension * model_count];
    float* const _y = new float[sample_size * model_count];
    for (int i = 0; i < sample_size; i++) {
        _X[i * dimension] = 1;
        for (int j = 1; j < dimension; j++) {
            fscanf(f, "%f", _X + i * dimension + j);
        }
        fscanf(f, "%f", _y + i);
    }
    fclose(f);
    for (int i = 0; i < model_count; i++) {
        memcpy(_X + i * sample_size * dimension, _X, sample_size * dimension * sizeof(float));
        memcpy(_y + i * sample_size, _y, sample_size * sizeof(float));
    }
    float* const _w = new float[dimension * model_count];

    // Start timing
    clock_t clk = clock();

    for (int model_id = 0; model_id < model_count; model_id++) {
        float* const X = _X + model_id * sample_size * dimension;
        float* const y = _y + model_id * sample_size;
        float* const w = _w + model_id * dimension;

        // Initialize weights
        for (int i = 0; i < dimension; i++) {
            w[i] = 1;
        }

        int indices[batch_size];
        int indices_copy[batch_size];
        float residuals[batch_size];
        float residuals_copy[batch_size];
        float gradient[dimension];
        float prev_loss = 0;
        for (int _ = -1; _ < max_iter; _++) {
            // Sample with replacement
            for (int i = 0; i < batch_size; i++) {
                indices[i] = ((float)rand()) / RAND_MAX * sample_size;
            }

            for (int i = 0; i < batch_size; i++) {
                float* const x = X + indices[i] * dimension;
                float residual = -y[indices[i]];
                for (int j = 0; j < dimension; j++) {
                    residual += x[j] * w[j];
                }
                residuals[i] = residual;
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
                for (int i = 0; i < batch_size; i += stride * 2) {
                    int j = i;
                    const int j_end = i + stride;
                    if (j_end >= batch_size) {
                        break;
                    }
                    int k = j_end;
                    const int k_end = ((i + stride * 2) > batch_size) ? batch_size : (i + stride * 2);
                    int l = i;
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
                        for (j = i; j < l; j++) {
                            residuals[j] = residuals_copy[j];
                            indices[j] = indices_copy[j];
                        }
                    }
                    else {
                        for (k = j_end - 1;k >= j; k--) {
                            residuals[k + k_end - j_end] = residuals[k];
                            indices[k + k_end - j_end] = indices[k];
                        }
                        for (k = i; k < l; k++) {
                            residuals[k] = residuals_copy[k];
                            indices[k] = indices_copy[k];
                        }
                    }
                }
            }

            // Epsilon-trimming
            int index_low = 0;
            float abs_residual_low = std::abs(residuals[0]);
            int index_high = batch_size - 1;
            float abs_residual_high = std::abs(residuals[batch_size - 1]);
            for (int i = 0; i < (int)(batch_size * epsilon); i++) {
                if (abs_residual_low < abs_residual_high) {
                    index_high--;
                    abs_residual_high = std::abs(residuals[index_high]);
                }
                else {
                    index_low++;
                    abs_residual_low = std::abs(residuals[index_low]);
                }
            }

            // Z-score trimming
            while (true) {
                float mean = 0;
                for (int i = index_low; i <= index_high; i++) {
                    mean += residuals[i];
                }
                mean /= (index_high - index_low);
                float stdev = 0;
                for (int i = index_low; i <= index_high; i++) {
                    const float diff = residuals[i] - mean;
                    stdev += diff * diff;
                }
                stdev = sqrt(stdev / (index_high - index_low));
                bool flag_converged = true;
                const float threashold_low = mean - stdev * z_score_trimming_threashold;
                while (residuals[index_low] < threashold_low) {
                    index_low++;
                    flag_converged = false;
                }
                const float threashold_high = mean + stdev * z_score_trimming_threashold;
                while (residuals[index_high] > threashold_high) {
                    index_high--;
                    flag_converged = false;
                }
                if (flag_converged) {
                    break;
                }
            }

            // Huber loss
            float loss = 0;
            for (int i = 0; i < dimension; i++) {
                gradient[i] = 0;
            }
            for (int i = index_low; i <= index_high; i++) {
                float* const x = X + indices[i] * dimension;
                const float residual = residuals[i];
                const float abs_residual = std::abs(residual);
                if (abs_residual <= huber_loss_threashold) {
                    loss += residual * residual / 2;
                    for (int j = 0; j < dimension; j++) {
                        gradient[j] += residual * x[j];
                    }
                }
                else {
                    loss += abs_residual * huber_loss_threashold - huber_loss_threashold * huber_loss_threashold / 2;
                    for (int j = 0; j < dimension; j++) {
                        gradient[j] += ((residual > 0) - (residual < 0)) * x[j] * huber_loss_threashold;
                    }
                }
            }
            for (int i = 0; i < dimension; i++) {
                gradient[i] /= batch_size;
            }
            loss /= batch_size;

            // Update weights
            for (int i = 0; i < dimension; i++) {
                w[i] -= learning_rate * gradient[i];
            }

            // Check convergence
            if (std::abs((loss - prev_loss) / prev_loss) < 1e-4) {
                //break;
            }
            prev_loss = loss;
        }
    }

    clk = clock() - clk;
    printf("C++ running time:\t%.3fms\n", (double)clk / CLOCKS_PER_SEC * 1000);

    // Write the trained weights
    f = fopen("out.txt", "w");
    fprintf(f, "%f", _w[(model_count - 1) * dimension]);
    for (int i = 1; i < dimension; i++) {
        fprintf(f, " %f", _w[(model_count - 1) * dimension + i]);
    }
    fclose(f);

    // Free resources
    delete[] _X;
    delete[] _y;
    delete[] _w;

    return 0;
}
