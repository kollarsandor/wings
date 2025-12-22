export uniform int VECTOR_WIDTH = TARGET_WIDTH;
export uniform int CACHE_LINE_SIZE = 64;
export uniform int PREFETCH_DISTANCE = 16;

struct TensorDescriptor {
    uniform float * uniform data;
    uniform int dims[8];
    uniform int strides[8];
    uniform int ndim;
    uniform int size;
    uniform int alignment;
};

struct MatrixDescriptor {
    uniform float * uniform data;
    uniform int rows;
    uniform int cols;
    uniform int row_stride;
    uniform int col_stride;
    uniform int alignment;
};

struct ConvolutionParams {
    uniform int batch_size;
    uniform int in_channels;
    uniform int out_channels;
    uniform int in_height;
    uniform int in_width;
    uniform int out_height;
    uniform int out_width;
    uniform int kernel_height;
    uniform int kernel_width;
    uniform int stride_h;
    uniform int stride_w;
    uniform int pad_h;
    uniform int pad_w;
    uniform int dilation_h;
    uniform int dilation_w;
    uniform int groups;
};

struct PoolingParams {
    uniform int batch_size;
    uniform int channels;
    uniform int in_height;
    uniform int in_width;
    uniform int out_height;
    uniform int out_width;
    uniform int kernel_height;
    uniform int kernel_width;
    uniform int stride_h;
    uniform int stride_w;
    uniform int pad_h;
    uniform int pad_w;
    uniform int pool_type;
};

struct BatchNormParams {
    uniform int batch_size;
    uniform int channels;
    uniform int spatial_size;
    uniform float epsilon;
    uniform float momentum;
    uniform int training;
};

struct AttentionParams {
    uniform int batch_size;
    uniform int num_heads;
    uniform int seq_length;
    uniform int head_dim;
    uniform float scale;
    uniform int causal;
};

inline float fast_exp(float x) {
    float t = 1.0f + x / 256.0f;
    t = t * t; t = t * t; t = t * t; t = t * t;
    t = t * t; t = t * t; t = t * t; t = t * t;
    return t;
}

inline float fast_tanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

inline float fast_gelu(float x) {
    float x3 = x * x * x;
    float inner = 0.797884560803f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + fast_tanh(inner));
}

inline float fast_silu(float x) {
    return x * fast_sigmoid(x);
}

inline float fast_rsqrt(float x) {
    float xhalf = 0.5f * x;
    int i = floatbits(x);
    i = 0x5f375a86 - (i >> 1);
    float y = intbits(i);
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    return y;
}

export void vector_add(
    uniform float output[],
    uniform float input_a[],
    uniform float input_b[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = input_a[i] + input_b[i];
    }
}

export void vector_subtract(
    uniform float output[],
    uniform float input_a[],
    uniform float input_b[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = input_a[i] - input_b[i];
    }
}

export void vector_multiply(
    uniform float output[],
    uniform float input_a[],
    uniform float input_b[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = input_a[i] * input_b[i];
    }
}

export void vector_divide(
    uniform float output[],
    uniform float input_a[],
    uniform float input_b[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        float b = input_b[i];
        if (abs(b) > 1e-10f) {
            output[i] = input_a[i] / b;
        } else {
            output[i] = 0.0f;
        }
    }
}

export void vector_scale(
    uniform float output[],
    uniform float input[],
    uniform float scale,
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = input[i] * scale;
    }
}

export void vector_add_scalar(
    uniform float output[],
    uniform float input[],
    uniform float scalar,
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = input[i] + scalar;
    }
}

export void vector_fma(
    uniform float output[],
    uniform float input_a[],
    uniform float input_b[],
    uniform float input_c[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = input_a[i] * input_b[i] + input_c[i];
    }
}

export void vector_relu(
    uniform float output[],
    uniform float input[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = max(0.0f, input[i]);
    }
}

export void vector_leaky_relu(
    uniform float output[],
    uniform float input[],
    uniform float alpha,
    uniform int size
) {
    foreach (i = 0 ... size) {
        float x = input[i];
        output[i] = x > 0.0f ? x : alpha * x;
    }
}

export void vector_elu(
    uniform float output[],
    uniform float input[],
    uniform float alpha,
    uniform int size
) {
    foreach (i = 0 ... size) {
        float x = input[i];
        output[i] = x > 0.0f ? x : alpha * (fast_exp(x) - 1.0f);
    }
}

export void vector_sigmoid(
    uniform float output[],
    uniform float input[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = fast_sigmoid(input[i]);
    }
}

export void vector_tanh(
    uniform float output[],
    uniform float input[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = fast_tanh(input[i]);
    }
}

export void vector_gelu(
    uniform float output[],
    uniform float input[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = fast_gelu(input[i]);
    }
}

export void vector_silu(
    uniform float output[],
    uniform float input[],
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = fast_silu(input[i]);
    }
}

export void vector_softplus(
    uniform float output[],
    uniform float input[],
    uniform float beta,
    uniform float threshold,
    uniform int size
) {
    foreach (i = 0 ... size) {
        float x = input[i];
        float bx = beta * x;
        if (bx > threshold) {
            output[i] = x;
        } else {
            output[i] = log(1.0f + fast_exp(bx)) / beta;
        }
    }
}

export void matmul_blocked(
    uniform float C[],
    uniform float A[],
    uniform float B[],
    uniform int M,
    uniform int N,
    uniform int K,
    uniform int lda,
    uniform int ldb,
    uniform int ldc
) {
    uniform int BLOCK_M = 64;
    uniform int BLOCK_N = 64;
    uniform int BLOCK_K = 64;

    for (uniform int i0 = 0; i0 < M; i0 += BLOCK_M) {
        for (uniform int j0 = 0; j0 < N; j0 += BLOCK_N) {
            uniform int i_end = min(i0 + BLOCK_M, M);
            uniform int j_end = min(j0 + BLOCK_N, N);

            for (uniform int k0 = 0; k0 < K; k0 += BLOCK_K) {
                uniform int k_end = min(k0 + BLOCK_K, K);

                for (uniform int i = i0; i < i_end; i++) {
                    foreach (j = j0 ... j_end) {
                        float sum = 0.0f;
                        for (uniform int k = k0; k < k_end; k++) {
                            sum += A[i * lda + k] * B[k * ldb + j];
                        }
                        if (k0 == 0) {
                            C[i * ldc + j] = sum;
                        } else {
                            C[i * ldc + j] += sum;
                        }
                    }
                }
            }
        }
    }
}

export void matmul_transposed_b(
    uniform float C[],
    uniform float A[],
    uniform float B[],
    uniform int M,
    uniform int N,
    uniform int K,
    uniform int lda,
    uniform int ldb,
    uniform int ldc
) {
    uniform int BLOCK = 32;

    for (uniform int i0 = 0; i0 < M; i0 += BLOCK) {
        for (uniform int j0 = 0; j0 < N; j0 += BLOCK) {
            uniform int i_end = min(i0 + BLOCK, M);
            uniform int j_end = min(j0 + BLOCK, N);

            for (uniform int i = i0; i < i_end; i++) {
                foreach (j = j0 ... j_end) {
                    float sum = 0.0f;
                    for (uniform int k = 0; k < K; k++) {
                        sum += A[i * lda + k] * B[j * ldb + k];
                    }
                    C[i * ldc + j] = sum;
                }
            }
        }
    }
}

export void batch_matmul(
    uniform float C[],
    uniform float A[],
    uniform float B[],
    uniform int batch_size,
    uniform int M,
    uniform int N,
    uniform int K
) {
    uniform int A_batch_stride = M * K;
    uniform int B_batch_stride = K * N;
    uniform int C_batch_stride = M * N;

    for (uniform int b = 0; b < batch_size; b++) {
        uniform float * uniform A_ptr = A + b * A_batch_stride;
        uniform float * uniform B_ptr = B + b * B_batch_stride;
        uniform float * uniform C_ptr = C + b * C_batch_stride;

        for (uniform int i = 0; i < M; i++) {
            foreach (j = 0 ... N) {
                float sum = 0.0f;
                for (uniform int k = 0; k < K; k++) {
                    sum += A_ptr[i * K + k] * B_ptr[k * N + j];
                }
                C_ptr[i * N + j] = sum;
            }
        }
    }
}

export void softmax_forward(
    uniform float output[],
    uniform float input[],
    uniform int batch_size,
    uniform int dim
) {
    for (uniform int b = 0; b < batch_size; b++) {
        uniform int offset = b * dim;

        float max_val = input[offset];
        for (uniform int i = 1; i < dim; i++) {
            max_val = max(max_val, input[offset + i]);
        }
        uniform float max_reduced = reduce_max(max_val);

        float sum_exp = 0.0f;
        foreach (i = 0 ... dim) {
            float exp_val = fast_exp(input[offset + i] - max_reduced);
            output[offset + i] = exp_val;
            sum_exp += exp_val;
        }
        uniform float sum_reduced = reduce_add(sum_exp);

        float inv_sum = 1.0f / sum_reduced;
        foreach (i = 0 ... dim) {
            output[offset + i] = output[offset + i] * inv_sum;
        }
    }
}

export void layer_norm_forward(
    uniform float output[],
    uniform float input[],
    uniform float gamma[],
    uniform float beta[],
    uniform float mean_out[],
    uniform float var_out[],
    uniform int batch_size,
    uniform int hidden_size,
    uniform float epsilon
) {
    for (uniform int b = 0; b < batch_size; b++) {
        uniform int offset = b * hidden_size;

        float sum = 0.0f;
        foreach (i = 0 ... hidden_size) {
            sum += input[offset + i];
        }
        uniform float mean = reduce_add(sum) / hidden_size;
        mean_out[b] = mean;

        float var_sum = 0.0f;
        foreach (i = 0 ... hidden_size) {
            float diff = input[offset + i] - mean;
            var_sum += diff * diff;
        }
        uniform float variance = reduce_add(var_sum) / hidden_size;
        var_out[b] = variance;

        uniform float inv_std = fast_rsqrt(variance + epsilon);

        foreach (i = 0 ... hidden_size) {
            float normalized = (input[offset + i] - mean) * inv_std;
            output[offset + i] = normalized * gamma[i] + beta[i];
        }
    }
}

export void rms_norm_forward(
    uniform float output[],
    uniform float input[],
    uniform float weight[],
    uniform int batch_size,
    uniform int hidden_size,
    uniform float epsilon
) {
    for (uniform int b = 0; b < batch_size; b++) {
        uniform int offset = b * hidden_size;

        float sum_sq = 0.0f;
        foreach (i = 0 ... hidden_size) {
            float x = input[offset + i];
            sum_sq += x * x;
        }
        uniform float mean_sq = reduce_add(sum_sq) / hidden_size;
        uniform float rms = fast_rsqrt(mean_sq + epsilon);

        foreach (i = 0 ... hidden_size) {
            output[offset + i] = input[offset + i] * rms * weight[i];
        }
    }
}

export void attention_scores(
    uniform float scores[],
    uniform float queries[],
    uniform float keys[],
    uniform int batch_size,
    uniform int num_heads,
    uniform int seq_length,
    uniform int head_dim,
    uniform float scale
) {
    uniform int head_stride = seq_length * head_dim;
    uniform int batch_stride = num_heads * head_stride;
    uniform int score_head_stride = seq_length * seq_length;
    uniform int score_batch_stride = num_heads * score_head_stride;

    for (uniform int b = 0; b < batch_size; b++) {
        for (uniform int h = 0; h < num_heads; h++) {
            uniform float * uniform Q = queries + b * batch_stride + h * head_stride;
            uniform float * uniform K = keys + b * batch_stride + h * head_stride;
            uniform float * uniform S = scores + b * score_batch_stride + h * score_head_stride;

            for (uniform int i = 0; i < seq_length; i++) {
                foreach (j = 0 ... seq_length) {
                    float dot = 0.0f;
                    for (uniform int d = 0; d < head_dim; d++) {
                        dot += Q[i * head_dim + d] * K[j * head_dim + d];
                    }
                    S[i * seq_length + j] = dot * scale;
                }
            }
        }
    }
}

export void attention_output(
    uniform float output[],
    uniform float attention_weights[],
    uniform float values[],
    uniform int batch_size,
    uniform int num_heads,
    uniform int seq_length,
    uniform int head_dim
) {
    uniform int head_stride = seq_length * head_dim;
    uniform int batch_stride = num_heads * head_stride;
    uniform int attn_head_stride = seq_length * seq_length;
    uniform int attn_batch_stride = num_heads * attn_head_stride;

    for (uniform int b = 0; b < batch_size; b++) {
        for (uniform int h = 0; h < num_heads; h++) {
            uniform float * uniform A = attention_weights + b * attn_batch_stride + h * attn_head_stride;
            uniform float * uniform V = values + b * batch_stride + h * head_stride;
            uniform float * uniform O = output + b * batch_stride + h * head_stride;

            for (uniform int i = 0; i < seq_length; i++) {
                foreach (d = 0 ... head_dim) {
                    float sum = 0.0f;
                    for (uniform int j = 0; j < seq_length; j++) {
                        sum += A[i * seq_length + j] * V[j * head_dim + d];
                    }
                    O[i * head_dim + d] = sum;
                }
            }
        }
    }
}

export void causal_mask(
    uniform float scores[],
    uniform int batch_size,
    uniform int num_heads,
    uniform int seq_length
) {
    uniform int head_stride = seq_length * seq_length;
    uniform int batch_stride = num_heads * head_stride;
    uniform float neg_inf = -1e9f;

    for (uniform int b = 0; b < batch_size; b++) {
        for (uniform int h = 0; h < num_heads; h++) {
            uniform float * uniform S = scores + b * batch_stride + h * head_stride;

            for (uniform int i = 0; i < seq_length; i++) {
                foreach (j = 0 ... seq_length) {
                    if (j > i) {
                        S[i * seq_length + j] = neg_inf;
                    }
                }
            }
        }
    }
}

export void conv2d_forward(
    uniform float output[],
    uniform float input[],
    uniform float weights[],
    uniform float bias[],
    uniform ConvolutionParams params
) {
    uniform int B = params.batch_size;
    uniform int IC = params.in_channels;
    uniform int OC = params.out_channels;
    uniform int IH = params.in_height;
    uniform int IW = params.in_width;
    uniform int OH = params.out_height;
    uniform int OW = params.out_width;
    uniform int KH = params.kernel_height;
    uniform int KW = params.kernel_width;
    uniform int SH = params.stride_h;
    uniform int SW = params.stride_w;
    uniform int PH = params.pad_h;
    uniform int PW = params.pad_w;
    uniform int DH = params.dilation_h;
    uniform int DW = params.dilation_w;
    uniform int G = params.groups;

    uniform int IC_per_group = IC / G;
    uniform int OC_per_group = OC / G;

    for (uniform int b = 0; b < B; b++) {
        for (uniform int g = 0; g < G; g++) {
            for (uniform int oc = 0; oc < OC_per_group; oc++) {
                uniform int oc_abs = g * OC_per_group + oc;

                for (uniform int oh = 0; oh < OH; oh++) {
                    foreach (ow = 0 ... OW) {
                        float sum = bias != NULL ? bias[oc_abs] : 0.0f;

                        for (uniform int ic = 0; ic < IC_per_group; ic++) {
                            uniform int ic_abs = g * IC_per_group + ic;

                            for (uniform int kh = 0; kh < KH; kh++) {
                                for (uniform int kw = 0; kw < KW; kw++) {
                                    int ih = oh * SH - PH + kh * DH;
                                    int iw = ow * SW - PW + kw * DW;

                                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                        int input_idx = ((b * IC + ic_abs) * IH + ih) * IW + iw;
                                        int weight_idx = ((oc_abs * IC_per_group + ic) * KH + kh) * KW + kw;
                                        sum += input[input_idx] * weights[weight_idx];
                                    }
                                }
                            }
                        }

                        int output_idx = ((b * OC + oc_abs) * OH + oh) * OW + ow;
                        output[output_idx] = sum;
                    }
                }
            }
        }
    }
}

export void max_pool2d_forward(
    uniform float output[],
    uniform int indices[],
    uniform float input[],
    uniform PoolingParams params
) {
    uniform int B = params.batch_size;
    uniform int C = params.channels;
    uniform int IH = params.in_height;
    uniform int IW = params.in_width;
    uniform int OH = params.out_height;
    uniform int OW = params.out_width;
    uniform int KH = params.kernel_height;
    uniform int KW = params.kernel_width;
    uniform int SH = params.stride_h;
    uniform int SW = params.stride_w;
    uniform int PH = params.pad_h;
    uniform int PW = params.pad_w;

    for (uniform int b = 0; b < B; b++) {
        for (uniform int c = 0; c < C; c++) {
            for (uniform int oh = 0; oh < OH; oh++) {
                foreach (ow = 0 ... OW) {
                    float max_val = -1e38f;
                    int max_idx = -1;

                    for (uniform int kh = 0; kh < KH; kh++) {
                        for (uniform int kw = 0; kw < KW; kw++) {
                            int ih = oh * SH - PH + kh;
                            int iw = ow * SW - PW + kw;

                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                int input_idx = ((b * C + c) * IH + ih) * IW + iw;
                                float val = input[input_idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = input_idx;
                                }
                            }
                        }
                    }

                    int output_idx = ((b * C + c) * OH + oh) * OW + ow;
                    output[output_idx] = max_val;
                    if (indices != NULL) {
                        indices[output_idx] = max_idx;
                    }
                }
            }
        }
    }
}

export void avg_pool2d_forward(
    uniform float output[],
    uniform float input[],
    uniform PoolingParams params
) {
    uniform int B = params.batch_size;
    uniform int C = params.channels;
    uniform int IH = params.in_height;
    uniform int IW = params.in_width;
    uniform int OH = params.out_height;
    uniform int OW = params.out_width;
    uniform int KH = params.kernel_height;
    uniform int KW = params.kernel_width;
    uniform int SH = params.stride_h;
    uniform int SW = params.stride_w;
    uniform int PH = params.pad_h;
    uniform int PW = params.pad_w;

    for (uniform int b = 0; b < B; b++) {
        for (uniform int c = 0; c < C; c++) {
            for (uniform int oh = 0; oh < OH; oh++) {
                foreach (ow = 0 ... OW) {
                    float sum = 0.0f;
                    int count = 0;

                    for (uniform int kh = 0; kh < KH; kh++) {
                        for (uniform int kw = 0; kw < KW; kw++) {
                            int ih = oh * SH - PH + kh;
                            int iw = ow * SW - PW + kw;

                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                int input_idx = ((b * C + c) * IH + ih) * IW + iw;
                                sum += input[input_idx];
                                count += 1;
                            }
                        }
                    }

                    int output_idx = ((b * C + c) * OH + oh) * OW + ow;
                    output[output_idx] = count > 0 ? sum / count : 0.0f;
                }
            }
        }
    }
}

export void batch_norm_forward(
    uniform float output[],
    uniform float input[],
    uniform float gamma[],
    uniform float beta[],
    uniform float running_mean[],
    uniform float running_var[],
    uniform float batch_mean[],
    uniform float batch_var[],
    uniform BatchNormParams params
) {
    uniform int B = params.batch_size;
    uniform int C = params.channels;
    uniform int S = params.spatial_size;
    uniform float eps = params.epsilon;
    uniform float momentum = params.momentum;
    uniform int training = params.training;

    if (training) {
        for (uniform int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (uniform int b = 0; b < B; b++) {
                foreach (s = 0 ... S) {
                    sum += input[(b * C + c) * S + s];
                }
            }
            uniform float mean = reduce_add(sum) / (B * S);
            batch_mean[c] = mean;

            float var_sum = 0.0f;
            for (uniform int b = 0; b < B; b++) {
                foreach (s = 0 ... S) {
                    float diff = input[(b * C + c) * S + s] - mean;
                    var_sum += diff * diff;
                }
            }
            uniform float var = reduce_add(var_sum) / (B * S);
            batch_var[c] = var;

            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;

            uniform float inv_std = fast_rsqrt(var + eps);

            for (uniform int b = 0; b < B; b++) {
                foreach (s = 0 ... S) {
                    int idx = (b * C + c) * S + s;
                    float normalized = (input[idx] - mean) * inv_std;
                    output[idx] = normalized * gamma[c] + beta[c];
                }
            }
        }
    } else {
        for (uniform int c = 0; c < C; c++) {
            uniform float mean = running_mean[c];
            uniform float var = running_var[c];
            uniform float inv_std = fast_rsqrt(var + eps);

            for (uniform int b = 0; b < B; b++) {
                foreach (s = 0 ... S) {
                    int idx = (b * C + c) * S + s;
                    float normalized = (input[idx] - mean) * inv_std;
                    output[idx] = normalized * gamma[c] + beta[c];
                }
            }
        }
    }
}

export void embedding_forward(
    uniform float output[],
    uniform int indices[],
    uniform float embedding_table[],
    uniform int batch_size,
    uniform int seq_length,
    uniform int embedding_dim
) {
    for (uniform int b = 0; b < batch_size; b++) {
        for (uniform int s = 0; s < seq_length; s++) {
            uniform int idx = indices[b * seq_length + s];
            uniform float * uniform emb = embedding_table + idx * embedding_dim;
            uniform float * uniform out = output + (b * seq_length + s) * embedding_dim;

            foreach (d = 0 ... embedding_dim) {
                out[d] = emb[d];
            }
        }
    }
}

export void rotary_embedding(
    uniform float output[],
    uniform float input[],
    uniform float cos_cache[],
    uniform float sin_cache[],
    uniform int batch_size,
    uniform int num_heads,
    uniform int seq_length,
    uniform int head_dim
) {
    uniform int half_dim = head_dim / 2;
    uniform int head_stride = seq_length * head_dim;
    uniform int batch_stride = num_heads * head_stride;

    for (uniform int b = 0; b < batch_size; b++) {
        for (uniform int h = 0; h < num_heads; h++) {
            for (uniform int s = 0; s < seq_length; s++) {
                uniform float * uniform x = input + b * batch_stride + h * head_stride + s * head_dim;
                uniform float * uniform y = output + b * batch_stride + h * head_stride + s * head_dim;
                uniform float * uniform cos_ptr = cos_cache + s * half_dim;
                uniform float * uniform sin_ptr = sin_cache + s * half_dim;

                foreach (d = 0 ... half_dim) {
                    float x1 = x[d];
                    float x2 = x[d + half_dim];
                    float cos_val = cos_ptr[d];
                    float sin_val = sin_ptr[d];

                    y[d] = x1 * cos_val - x2 * sin_val;
                    y[d + half_dim] = x1 * sin_val + x2 * cos_val;
                }
            }
        }
    }
}

export void dropout_forward(
    uniform float output[],
    uniform float input[],
    uniform float mask[],
    uniform float scale,
    uniform int size
) {
    foreach (i = 0 ... size) {
        output[i] = input[i] * mask[i] * scale;
    }
}

export void cross_entropy_loss(
    uniform float loss[],
    uniform float grad[],
    uniform float logits[],
    uniform int targets[],
    uniform int batch_size,
    uniform int num_classes
) {
    for (uniform int b = 0; b < batch_size; b++) {
        uniform float * uniform logit = logits + b * num_classes;
        uniform int target = targets[b];

        float max_logit = logit[0];
        for (uniform int c = 1; c < num_classes; c++) {
            max_logit = max(max_logit, logit[c]);
        }
        uniform float max_val = reduce_max(max_logit);

        float sum_exp = 0.0f;
        foreach (c = 0 ... num_classes) {
            sum_exp += fast_exp(logit[c] - max_val);
        }
        uniform float log_sum_exp = log(reduce_add(sum_exp)) + max_val;

        loss[b] = log_sum_exp - logit[target];

        if (grad != NULL) {
            uniform float * uniform g = grad + b * num_classes;
            foreach (c = 0 ... num_classes) {
                float prob = fast_exp(logit[c] - log_sum_exp);
                g[c] = prob;
            }
            g[target] -= 1.0f;
        }
    }
}

export void adam_update(
    uniform float params[],
    uniform float grads[],
    uniform float m[],
    uniform float v[],
    uniform float lr,
    uniform float beta1,
    uniform float beta2,
    uniform float epsilon,
    uniform float weight_decay,
    uniform int t,
    uniform int size
) {
    uniform float beta1_t = 1.0f;
    uniform float beta2_t = 1.0f;
    for (uniform int i = 0; i < t; i++) {
        beta1_t *= beta1;
        beta2_t *= beta2;
    }
    uniform float bias_correction1 = 1.0f - beta1_t;
    uniform float bias_correction2 = 1.0f - beta2_t;

    foreach (i = 0 ... size) {
        float g = grads[i];
        if (weight_decay != 0.0f) {
            g += weight_decay * params[i];
        }

        float m_i = beta1 * m[i] + (1.0f - beta1) * g;
        float v_i = beta2 * v[i] + (1.0f - beta2) * g * g;

        m[i] = m_i;
        v[i] = v_i;

        float m_hat = m_i / bias_correction1;
        float v_hat = v_i / bias_correction2;

        params[i] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

export void sgd_momentum_update(
    uniform float params[],
    uniform float grads[],
    uniform float velocity[],
    uniform float lr,
    uniform float momentum,
    uniform float weight_decay,
    uniform float dampening,
    uniform int nesterov,
    uniform int size
) {
    foreach (i = 0 ... size) {
        float g = grads[i];
        if (weight_decay != 0.0f) {
            g += weight_decay * params[i];
        }

        float v_i = momentum * velocity[i] + (1.0f - dampening) * g;
        velocity[i] = v_i;

        float update;
        if (nesterov) {
            update = g + momentum * v_i;
        } else {
            update = v_i;
        }

        params[i] -= lr * update;
    }
}

export uniform float reduce_sum(uniform float input[], uniform int size) {
    float sum = 0.0f;
    foreach (i = 0 ... size) {
        sum += input[i];
    }
    return reduce_add(sum);
}

export uniform float reduce_max_val(uniform float input[], uniform int size) {
    float max_val = input[0];
    foreach (i = 1 ... size) {
        max_val = max(max_val, input[i]);
    }
    return reduce_max(max_val);
}

export uniform float reduce_min_val(uniform float input[], uniform int size) {
    float min_val = input[0];
    foreach (i = 1 ... size) {
        min_val = min(min_val, input[i]);
    }
    return reduce_min(min_val);
}

export void transpose_2d(
    uniform float output[],
    uniform float input[],
    uniform int rows,
    uniform int cols
) {
    uniform int BLOCK = 16;

    for (uniform int i0 = 0; i0 < rows; i0 += BLOCK) {
        for (uniform int j0 = 0; j0 < cols; j0 += BLOCK) {
            uniform int i_end = min(i0 + BLOCK, rows);
            uniform int j_end = min(j0 + BLOCK, cols);

            for (uniform int i = i0; i < i_end; i++) {
                foreach (j = j0 ... j_end) {
                    output[j * rows + i] = input[i * cols + j];
                }
            }
        }
    }
}

export void concat_tensors(
    uniform float output[],
    uniform float * uniform inputs[],
    uniform int input_sizes[],
    uniform int num_inputs,
    uniform int batch_size,
    uniform int axis_dim[]
) {
    uniform int total_offset = 0;

    for (uniform int n = 0; n < num_inputs; n++) {
        uniform float * uniform src = inputs[n];
        uniform int src_size = input_sizes[n];

        foreach (i = 0 ... src_size) {
            output[total_offset + i] = src[i];
        }

        total_offset += src_size;
    }
}

export void split_tensor(
    uniform float * uniform outputs[],
    uniform float input[],
    uniform int output_sizes[],
    uniform int num_outputs
) {
    uniform int offset = 0;

    for (uniform int n = 0; n < num_outputs; n++) {
        uniform float * uniform dst = outputs[n];
        uniform int size = output_sizes[n];

        foreach (i = 0 ... size) {
            dst[i] = input[offset + i];
        }

        offset += size;
    }
}

export void broadcast_add(
    uniform float output[],
    uniform float input[],
    uniform float bias[],
    uniform int batch_size,
    uniform int channels,
    uniform int spatial_size
) {
    for (uniform int b = 0; b < batch_size; b++) {
        for (uniform int c = 0; c < channels; c++) {
            uniform int offset = (b * channels + c) * spatial_size;
            uniform float b_val = bias[c];

            foreach (s = 0 ... spatial_size) {
                output[offset + s] = input[offset + s] + b_val;
            }
        }
    }
}

export void broadcast_multiply(
    uniform float output[],
    uniform float input[],
    uniform float scale[],
    uniform int batch_size,
    uniform int channels,
    uniform int spatial_size
) {
    for (uniform int b = 0; b < batch_size; b++) {
        for (uniform int c = 0; c < channels; c++) {
            uniform int offset = (b * channels + c) * spatial_size;
            uniform float s_val = scale[c];

            foreach (s = 0 ... spatial_size) {
                output[offset + s] = input[offset + s] * s_val;
            }
        }
    }
}

export void clip_gradient(
    uniform float grads[],
    uniform float max_norm,
    uniform int size
) {
    float sum_sq = 0.0f;
    foreach (i = 0 ... size) {
        float g = grads[i];
        sum_sq += g * g;
    }
    uniform float total_norm = sqrt(reduce_add(sum_sq));

    if (total_norm > max_norm) {
        uniform float scale = max_norm / total_norm;
        foreach (i = 0 ... size) {
            grads[i] *= scale;
        }
    }
}
