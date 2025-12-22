const std = @import("std");
const math = std.math;
const builtin = @import("builtin");

pub const InferenceError = error{
    ModelNotLoaded,
    InvalidInput,
    InvalidShape,
    OutOfMemory,
    ComputationError,
    UnsupportedOperation,
    InvalidModelFormat,
    VersionMismatch,
    QuantizationError,
    ThreadPoolError,
    CacheError,
    ProfilerError,
};

pub const DataType = enum(u8) {
    float32,
    float16,
    bfloat16,
    int32,
    int16,
    int8,
    uint8,
    bool_type,

    pub fn size(self: DataType) usize {
        return switch (self) {
            .float32, .int32 => 4,
            .float16, .bfloat16, .int16 => 2,
            .int8, .uint8, .bool_type => 1,
        };
    }
};

pub const TensorShape = struct {
    dims: [8]usize,
    ndim: usize,

    pub fn init(dims: []const usize) TensorShape {
        var shape = TensorShape{
            .dims = [_]usize{0} ** 8,
            .ndim = dims.len,
        };
        for (dims, 0..) |d, i| {
            shape.dims[i] = d;
        }
        return shape;
    }

    pub fn size(self: TensorShape) usize {
        var total: usize = 1;
        for (0..self.ndim) |i| {
            total *= self.dims[i];
        }
        return total;
    }

    pub fn equals(self: TensorShape, other: TensorShape) bool {
        if (self.ndim != other.ndim) return false;
        for (0..self.ndim) |i| {
            if (self.dims[i] != other.dims[i]) return false;
        }
        return true;
    }
};

pub const Tensor = struct {
    data: []f32,
    shape: TensorShape,
    dtype: DataType,
    allocator: std.mem.Allocator,
    is_contiguous: bool,
    strides: [8]usize,

    pub fn init(allocator: std.mem.Allocator, shape: TensorShape) !Tensor {
        const size = shape.size();
        const data = try allocator.alloc(f32, size);
        @memset(data, 0.0);

        var strides: [8]usize = [_]usize{0} ** 8;
        var stride: usize = 1;
        var i = shape.ndim;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= shape.dims[i];
        }

        return Tensor{
            .data = data,
            .shape = shape,
            .dtype = .float32,
            .allocator = allocator,
            .is_contiguous = true,
            .strides = strides,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
    }

    pub fn clone(self: *const Tensor) !Tensor {
        const data = try self.allocator.alloc(f32, self.data.len);
        @memcpy(data, self.data);
        return Tensor{
            .data = data,
            .shape = self.shape,
            .dtype = self.dtype,
            .allocator = self.allocator,
            .is_contiguous = self.is_contiguous,
            .strides = self.strides,
        };
    }

    pub fn fill(self: *Tensor, value: f32) void {
        @memset(self.data, value);
    }

    pub fn get(self: *const Tensor, indices: []const usize) f32 {
        var offset: usize = 0;
        for (indices, 0..) |idx, i| {
            offset += idx * self.strides[i];
        }
        return self.data[offset];
    }

    pub fn set(self: *Tensor, indices: []const usize, value: f32) void {
        var offset: usize = 0;
        for (indices, 0..) |idx, i| {
            offset += idx * self.strides[i];
        }
        self.data[offset] = value;
    }
};

pub const LayerType = enum(u8) {
    linear,
    conv2d,
    batch_norm,
    layer_norm,
    rms_norm,
    relu,
    gelu,
    silu,
    softmax,
    embedding,
    attention,
    dropout,
    residual,
};

pub const LayerConfig = struct {
    layer_type: LayerType,
    in_features: usize,
    out_features: usize,
    num_heads: usize,
    head_dim: usize,
    epsilon: f32,
    bias: bool,
    activation: LayerType,

    pub fn default() LayerConfig {
        return LayerConfig{
            .layer_type = .linear,
            .in_features = 0,
            .out_features = 0,
            .num_heads = 1,
            .head_dim = 64,
            .epsilon = 1e-5,
            .bias = true,
            .activation = .relu,
        };
    }
};

pub const InferenceLayer = struct {
    config: LayerConfig,
    weight: ?Tensor,
    bias: ?Tensor,
    gamma: ?Tensor,
    beta: ?Tensor,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: LayerConfig) !InferenceLayer {
        var layer = InferenceLayer{
            .config = config,
            .weight = null,
            .bias = null,
            .gamma = null,
            .beta = null,
            .allocator = allocator,
        };

        switch (config.layer_type) {
            .linear => {
                const weight_shape = TensorShape.init(&[_]usize{ config.out_features, config.in_features });
                layer.weight = try Tensor.init(allocator, weight_shape);

                if (config.bias) {
                    const bias_shape = TensorShape.init(&[_]usize{config.out_features});
                    layer.bias = try Tensor.init(allocator, bias_shape);
                }
            },
            .layer_norm, .rms_norm => {
                const gamma_shape = TensorShape.init(&[_]usize{config.in_features});
                layer.gamma = try Tensor.init(allocator, gamma_shape);
                layer.gamma.?.fill(1.0);

                const beta_shape = TensorShape.init(&[_]usize{config.in_features});
                layer.beta = try Tensor.init(allocator, beta_shape);
            },
            .embedding => {
                const weight_shape = TensorShape.init(&[_]usize{ config.in_features, config.out_features });
                layer.weight = try Tensor.init(allocator, weight_shape);
            },
            else => {},
        }

        return layer;
    }

    pub fn deinit(self: *InferenceLayer) void {
        if (self.weight) |*w| w.deinit();
        if (self.bias) |*b| b.deinit();
        if (self.gamma) |*g| g.deinit();
        if (self.beta) |*bt| bt.deinit();
    }

    pub fn forward(self: *const InferenceLayer, allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
        return switch (self.config.layer_type) {
            .linear => self.forwardLinear(allocator, input),
            .layer_norm => self.forwardLayerNorm(allocator, input),
            .rms_norm => self.forwardRMSNorm(allocator, input),
            .relu => self.forwardRelu(allocator, input),
            .gelu => self.forwardGelu(allocator, input),
            .silu => self.forwardSilu(allocator, input),
            .softmax => self.forwardSoftmax(allocator, input),
            .embedding => self.forwardEmbedding(allocator, input),
            else => try input.clone(),
        };
    }

    fn forwardLinear(self: *const InferenceLayer, allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
        const batch_size = input.shape.dims[0];
        const seq_len = if (input.shape.ndim > 2) input.shape.dims[1] else 1;
        const in_features = self.config.in_features;
        const out_features = self.config.out_features;

        const output_shape = if (input.shape.ndim > 2)
            TensorShape.init(&[_]usize{ batch_size, seq_len, out_features })
        else
            TensorShape.init(&[_]usize{ batch_size, out_features });

        var output = try Tensor.init(allocator, output_shape);
        const weight = self.weight.?.data;
        const bias_data = if (self.bias) |b| b.data else null;

        if (input.shape.ndim > 2) {
            for (0..batch_size) |b| {
                for (0..seq_len) |s| {
                    for (0..out_features) |o| {
                        var sum: f32 = 0.0;
                        for (0..in_features) |i| {
                            sum += input.data[(b * seq_len + s) * in_features + i] * weight[o * in_features + i];
                        }
                        if (bias_data) |bd| {
                            sum += bd[o];
                        }
                        output.data[(b * seq_len + s) * out_features + o] = sum;
                    }
                }
            }
        } else {
            for (0..batch_size) |b| {
                for (0..out_features) |o| {
                    var sum: f32 = 0.0;
                    for (0..in_features) |i| {
                        sum += input.data[b * in_features + i] * weight[o * in_features + i];
                    }
                    if (bias_data) |bd| {
                        sum += bd[o];
                    }
                    output.data[b * out_features + o] = sum;
                }
            }
        }

        return output;
    }

    fn forwardLayerNorm(self: *const InferenceLayer, allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
        const batch_size = input.shape.dims[0];
        const seq_len = if (input.shape.ndim > 2) input.shape.dims[1] else 1;
        const hidden_size = self.config.in_features;
        const epsilon = self.config.epsilon;

        var output = try Tensor.init(allocator, input.shape);
        const gamma = if (self.gamma) |g| g.data else null;
        const beta = if (self.beta) |b| b.data else null;

        const total_sequences = batch_size * seq_len;
        for (0..total_sequences) |idx| {
            var sum: f32 = 0.0;
            for (0..hidden_size) |h| {
                sum += input.data[idx * hidden_size + h];
            }
            const mean = sum / @as(f32, @floatFromInt(hidden_size));

            var var_sum: f32 = 0.0;
            for (0..hidden_size) |h| {
                const diff = input.data[idx * hidden_size + h] - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f32, @floatFromInt(hidden_size));
            const inv_std = 1.0 / @sqrt(variance + epsilon);

            for (0..hidden_size) |h| {
                var normalized = (input.data[idx * hidden_size + h] - mean) * inv_std;
                if (gamma) |g| {
                    normalized *= g[h];
                }
                if (beta) |b| {
                    normalized += b[h];
                }
                output.data[idx * hidden_size + h] = normalized;
            }
        }

        return output;
    }

    fn forwardRMSNorm(self: *const InferenceLayer, allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
        const batch_size = input.shape.dims[0];
        const seq_len = if (input.shape.ndim > 2) input.shape.dims[1] else 1;
        const hidden_size = self.config.in_features;
        const epsilon = self.config.epsilon;

        var output = try Tensor.init(allocator, input.shape);
        const gamma = if (self.gamma) |g| g.data else null;

        const total_sequences = batch_size * seq_len;
        for (0..total_sequences) |idx| {
            var sum_sq: f32 = 0.0;
            for (0..hidden_size) |h| {
                const x = input.data[idx * hidden_size + h];
                sum_sq += x * x;
            }
            const rms = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(hidden_size)) + epsilon);

            for (0..hidden_size) |h| {
                var result = input.data[idx * hidden_size + h] * rms;
                if (gamma) |g| {
                    result *= g[h];
                }
                output.data[idx * hidden_size + h] = result;
            }
        }

        return output;
    }

    fn forwardRelu(self: *const InferenceLayer, allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
        _ = self;
        var output = try Tensor.init(allocator, input.shape);
        for (input.data, 0..) |x, i| {
            output.data[i] = @max(0.0, x);
        }
        return output;
    }

    fn forwardGelu(self: *const InferenceLayer, allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
        _ = self;
        var output = try Tensor.init(allocator, input.shape);
        for (input.data, 0..) |x, i| {
            const x3 = x * x * x;
            const inner = 0.797884560803 * (x + 0.044715 * x3);
            output.data[i] = 0.5 * x * (1.0 + @tanh(inner));
        }
        return output;
    }

    fn forwardSilu(self: *const InferenceLayer, allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
        _ = self;
        var output = try Tensor.init(allocator, input.shape);
        for (input.data, 0..) |x, i| {
            output.data[i] = x / (1.0 + @exp(-x));
        }
        return output;
    }

    fn forwardSoftmax(self: *const InferenceLayer, allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
        _ = self;
        const batch_size = input.shape.dims[0];
        const num_classes = input.shape.dims[input.shape.ndim - 1];

        var output = try Tensor.init(allocator, input.shape);
        const num_sequences = input.shape.size() / num_classes;

        for (0..num_sequences) |seq| {
            var max_val = input.data[seq * num_classes];
            for (1..num_classes) |c| {
                if (input.data[seq * num_classes + c] > max_val) {
                    max_val = input.data[seq * num_classes + c];
                }
            }

            var sum_exp: f32 = 0.0;
            for (0..num_classes) |c| {
                output.data[seq * num_classes + c] = @exp(input.data[seq * num_classes + c] - max_val);
                sum_exp += output.data[seq * num_classes + c];
            }

            for (0..num_classes) |c| {
                output.data[seq * num_classes + c] /= sum_exp;
            }
        }

        return output;
    }

    fn forwardEmbedding(self: *const InferenceLayer, allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
        const batch_size = input.shape.dims[0];
        const seq_len = input.shape.dims[1];
        const embedding_dim = self.config.out_features;

        const output_shape = TensorShape.init(&[_]usize{ batch_size, seq_len, embedding_dim });
        var output = try Tensor.init(allocator, output_shape);
        const weight = self.weight.?.data;

        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                const idx: usize = @intFromFloat(input.data[b * seq_len + s]);
                for (0..embedding_dim) |d| {
                    output.data[(b * seq_len + s) * embedding_dim + d] = weight[idx * embedding_dim + d];
                }
            }
        }

        return output;
    }
};

pub const ModelConfig = struct {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    max_seq_len: usize,
    epsilon: f32,
    rope_theta: f32,
    use_cache: bool,

    pub fn default() ModelConfig {
        return ModelConfig{
            .vocab_size = 32000,
            .hidden_size = 4096,
            .num_layers = 32,
            .num_heads = 32,
            .head_dim = 128,
            .intermediate_size = 11008,
            .max_seq_len = 4096,
            .epsilon = 1e-5,
            .rope_theta = 10000.0,
            .use_cache = true,
        };
    }
};

pub const KVCache = struct {
    keys: []Tensor,
    values: []Tensor,
    sequence_length: usize,
    max_length: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !KVCache {
        const keys = try allocator.alloc(Tensor, config.num_layers);
        const values = try allocator.alloc(Tensor, config.num_layers);

        for (0..config.num_layers) |i| {
            const shape = TensorShape.init(&[_]usize{ 1, config.max_seq_len, config.num_heads, config.head_dim });
            keys[i] = try Tensor.init(allocator, shape);
            values[i] = try Tensor.init(allocator, shape);
        }

        return KVCache{
            .keys = keys,
            .values = values,
            .sequence_length = 0,
            .max_length = config.max_seq_len,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.keys) |*k| k.deinit();
        for (self.values) |*v| v.deinit();
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
    }

    pub fn reset(self: *KVCache) void {
        self.sequence_length = 0;
    }

    pub fn append(self: *KVCache, layer_idx: usize, new_keys: *const Tensor, new_values: *const Tensor) void {
        const num_new = new_keys.shape.dims[1];
        const num_heads = new_keys.shape.dims[2];
        const head_dim = new_keys.shape.dims[3];

        for (0..num_new) |n| {
            for (0..num_heads) |h| {
                for (0..head_dim) |d| {
                    const src_idx = (n * num_heads + h) * head_dim + d;
                    const dst_idx = ((self.sequence_length + n) * num_heads + h) * head_dim + d;
                    self.keys[layer_idx].data[dst_idx] = new_keys.data[src_idx];
                    self.values[layer_idx].data[dst_idx] = new_values.data[src_idx];
                }
            }
        }
        self.sequence_length += num_new;
    }
};

pub const InferenceEngine = struct {
    config: ModelConfig,
    layers: std.ArrayList(InferenceLayer),
    kv_cache: ?KVCache,
    allocator: std.mem.Allocator,
    profiler: ?*InferenceProfiler,
    is_initialized: bool,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !InferenceEngine {
        var engine = InferenceEngine{
            .config = config,
            .layers = std.ArrayList(InferenceLayer).init(allocator),
            .kv_cache = null,
            .allocator = allocator,
            .profiler = null,
            .is_initialized = false,
        };

        if (config.use_cache) {
            engine.kv_cache = try KVCache.init(allocator, config);
        }

        return engine;
    }

    pub fn deinit(self: *InferenceEngine) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.layers.deinit();

        if (self.kv_cache) |*cache| {
            cache.deinit();
        }
    }

    pub fn addLayer(self: *InferenceEngine, config: LayerConfig) !void {
        const layer = try InferenceLayer.init(self.allocator, config);
        try self.layers.append(layer);
    }

    pub fn forward(self: *InferenceEngine, input: *const Tensor) !Tensor {
        if (self.layers.items.len == 0) {
            return InferenceError.ModelNotLoaded;
        }

        var current = try input.clone();

        for (self.layers.items) |*layer| {
            const output = try layer.forward(self.allocator, &current);
            current.deinit();
            current = output;
        }

        return current;
    }

    pub fn generate(self: *InferenceEngine, input_ids: []const i32, max_new_tokens: usize, temperature: f32, top_p: f32) ![]i32 {
        var output_ids = try self.allocator.alloc(i32, input_ids.len + max_new_tokens);
        @memcpy(output_ids[0..input_ids.len], input_ids);

        var current_len = input_ids.len;

        for (0..max_new_tokens) |_| {
            const logits = try self.forwardStep(output_ids[0..current_len]);
            defer logits.deinit();

            const next_token = try self.sampleToken(&logits, temperature, top_p);
            output_ids[current_len] = next_token;
            current_len += 1;
        }

        return output_ids[0..current_len];
    }

    fn forwardStep(self: *InferenceEngine, input_ids: []const i32) !Tensor {
        const batch_size: usize = 1;
        const seq_len = input_ids.len;

        const input_shape = TensorShape.init(&[_]usize{ batch_size, seq_len });
        var input_tensor = try Tensor.init(self.allocator, input_shape);
        defer input_tensor.deinit();

        for (input_ids, 0..) |id, i| {
            input_tensor.data[i] = @floatFromInt(id);
        }

        return self.forward(&input_tensor);
    }

    fn sampleToken(self: *InferenceEngine, logits: *const Tensor, temperature: f32, top_p: f32) !i32 {
        const vocab_size = logits.shape.dims[logits.shape.ndim - 1];
        const last_logits = logits.data[logits.data.len - vocab_size ..];

        var probs = try self.allocator.alloc(f32, vocab_size);
        defer self.allocator.free(probs);

        var max_logit = last_logits[0];
        for (last_logits[1..]) |l| {
            if (l > max_logit) max_logit = l;
        }

        var sum: f32 = 0.0;
        for (last_logits, 0..) |l, i| {
            probs[i] = @exp((l - max_logit) / temperature);
            sum += probs[i];
        }

        for (probs) |*p| {
            p.* /= sum;
        }

        var indices = try self.allocator.alloc(usize, vocab_size);
        defer self.allocator.free(indices);
        for (indices, 0..) |*idx, i| idx.* = i;

        std.mem.sort(usize, indices, probs, struct {
            fn lessThan(p: []f32, a: usize, b: usize) bool {
                return p[a] > p[b];
            }
        }.lessThan);

        var cumsum: f32 = 0.0;
        var cutoff_idx: usize = vocab_size;
        for (indices, 0..) |idx, i| {
            cumsum += probs[idx];
            if (cumsum > top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }

        var random_val: f32 = 0.5;
        var selected: f32 = 0.0;
        for (0..cutoff_idx) |i| {
            selected += probs[indices[i]];
            if (selected > random_val * cumsum) {
                return @intCast(indices[i]);
            }
        }

        return @intCast(indices[0]);
    }

    pub fn resetCache(self: *InferenceEngine) void {
        if (self.kv_cache) |*cache| {
            cache.reset();
        }
    }

    pub fn setProfiler(self: *InferenceEngine, profiler: *InferenceProfiler) void {
        self.profiler = profiler;
    }
};

pub const InferenceProfiler = struct {
    layer_times: std.ArrayList(u64),
    total_time: u64,
    num_tokens: usize,
    num_forward_passes: usize,
    memory_usage: usize,
    peak_memory: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) InferenceProfiler {
        return InferenceProfiler{
            .layer_times = std.ArrayList(u64).init(allocator),
            .total_time = 0,
            .num_tokens = 0,
            .num_forward_passes = 0,
            .memory_usage = 0,
            .peak_memory = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *InferenceProfiler) void {
        self.layer_times.deinit();
    }

    pub fn recordLayerTime(self: *InferenceProfiler, time_ns: u64) !void {
        try self.layer_times.append(time_ns);
    }

    pub fn recordForwardPass(self: *InferenceProfiler, time_ns: u64, num_tokens: usize) void {
        self.total_time += time_ns;
        self.num_tokens += num_tokens;
        self.num_forward_passes += 1;
    }

    pub fn tokensPerSecond(self: *const InferenceProfiler) f64 {
        if (self.total_time == 0) return 0.0;
        return @as(f64, @floatFromInt(self.num_tokens)) / (@as(f64, @floatFromInt(self.total_time)) / 1e9);
    }

    pub fn averageLatencyMs(self: *const InferenceProfiler) f64 {
        if (self.num_forward_passes == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_time)) / @as(f64, @floatFromInt(self.num_forward_passes)) / 1e6;
    }

    pub fn reset(self: *InferenceProfiler) void {
        self.layer_times.clearRetainingCapacity();
        self.total_time = 0;
        self.num_tokens = 0;
        self.num_forward_passes = 0;
    }
};

pub const QuantizationConfig = struct {
    bits: u8,
    group_size: usize,
    symmetric: bool,
    per_channel: bool,

    pub fn int8() QuantizationConfig {
        return QuantizationConfig{
            .bits = 8,
            .group_size = 128,
            .symmetric = true,
            .per_channel = true,
        };
    }

    pub fn int4() QuantizationConfig {
        return QuantizationConfig{
            .bits = 4,
            .group_size = 32,
            .symmetric = false,
            .per_channel = true,
        };
    }
};

pub const QuantizedTensor = struct {
    data: []i8,
    scales: []f32,
    zero_points: []i8,
    shape: TensorShape,
    config: QuantizationConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, shape: TensorShape, config: QuantizationConfig) !QuantizedTensor {
        const size = shape.size();
        const num_groups = (size + config.group_size - 1) / config.group_size;

        return QuantizedTensor{
            .data = try allocator.alloc(i8, size),
            .scales = try allocator.alloc(f32, num_groups),
            .zero_points = try allocator.alloc(i8, num_groups),
            .shape = shape,
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *QuantizedTensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.scales);
        self.allocator.free(self.zero_points);
    }

    pub fn quantize(self: *QuantizedTensor, input: *const Tensor) void {
        const size = input.shape.size();
        const group_size = self.config.group_size;

        var group_idx: usize = 0;
        var i: usize = 0;
        while (i < size) {
            const group_end = @min(i + group_size, size);

            var min_val: f32 = input.data[i];
            var max_val: f32 = input.data[i];
            for (i + 1..group_end) |j| {
                if (input.data[j] < min_val) min_val = input.data[j];
                if (input.data[j] > max_val) max_val = input.data[j];
            }

            const qmin: f32 = -128.0;
            const qmax: f32 = 127.0;
            const scale = (max_val - min_val) / (qmax - qmin);
            const zero_point = @as(i8, @intFromFloat(-min_val / scale + qmin));

            self.scales[group_idx] = scale;
            self.zero_points[group_idx] = zero_point;

            for (i..group_end) |j| {
                const q = @round(input.data[j] / scale) + @as(f32, @floatFromInt(zero_point));
                self.data[j] = @intFromFloat(@max(qmin, @min(qmax, q)));
            }

            i = group_end;
            group_idx += 1;
        }
    }

    pub fn dequantize(self: *const QuantizedTensor, allocator: std.mem.Allocator) !Tensor {
        var output = try Tensor.init(allocator, self.shape);
        const size = self.shape.size();
        const group_size = self.config.group_size;

        var group_idx: usize = 0;
        var i: usize = 0;
        while (i < size) {
            const group_end = @min(i + group_size, size);
            const scale = self.scales[group_idx];
            const zero_point = @as(f32, @floatFromInt(self.zero_points[group_idx]));

            for (i..group_end) |j| {
                output.data[j] = (@as(f32, @floatFromInt(self.data[j])) - zero_point) * scale;
            }

            i = group_end;
            group_idx += 1;
        }

        return output;
    }
};

pub const BatchProcessor = struct {
    max_batch_size: usize,
    max_seq_len: usize,
    padding_idx: i32,
    attention_masks: std.ArrayList(Tensor),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_batch_size: usize, max_seq_len: usize) BatchProcessor {
        return BatchProcessor{
            .max_batch_size = max_batch_size,
            .max_seq_len = max_seq_len,
            .padding_idx = 0,
            .attention_masks = std.ArrayList(Tensor).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BatchProcessor) void {
        for (self.attention_masks.items) |*mask| {
            mask.deinit();
        }
        self.attention_masks.deinit();
    }

    pub fn padSequences(self: *BatchProcessor, sequences: []const []const i32) !struct { padded: Tensor, mask: Tensor } {
        var max_len: usize = 0;
        for (sequences) |seq| {
            if (seq.len > max_len) max_len = seq.len;
        }
        max_len = @min(max_len, self.max_seq_len);

        const batch_size = sequences.len;
        const padded_shape = TensorShape.init(&[_]usize{ batch_size, max_len });
        var padded = try Tensor.init(self.allocator, padded_shape);
        var mask = try Tensor.init(self.allocator, padded_shape);

        for (sequences, 0..) |seq, b| {
            const seq_len = @min(seq.len, max_len);
            for (0..seq_len) |s| {
                padded.data[b * max_len + s] = @floatFromInt(seq[s]);
                mask.data[b * max_len + s] = 1.0;
            }
            for (seq_len..max_len) |s| {
                padded.data[b * max_len + s] = @floatFromInt(self.padding_idx);
                mask.data[b * max_len + s] = 0.0;
            }
        }

        return .{ .padded = padded, .mask = mask };
    }
};

pub const TargetPlatform = enum {
    x86_64,
    x86_64_avx2,
    x86_64_avx512,
    aarch64,
    aarch64_neon,
    wasm32,
    generic,
};

pub const PlatformOptimizer = struct {
    target: TargetPlatform,
    use_simd: bool,
    num_threads: usize,

    pub fn detect() PlatformOptimizer {
        const target: TargetPlatform = switch (builtin.cpu.arch) {
            .x86_64 => blk: {
                if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) {
                    break :blk .x86_64_avx512;
                } else if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
                    break :blk .x86_64_avx2;
                } else {
                    break :blk .x86_64;
                }
            },
            .aarch64 => .aarch64_neon,
            .wasm32 => .wasm32,
            else => .generic,
        };

        return PlatformOptimizer{
            .target = target,
            .use_simd = target != .generic and target != .wasm32,
            .num_threads = @max(1, std.Thread.getCpuCount() catch 1),
        };
    }

    pub fn vectorWidth(self: *const PlatformOptimizer) usize {
        return switch (self.target) {
            .x86_64_avx512 => 16,
            .x86_64_avx2 => 8,
            .x86_64 => 4,
            .aarch64_neon, .aarch64 => 4,
            .wasm32 => 4,
            .generic => 1,
        };
    }
};

test "tensor operations" {
    const allocator = std.testing.allocator;

    const shape = TensorShape.init(&[_]usize{ 2, 3 });
    var tensor = try Tensor.init(allocator, shape);
    defer tensor.deinit();

    try std.testing.expect(tensor.shape.size() == 6);
    try std.testing.expect(tensor.data.len == 6);

    tensor.fill(1.0);
    for (tensor.data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v, 1e-6);
    }
}

test "inference layer" {
    const allocator = std.testing.allocator;

    var config = LayerConfig.default();
    config.layer_type = .relu;

    var layer = try InferenceLayer.init(allocator, config);
    defer layer.deinit();

    const input_shape = TensorShape.init(&[_]usize{ 1, 4 });
    var input = try Tensor.init(allocator, input_shape);
    defer input.deinit();

    input.data[0] = -1.0;
    input.data[1] = 0.0;
    input.data[2] = 1.0;
    input.data[3] = -2.0;

    var output = try layer.forward(allocator, &input);
    defer output.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output.data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output.data[3], 1e-6);
}

test "quantization" {
    const allocator = std.testing.allocator;

    const shape = TensorShape.init(&[_]usize{ 1, 128 });
    var tensor = try Tensor.init(allocator, shape);
    defer tensor.deinit();

    for (tensor.data, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / 128.0 - 0.5;
    }

    const config = QuantizationConfig.int8();
    var quantized = try QuantizedTensor.init(allocator, shape, config);
    defer quantized.deinit();

    quantized.quantize(&tensor);

    var dequantized = try quantized.dequantize(allocator);
    defer dequantized.deinit();

    var max_error: f32 = 0.0;
    for (tensor.data, 0..) |v, i| {
        const error_val = @abs(v - dequantized.data[i]);
        if (error_val > max_error) max_error = error_val;
    }

    try std.testing.expect(max_error < 0.01);
}
