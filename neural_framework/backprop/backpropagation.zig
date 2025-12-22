const std = @import("std");
const math = std.math;

pub const LayerType = enum(u8) {
    linear,
    conv2d,
    conv_transpose2d,
    batch_norm,
    layer_norm,
    rms_norm,
    group_norm,
    instance_norm,
    dropout,
    relu,
    leaky_relu,
    gelu,
    silu,
    tanh,
    sigmoid,
    softmax,
    log_softmax,
    max_pool2d,
    avg_pool2d,
    adaptive_avg_pool2d,
    flatten,
    unflatten,
    embedding,
    multi_head_attention,
    cross_attention,
    positional_encoding,
    residual,
    sequential,
};

pub const ActivationType = enum(u8) {
    none,
    relu,
    leaky_relu,
    elu,
    selu,
    gelu,
    gelu_new,
    quick_gelu,
    silu,
    mish,
    softplus,
    softsign,
    tanh,
    sigmoid,
    hard_sigmoid,
    hard_swish,
    hard_tanh,
};

pub const TensorDescriptor = struct {
    data: []f32,
    shape: []const usize,
    strides: []const usize,
    ndim: usize,
    size: usize,
    requires_grad: bool,
    grad: ?[]f32,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize, requires_grad: bool) !TensorDescriptor {
        var size: usize = 1;
        for (shape) |dim| {
            size *= dim;
        }

        const strides = try allocator.alloc(usize, shape.len);
        var stride: usize = 1;
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= shape[i];
        }

        const data = try allocator.alloc(f32, size);
        @memset(data, 0.0);

        const grad = if (requires_grad) try allocator.alloc(f32, size) else null;
        if (grad) |g| {
            @memset(g, 0.0);
        }

        return TensorDescriptor{
            .data = data,
            .shape = shape,
            .strides = strides,
            .ndim = shape.len,
            .size = size,
            .requires_grad = requires_grad,
            .grad = grad,
        };
    }

    pub fn deinit(self: *TensorDescriptor, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        allocator.free(self.strides);
        if (self.grad) |g| {
            allocator.free(g);
        }
    }

    pub fn zeroGrad(self: *TensorDescriptor) void {
        if (self.grad) |g| {
            @memset(g, 0.0);
        }
    }
};

pub const LayerParams = struct {
    weight: ?TensorDescriptor,
    bias: ?TensorDescriptor,
    running_mean: ?TensorDescriptor,
    running_var: ?TensorDescriptor,
    gamma: ?TensorDescriptor,
    beta: ?TensorDescriptor,

    pub fn init() LayerParams {
        return LayerParams{
            .weight = null,
            .bias = null,
            .running_mean = null,
            .running_var = null,
            .gamma = null,
            .beta = null,
        };
    }
};

pub const LayerCache = struct {
    input: ?[]f32,
    output: ?[]f32,
    mask: ?[]f32,
    pre_activation: ?[]f32,
    normalized: ?[]f32,
    mean: ?[]f32,
    variance: ?[]f32,
    max_indices: ?[]i32,
    attention_weights: ?[]f32,

    pub fn init() LayerCache {
        return LayerCache{
            .input = null,
            .output = null,
            .mask = null,
            .pre_activation = null,
            .normalized = null,
            .mean = null,
            .variance = null,
            .max_indices = null,
            .attention_weights = null,
        };
    }

    pub fn clear(self: *LayerCache, allocator: std.mem.Allocator) void {
        if (self.input) |i| allocator.free(i);
        if (self.output) |o| allocator.free(o);
        if (self.mask) |m| allocator.free(m);
        if (self.pre_activation) |p| allocator.free(p);
        if (self.normalized) |n| allocator.free(n);
        if (self.mean) |m| allocator.free(m);
        if (self.variance) |v| allocator.free(v);
        if (self.max_indices) |i| allocator.free(i);
        if (self.attention_weights) |a| allocator.free(a);
        self.* = LayerCache.init();
    }
};

pub const LayerConfig = struct {
    layer_type: LayerType,
    in_features: usize,
    out_features: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [2]usize,
    stride: [2]usize,
    padding: [2]usize,
    dilation: [2]usize,
    groups: usize,
    bias: bool,
    dropout_p: f32,
    epsilon: f32,
    momentum: f32,
    affine: bool,
    track_running_stats: bool,
    num_heads: usize,
    head_dim: usize,
    activation: ActivationType,
    negative_slope: f32,

    pub fn default() LayerConfig {
        return LayerConfig{
            .layer_type = .linear,
            .in_features = 0,
            .out_features = 0,
            .in_channels = 0,
            .out_channels = 0,
            .kernel_size = .{ 3, 3 },
            .stride = .{ 1, 1 },
            .padding = .{ 0, 0 },
            .dilation = .{ 1, 1 },
            .groups = 1,
            .bias = true,
            .dropout_p = 0.0,
            .epsilon = 1e-5,
            .momentum = 0.1,
            .affine = true,
            .track_running_stats = true,
            .num_heads = 1,
            .head_dim = 64,
            .activation = .none,
            .negative_slope = 0.01,
        };
    }
};

pub const ComputationNode = struct {
    layer_idx: usize,
    inputs: std.ArrayList(usize),
    outputs: std.ArrayList(usize),
    grad_computed: bool,

    pub fn init(allocator: std.mem.Allocator, layer_idx: usize) ComputationNode {
        return ComputationNode{
            .layer_idx = layer_idx,
            .inputs = std.ArrayList(usize).init(allocator),
            .outputs = std.ArrayList(usize).init(allocator),
            .grad_computed = false,
        };
    }

    pub fn deinit(self: *ComputationNode) void {
        self.inputs.deinit();
        self.outputs.deinit();
    }
};

pub const BackpropEngine = struct {
    allocator: std.mem.Allocator,
    layers: std.ArrayList(Layer),
    nodes: std.ArrayList(ComputationNode),
    param_grads: std.AutoHashMap(usize, []f32),
    gradient_checkpointing: bool,
    checkpoints: std.ArrayList(usize),
    mixed_precision: bool,
    grad_scaling: f32,
    max_grad_norm: f32,
    training: bool,

    pub fn init(allocator: std.mem.Allocator) BackpropEngine {
        return BackpropEngine{
            .allocator = allocator,
            .layers = std.ArrayList(Layer).init(allocator),
            .nodes = std.ArrayList(ComputationNode).init(allocator),
            .param_grads = std.AutoHashMap(usize, []f32).init(allocator),
            .gradient_checkpointing = false,
            .checkpoints = std.ArrayList(usize).init(allocator),
            .mixed_precision = false,
            .grad_scaling = 1.0,
            .max_grad_norm = 1.0,
            .training = true,
        };
    }

    pub fn deinit(self: *BackpropEngine) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.layers.deinit();

        for (self.nodes.items) |*node| {
            node.deinit();
        }
        self.nodes.deinit();

        var iter = self.param_grads.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.param_grads.deinit();
        self.checkpoints.deinit();
    }

    pub fn addLayer(self: *BackpropEngine, config: LayerConfig) !usize {
        const layer = try Layer.init(self.allocator, config);
        try self.layers.append(layer);
        
        const node = ComputationNode.init(self.allocator, self.layers.items.len - 1);
        try self.nodes.append(node);
        
        return self.layers.items.len - 1;
    }

    pub fn forward(self: *BackpropEngine, input: []const f32, input_shape: []const usize) ![]f32 {
        var current_data = try self.allocator.alloc(f32, input.len);
        @memcpy(current_data, input);

        var current_shape = try self.allocator.alloc(usize, input_shape.len);
        @memcpy(current_shape, input_shape);

        for (self.layers.items, 0..) |*layer, idx| {
            const result = try layer.forward(self.allocator, current_data, current_shape, self.training);
            
            if (self.gradient_checkpointing) {
                for (self.checkpoints.items) |cp| {
                    if (cp == idx) {
                        layer.cache.input = try self.allocator.alloc(f32, current_data.len);
                        @memcpy(layer.cache.input.?, current_data);
                        break;
                    }
                }
            } else {
                layer.cache.input = try self.allocator.alloc(f32, current_data.len);
                @memcpy(layer.cache.input.?, current_data);
            }

            self.allocator.free(current_data);
            self.allocator.free(current_shape);

            current_data = result.data;
            current_shape = result.shape;
        }

        self.allocator.free(current_shape);
        return current_data;
    }

    pub fn backward(self: *BackpropEngine, grad_output: []const f32) !void {
        var current_grad = try self.allocator.alloc(f32, grad_output.len);
        @memcpy(current_grad, grad_output);

        if (self.mixed_precision) {
            for (current_grad) |*g| {
                g.* *= self.grad_scaling;
            }
        }

        var i = self.layers.items.len;
        while (i > 0) {
            i -= 1;
            const layer = &self.layers.items[i];
            
            const result = try layer.backward(self.allocator, current_grad);
            
            self.allocator.free(current_grad);
            current_grad = result.grad_input;

            if (result.grad_weight) |gw| {
                try self.param_grads.put(i * 2, gw);
            }
            if (result.grad_bias) |gb| {
                try self.param_grads.put(i * 2 + 1, gb);
            }
        }

        self.allocator.free(current_grad);
    }

    pub fn clipGradNorm(self: *BackpropEngine) f32 {
        var total_norm: f32 = 0.0;
        
        var iter = self.param_grads.iterator();
        while (iter.next()) |entry| {
            for (entry.value_ptr.*) |g| {
                total_norm += g * g;
            }
        }
        total_norm = @sqrt(total_norm);

        if (total_norm > self.max_grad_norm) {
            const scale = self.max_grad_norm / total_norm;
            var iter2 = self.param_grads.iterator();
            while (iter2.next()) |entry| {
                for (entry.value_ptr.*) |*g| {
                    g.* *= scale;
                }
            }
        }

        return total_norm;
    }

    pub fn zeroGrad(self: *BackpropEngine) void {
        for (self.layers.items) |*layer| {
            if (layer.params.weight) |*w| w.zeroGrad();
            if (layer.params.bias) |*b| b.zeroGrad();
            if (layer.params.gamma) |*g| g.zeroGrad();
            if (layer.params.beta) |*bt| bt.zeroGrad();
        }

        var iter = self.param_grads.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.param_grads.clearRetainingCapacity();
    }

    pub fn setTraining(self: *BackpropEngine, training: bool) void {
        self.training = training;
    }

    pub fn enableGradientCheckpointing(self: *BackpropEngine, checkpoints: []const usize) !void {
        self.gradient_checkpointing = true;
        self.checkpoints.clearRetainingCapacity();
        for (checkpoints) |cp| {
            try self.checkpoints.append(cp);
        }
    }
};

pub const Layer = struct {
    config: LayerConfig,
    params: LayerParams,
    cache: LayerCache,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: LayerConfig) !Layer {
        var layer = Layer{
            .config = config,
            .params = LayerParams.init(),
            .cache = LayerCache.init(),
            .allocator = allocator,
        };

        try layer.initParams();
        return layer;
    }

    pub fn deinit(self: *Layer) void {
        if (self.params.weight) |*w| w.deinit(self.allocator);
        if (self.params.bias) |*b| b.deinit(self.allocator);
        if (self.params.running_mean) |*rm| rm.deinit(self.allocator);
        if (self.params.running_var) |*rv| rv.deinit(self.allocator);
        if (self.params.gamma) |*g| g.deinit(self.allocator);
        if (self.params.beta) |*bt| bt.deinit(self.allocator);
        self.cache.clear(self.allocator);
    }

    fn initParams(self: *Layer) !void {
        switch (self.config.layer_type) {
            .linear => {
                const weight_shape = [_]usize{ self.config.out_features, self.config.in_features };
                self.params.weight = try TensorDescriptor.init(self.allocator, &weight_shape, true);
                try self.initWeight(self.params.weight.?.data, self.config.in_features);

                if (self.config.bias) {
                    const bias_shape = [_]usize{self.config.out_features};
                    self.params.bias = try TensorDescriptor.init(self.allocator, &bias_shape, true);
                }
            },
            .conv2d => {
                const weight_shape = [_]usize{
                    self.config.out_channels,
                    self.config.in_channels / self.config.groups,
                    self.config.kernel_size[0],
                    self.config.kernel_size[1],
                };
                self.params.weight = try TensorDescriptor.init(self.allocator, &weight_shape, true);
                const fan_in = (self.config.in_channels / self.config.groups) * self.config.kernel_size[0] * self.config.kernel_size[1];
                try self.initWeight(self.params.weight.?.data, fan_in);

                if (self.config.bias) {
                    const bias_shape = [_]usize{self.config.out_channels};
                    self.params.bias = try TensorDescriptor.init(self.allocator, &bias_shape, true);
                }
            },
            .batch_norm, .layer_norm, .group_norm, .instance_norm, .rms_norm => {
                const num_features = if (self.config.layer_type == .batch_norm or self.config.layer_type == .instance_norm)
                    self.config.in_channels
                else
                    self.config.in_features;

                if (self.config.affine) {
                    const gamma_shape = [_]usize{num_features};
                    self.params.gamma = try TensorDescriptor.init(self.allocator, &gamma_shape, true);
                    for (self.params.gamma.?.data) |*g| {
                        g.* = 1.0;
                    }

                    const beta_shape = [_]usize{num_features};
                    self.params.beta = try TensorDescriptor.init(self.allocator, &beta_shape, true);
                }

                if (self.config.track_running_stats and (self.config.layer_type == .batch_norm)) {
                    const mean_shape = [_]usize{num_features};
                    self.params.running_mean = try TensorDescriptor.init(self.allocator, &mean_shape, false);

                    const var_shape = [_]usize{num_features};
                    self.params.running_var = try TensorDescriptor.init(self.allocator, &var_shape, false);
                    for (self.params.running_var.?.data) |*v| {
                        v.* = 1.0;
                    }
                }
            },
            .embedding => {
                const weight_shape = [_]usize{ self.config.in_features, self.config.out_features };
                self.params.weight = try TensorDescriptor.init(self.allocator, &weight_shape, true);
                try self.initWeightNormal(self.params.weight.?.data, 0.0, 1.0);
            },
            .multi_head_attention, .cross_attention => {
                const qkv_size = self.config.num_heads * self.config.head_dim;
                
                const q_shape = [_]usize{ qkv_size, self.config.in_features };
                self.params.weight = try TensorDescriptor.init(self.allocator, &q_shape, true);
                try self.initWeight(self.params.weight.?.data, self.config.in_features);
            },
            else => {},
        }
    }

    fn initWeight(self: *Layer, data: []f32, fan_in: usize) !void {
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in)));

        for (data) |*w| {
            const u1 = rng.random().float(f32);
            const u2 = rng.random().float(f32);
            const normal = @sqrt(-2.0 * @log(u1 + 1e-10)) * @cos(2.0 * math.pi * u2);
            w.* = normal * std_dev;
        }
    }

    fn initWeightNormal(self: *Layer, data: []f32, mean: f32, std_dev: f32) !void {
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));

        for (data) |*w| {
            const u1 = rng.random().float(f32);
            const u2 = rng.random().float(f32);
            const normal = @sqrt(-2.0 * @log(u1 + 1e-10)) * @cos(2.0 * math.pi * u2);
            w.* = mean + normal * std_dev;
        }
    }

    pub const ForwardResult = struct {
        data: []f32,
        shape: []usize,
    };

    pub fn forward(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize, training: bool) !ForwardResult {
        return switch (self.config.layer_type) {
            .linear => self.forwardLinear(allocator, input, input_shape),
            .conv2d => self.forwardConv2d(allocator, input, input_shape),
            .batch_norm => self.forwardBatchNorm(allocator, input, input_shape, training),
            .layer_norm => self.forwardLayerNorm(allocator, input, input_shape),
            .rms_norm => self.forwardRMSNorm(allocator, input, input_shape),
            .relu => self.forwardRelu(allocator, input, input_shape),
            .leaky_relu => self.forwardLeakyRelu(allocator, input, input_shape),
            .gelu => self.forwardGelu(allocator, input, input_shape),
            .silu => self.forwardSilu(allocator, input, input_shape),
            .tanh => self.forwardTanh(allocator, input, input_shape),
            .sigmoid => self.forwardSigmoid(allocator, input, input_shape),
            .softmax => self.forwardSoftmax(allocator, input, input_shape),
            .dropout => self.forwardDropout(allocator, input, input_shape, training),
            .max_pool2d => self.forwardMaxPool2d(allocator, input, input_shape),
            .avg_pool2d => self.forwardAvgPool2d(allocator, input, input_shape),
            .embedding => self.forwardEmbedding(allocator, input, input_shape),
            else => {
                const output = try allocator.alloc(f32, input.len);
                @memcpy(output, input);
                const shape = try allocator.alloc(usize, input_shape.len);
                @memcpy(shape, input_shape);
                return ForwardResult{ .data = output, .shape = shape };
            },
        };
    }

    fn forwardLinear(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        const batch_size = input_shape[0];
        const in_features = self.config.in_features;
        const out_features = self.config.out_features;

        const output = try allocator.alloc(f32, batch_size * out_features);

        const weight = self.params.weight.?.data;
        const bias = if (self.params.bias) |b| b.data else null;

        for (0..batch_size) |b| {
            for (0..out_features) |o| {
                var sum: f32 = 0.0;
                for (0..in_features) |i| {
                    sum += input[b * in_features + i] * weight[o * in_features + i];
                }
                if (bias) |bi| {
                    sum += bi[o];
                }
                output[b * out_features + o] = sum;
            }
        }

        const shape = try allocator.alloc(usize, 2);
        shape[0] = batch_size;
        shape[1] = out_features;

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardConv2d(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        const batch_size = input_shape[0];
        const in_channels = input_shape[1];
        const in_height = input_shape[2];
        const in_width = input_shape[3];

        const out_height = (in_height + 2 * self.config.padding[0] - self.config.dilation[0] * (self.config.kernel_size[0] - 1) - 1) / self.config.stride[0] + 1;
        const out_width = (in_width + 2 * self.config.padding[1] - self.config.dilation[1] * (self.config.kernel_size[1] - 1) - 1) / self.config.stride[1] + 1;

        const output = try allocator.alloc(f32, batch_size * self.config.out_channels * out_height * out_width);
        @memset(output, 0.0);

        const weight = self.params.weight.?.data;
        const bias = if (self.params.bias) |b| b.data else null;

        const ic_per_group = in_channels / self.config.groups;
        const oc_per_group = self.config.out_channels / self.config.groups;

        for (0..batch_size) |b| {
            for (0..self.config.groups) |g| {
                for (0..oc_per_group) |oc_g| {
                    const oc = g * oc_per_group + oc_g;
                    for (0..out_height) |oh| {
                        for (0..out_width) |ow| {
                            var sum: f32 = 0.0;
                            for (0..ic_per_group) |ic_g| {
                                const ic = g * ic_per_group + ic_g;
                                for (0..self.config.kernel_size[0]) |kh| {
                                    for (0..self.config.kernel_size[1]) |kw| {
                                        const ih_s: i64 = @as(i64, @intCast(oh * self.config.stride[0])) - @as(i64, @intCast(self.config.padding[0])) + @as(i64, @intCast(kh * self.config.dilation[0]));
                                        const iw_s: i64 = @as(i64, @intCast(ow * self.config.stride[1])) - @as(i64, @intCast(self.config.padding[1])) + @as(i64, @intCast(kw * self.config.dilation[1]));

                                        if (ih_s >= 0 and ih_s < @as(i64, @intCast(in_height)) and iw_s >= 0 and iw_s < @as(i64, @intCast(in_width))) {
                                            const ih: usize = @intCast(ih_s);
                                            const iw: usize = @intCast(iw_s);
                                            const input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                            const weight_idx = ((oc * ic_per_group + ic_g) * self.config.kernel_size[0] + kh) * self.config.kernel_size[1] + kw;
                                            sum += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            if (bias) |bi| {
                                sum += bi[oc];
                            }
                            const output_idx = ((b * self.config.out_channels + oc) * out_height + oh) * out_width + ow;
                            output[output_idx] = sum;
                        }
                    }
                }
            }
        }

        const shape = try allocator.alloc(usize, 4);
        shape[0] = batch_size;
        shape[1] = self.config.out_channels;
        shape[2] = out_height;
        shape[3] = out_width;

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardBatchNorm(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize, training: bool) !ForwardResult {
        const batch_size = input_shape[0];
        const channels = input_shape[1];
        const spatial_size = if (input_shape.len > 2) input_shape[2] * (if (input_shape.len > 3) input_shape[3] else 1) else 1;

        const output = try allocator.alloc(f32, input.len);

        self.cache.mean = try allocator.alloc(f32, channels);
        self.cache.variance = try allocator.alloc(f32, channels);

        if (training) {
            for (0..channels) |c| {
                var sum: f32 = 0.0;
                for (0..batch_size) |b| {
                    for (0..spatial_size) |s| {
                        sum += input[(b * channels + c) * spatial_size + s];
                    }
                }
                const mean = sum / @as(f32, @floatFromInt(batch_size * spatial_size));
                self.cache.mean.?[c] = mean;

                var var_sum: f32 = 0.0;
                for (0..batch_size) |b| {
                    for (0..spatial_size) |s| {
                        const diff = input[(b * channels + c) * spatial_size + s] - mean;
                        var_sum += diff * diff;
                    }
                }
                const variance = var_sum / @as(f32, @floatFromInt(batch_size * spatial_size));
                self.cache.variance.?[c] = variance;

                if (self.params.running_mean) |*rm| {
                    rm.data[c] = (1.0 - self.config.momentum) * rm.data[c] + self.config.momentum * mean;
                }
                if (self.params.running_var) |*rv| {
                    rv.data[c] = (1.0 - self.config.momentum) * rv.data[c] + self.config.momentum * variance;
                }
            }
        } else {
            if (self.params.running_mean) |rm| {
                @memcpy(self.cache.mean.?, rm.data);
            }
            if (self.params.running_var) |rv| {
                @memcpy(self.cache.variance.?, rv.data);
            }
        }

        for (0..batch_size) |b| {
            for (0..channels) |c| {
                const mean = self.cache.mean.?[c];
                const variance = self.cache.variance.?[c];
                const inv_std = 1.0 / @sqrt(variance + self.config.epsilon);

                const gamma = if (self.params.gamma) |g| g.data[c] else 1.0;
                const beta = if (self.params.beta) |bt| bt.data[c] else 0.0;

                for (0..spatial_size) |s| {
                    const idx = (b * channels + c) * spatial_size + s;
                    const normalized = (input[idx] - mean) * inv_std;
                    output[idx] = normalized * gamma + beta;
                }
            }
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardLayerNorm(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        const batch_size = input_shape[0];
        const hidden_size = self.config.in_features;

        const output = try allocator.alloc(f32, input.len);

        self.cache.mean = try allocator.alloc(f32, batch_size);
        self.cache.variance = try allocator.alloc(f32, batch_size);

        for (0..batch_size) |b| {
            var sum: f32 = 0.0;
            for (0..hidden_size) |h| {
                sum += input[b * hidden_size + h];
            }
            const mean = sum / @as(f32, @floatFromInt(hidden_size));
            self.cache.mean.?[b] = mean;

            var var_sum: f32 = 0.0;
            for (0..hidden_size) |h| {
                const diff = input[b * hidden_size + h] - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / @as(f32, @floatFromInt(hidden_size));
            self.cache.variance.?[b] = variance;

            const inv_std = 1.0 / @sqrt(variance + self.config.epsilon);

            for (0..hidden_size) |h| {
                const idx = b * hidden_size + h;
                const normalized = (input[idx] - mean) * inv_std;
                const gamma = if (self.params.gamma) |g| g.data[h] else 1.0;
                const beta = if (self.params.beta) |bt| bt.data[h] else 0.0;
                output[idx] = normalized * gamma + beta;
            }
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardRMSNorm(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        const batch_size = input_shape[0];
        const hidden_size = self.config.in_features;

        const output = try allocator.alloc(f32, input.len);

        self.cache.variance = try allocator.alloc(f32, batch_size);

        for (0..batch_size) |b| {
            var sum_sq: f32 = 0.0;
            for (0..hidden_size) |h| {
                const x = input[b * hidden_size + h];
                sum_sq += x * x;
            }
            const mean_sq = sum_sq / @as(f32, @floatFromInt(hidden_size));
            self.cache.variance.?[b] = mean_sq;

            const rms = 1.0 / @sqrt(mean_sq + self.config.epsilon);

            for (0..hidden_size) |h| {
                const idx = b * hidden_size + h;
                const gamma = if (self.params.gamma) |g| g.data[h] else 1.0;
                output[idx] = input[idx] * rms * gamma;
            }
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardRelu(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        _ = self;
        const output = try allocator.alloc(f32, input.len);

        for (input, 0..) |x, i| {
            output[i] = @max(0.0, x);
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardLeakyRelu(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        const output = try allocator.alloc(f32, input.len);

        for (input, 0..) |x, i| {
            output[i] = if (x > 0) x else self.config.negative_slope * x;
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardGelu(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        _ = self;
        const output = try allocator.alloc(f32, input.len);

        for (input, 0..) |x, i| {
            const x3 = x * x * x;
            const inner = 0.797884560803 * (x + 0.044715 * x3);
            output[i] = 0.5 * x * (1.0 + @tanh(inner));
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardSilu(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        _ = self;
        const output = try allocator.alloc(f32, input.len);

        for (input, 0..) |x, i| {
            output[i] = x / (1.0 + @exp(-x));
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardTanh(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        _ = self;
        const output = try allocator.alloc(f32, input.len);

        for (input, 0..) |x, i| {
            output[i] = @tanh(x);
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardSigmoid(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        _ = self;
        const output = try allocator.alloc(f32, input.len);

        for (input, 0..) |x, i| {
            output[i] = 1.0 / (1.0 + @exp(-x));
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardSoftmax(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        _ = self;
        const batch_size = input_shape[0];
        const num_classes = input_shape[1];

        const output = try allocator.alloc(f32, input.len);

        for (0..batch_size) |b| {
            var max_val: f32 = input[b * num_classes];
            for (1..num_classes) |c| {
                if (input[b * num_classes + c] > max_val) {
                    max_val = input[b * num_classes + c];
                }
            }

            var sum_exp: f32 = 0.0;
            for (0..num_classes) |c| {
                output[b * num_classes + c] = @exp(input[b * num_classes + c] - max_val);
                sum_exp += output[b * num_classes + c];
            }

            for (0..num_classes) |c| {
                output[b * num_classes + c] /= sum_exp;
            }
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardDropout(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize, training: bool) !ForwardResult {
        const output = try allocator.alloc(f32, input.len);

        if (!training or self.config.dropout_p == 0.0) {
            @memcpy(output, input);
        } else {
            self.cache.mask = try allocator.alloc(f32, input.len);
            var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));

            const scale = 1.0 / (1.0 - self.config.dropout_p);
            for (input, 0..) |x, i| {
                if (rng.random().float(f32) > self.config.dropout_p) {
                    output[i] = x * scale;
                    self.cache.mask.?[i] = scale;
                } else {
                    output[i] = 0.0;
                    self.cache.mask.?[i] = 0.0;
                }
            }
        }

        const shape = try allocator.alloc(usize, input_shape.len);
        @memcpy(shape, input_shape);

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardMaxPool2d(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        const batch_size = input_shape[0];
        const channels = input_shape[1];
        const in_height = input_shape[2];
        const in_width = input_shape[3];

        const out_height = (in_height + 2 * self.config.padding[0] - self.config.kernel_size[0]) / self.config.stride[0] + 1;
        const out_width = (in_width + 2 * self.config.padding[1] - self.config.kernel_size[1]) / self.config.stride[1] + 1;

        const output = try allocator.alloc(f32, batch_size * channels * out_height * out_width);
        self.cache.max_indices = try allocator.alloc(i32, batch_size * channels * out_height * out_width);

        for (0..batch_size) |b| {
            for (0..channels) |c| {
                for (0..out_height) |oh| {
                    for (0..out_width) |ow| {
                        var max_val: f32 = -math.inf(f32);
                        var max_idx: i32 = -1;

                        for (0..self.config.kernel_size[0]) |kh| {
                            for (0..self.config.kernel_size[1]) |kw| {
                                const ih_s: i64 = @as(i64, @intCast(oh * self.config.stride[0])) - @as(i64, @intCast(self.config.padding[0])) + @as(i64, @intCast(kh));
                                const iw_s: i64 = @as(i64, @intCast(ow * self.config.stride[1])) - @as(i64, @intCast(self.config.padding[1])) + @as(i64, @intCast(kw));

                                if (ih_s >= 0 and ih_s < @as(i64, @intCast(in_height)) and iw_s >= 0 and iw_s < @as(i64, @intCast(in_width))) {
                                    const ih: usize = @intCast(ih_s);
                                    const iw: usize = @intCast(iw_s);
                                    const input_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                                    if (input[input_idx] > max_val) {
                                        max_val = input[input_idx];
                                        max_idx = @intCast(input_idx);
                                    }
                                }
                            }
                        }

                        const output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                        output[output_idx] = max_val;
                        self.cache.max_indices.?[output_idx] = max_idx;
                    }
                }
            }
        }

        const shape = try allocator.alloc(usize, 4);
        shape[0] = batch_size;
        shape[1] = channels;
        shape[2] = out_height;
        shape[3] = out_width;

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardAvgPool2d(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        const batch_size = input_shape[0];
        const channels = input_shape[1];
        const in_height = input_shape[2];
        const in_width = input_shape[3];

        const out_height = (in_height + 2 * self.config.padding[0] - self.config.kernel_size[0]) / self.config.stride[0] + 1;
        const out_width = (in_width + 2 * self.config.padding[1] - self.config.kernel_size[1]) / self.config.stride[1] + 1;

        const output = try allocator.alloc(f32, batch_size * channels * out_height * out_width);

        for (0..batch_size) |b| {
            for (0..channels) |c| {
                for (0..out_height) |oh| {
                    for (0..out_width) |ow| {
                        var sum: f32 = 0.0;
                        var count: f32 = 0.0;

                        for (0..self.config.kernel_size[0]) |kh| {
                            for (0..self.config.kernel_size[1]) |kw| {
                                const ih_s: i64 = @as(i64, @intCast(oh * self.config.stride[0])) - @as(i64, @intCast(self.config.padding[0])) + @as(i64, @intCast(kh));
                                const iw_s: i64 = @as(i64, @intCast(ow * self.config.stride[1])) - @as(i64, @intCast(self.config.padding[1])) + @as(i64, @intCast(kw));

                                if (ih_s >= 0 and ih_s < @as(i64, @intCast(in_height)) and iw_s >= 0 and iw_s < @as(i64, @intCast(in_width))) {
                                    const ih: usize = @intCast(ih_s);
                                    const iw: usize = @intCast(iw_s);
                                    const input_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                                    sum += input[input_idx];
                                    count += 1.0;
                                }
                            }
                        }

                        const output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                        output[output_idx] = if (count > 0) sum / count else 0.0;
                    }
                }
            }
        }

        const shape = try allocator.alloc(usize, 4);
        shape[0] = batch_size;
        shape[1] = channels;
        shape[2] = out_height;
        shape[3] = out_width;

        return ForwardResult{ .data = output, .shape = shape };
    }

    fn forwardEmbedding(self: *Layer, allocator: std.mem.Allocator, input: []const f32, input_shape: []const usize) !ForwardResult {
        const batch_size = input_shape[0];
        const seq_length = input_shape[1];
        const embedding_dim = self.config.out_features;

        const output = try allocator.alloc(f32, batch_size * seq_length * embedding_dim);
        const weight = self.params.weight.?.data;

        for (0..batch_size) |b| {
            for (0..seq_length) |s| {
                const idx: usize = @intFromFloat(input[b * seq_length + s]);
                for (0..embedding_dim) |d| {
                    output[(b * seq_length + s) * embedding_dim + d] = weight[idx * embedding_dim + d];
                }
            }
        }

        const shape = try allocator.alloc(usize, 3);
        shape[0] = batch_size;
        shape[1] = seq_length;
        shape[2] = embedding_dim;

        return ForwardResult{ .data = output, .shape = shape };
    }

    pub const BackwardResult = struct {
        grad_input: []f32,
        grad_weight: ?[]f32,
        grad_bias: ?[]f32,
    };

    pub fn backward(self: *Layer, allocator: std.mem.Allocator, grad_output: []const f32) !BackwardResult {
        return switch (self.config.layer_type) {
            .linear => self.backwardLinear(allocator, grad_output),
            .relu => self.backwardRelu(allocator, grad_output),
            .sigmoid => self.backwardSigmoid(allocator, grad_output),
            .tanh => self.backwardTanh(allocator, grad_output),
            .dropout => self.backwardDropout(allocator, grad_output),
            else => {
                const grad_input = try allocator.alloc(f32, grad_output.len);
                @memcpy(grad_input, grad_output);
                return BackwardResult{ .grad_input = grad_input, .grad_weight = null, .grad_bias = null };
            },
        };
    }

    fn backwardLinear(self: *Layer, allocator: std.mem.Allocator, grad_output: []const f32) !BackwardResult {
        const input = self.cache.input orelse return BackwardResult{
            .grad_input = try allocator.alloc(f32, 1),
            .grad_weight = null,
            .grad_bias = null,
        };

        const batch_size = grad_output.len / self.config.out_features;
        const in_features = self.config.in_features;
        const out_features = self.config.out_features;

        const weight = self.params.weight.?.data;

        const grad_input = try allocator.alloc(f32, batch_size * in_features);
        @memset(grad_input, 0.0);

        for (0..batch_size) |b| {
            for (0..in_features) |i| {
                var sum: f32 = 0.0;
                for (0..out_features) |o| {
                    sum += grad_output[b * out_features + o] * weight[o * in_features + i];
                }
                grad_input[b * in_features + i] = sum;
            }
        }

        const grad_weight = try allocator.alloc(f32, out_features * in_features);
        @memset(grad_weight, 0.0);

        for (0..batch_size) |b| {
            for (0..out_features) |o| {
                for (0..in_features) |i| {
                    grad_weight[o * in_features + i] += grad_output[b * out_features + o] * input[b * in_features + i];
                }
            }
        }

        var grad_bias: ?[]f32 = null;
        if (self.config.bias) {
            grad_bias = try allocator.alloc(f32, out_features);
            @memset(grad_bias.?, 0.0);

            for (0..batch_size) |b| {
                for (0..out_features) |o| {
                    grad_bias.?[o] += grad_output[b * out_features + o];
                }
            }
        }

        return BackwardResult{
            .grad_input = grad_input,
            .grad_weight = grad_weight,
            .grad_bias = grad_bias,
        };
    }

    fn backwardRelu(self: *Layer, allocator: std.mem.Allocator, grad_output: []const f32) !BackwardResult {
        const input = self.cache.input orelse return BackwardResult{
            .grad_input = try allocator.alloc(f32, grad_output.len),
            .grad_weight = null,
            .grad_bias = null,
        };

        const grad_input = try allocator.alloc(f32, grad_output.len);

        for (grad_output, 0..) |g, i| {
            grad_input[i] = if (input[i] > 0) g else 0.0;
        }

        return BackwardResult{
            .grad_input = grad_input,
            .grad_weight = null,
            .grad_bias = null,
        };
    }

    fn backwardSigmoid(self: *Layer, allocator: std.mem.Allocator, grad_output: []const f32) !BackwardResult {
        const output = self.cache.output orelse return BackwardResult{
            .grad_input = try allocator.alloc(f32, grad_output.len),
            .grad_weight = null,
            .grad_bias = null,
        };

        const grad_input = try allocator.alloc(f32, grad_output.len);

        for (grad_output, 0..) |g, i| {
            const s = output[i];
            grad_input[i] = g * s * (1.0 - s);
        }

        return BackwardResult{
            .grad_input = grad_input,
            .grad_weight = null,
            .grad_bias = null,
        };
    }

    fn backwardTanh(self: *Layer, allocator: std.mem.Allocator, grad_output: []const f32) !BackwardResult {
        const output = self.cache.output orelse return BackwardResult{
            .grad_input = try allocator.alloc(f32, grad_output.len),
            .grad_weight = null,
            .grad_bias = null,
        };

        const grad_input = try allocator.alloc(f32, grad_output.len);

        for (grad_output, 0..) |g, i| {
            const t = output[i];
            grad_input[i] = g * (1.0 - t * t);
        }

        return BackwardResult{
            .grad_input = grad_input,
            .grad_weight = null,
            .grad_bias = null,
        };
    }

    fn backwardDropout(self: *Layer, allocator: std.mem.Allocator, grad_output: []const f32) !BackwardResult {
        const mask = self.cache.mask orelse {
            const grad_input = try allocator.alloc(f32, grad_output.len);
            @memcpy(grad_input, grad_output);
            return BackwardResult{
                .grad_input = grad_input,
                .grad_weight = null,
                .grad_bias = null,
            };
        };

        const grad_input = try allocator.alloc(f32, grad_output.len);

        for (grad_output, 0..) |g, i| {
            grad_input[i] = g * mask[i];
        }

        return BackwardResult{
            .grad_input = grad_input,
            .grad_weight = null,
            .grad_bias = null,
        };
    }
};

test "backprop engine initialization" {
    const allocator = std.testing.allocator;
    var engine = BackpropEngine.init(allocator);
    defer engine.deinit();

    var config = LayerConfig.default();
    config.layer_type = .linear;
    config.in_features = 784;
    config.out_features = 256;

    _ = try engine.addLayer(config);
    try std.testing.expect(engine.layers.items.len == 1);
}

test "relu forward" {
    const allocator = std.testing.allocator;
    var config = LayerConfig.default();
    config.layer_type = .relu;

    var layer = try Layer.init(allocator, config);
    defer layer.deinit();

    const input = [_]f32{ -1.0, 0.0, 1.0, -2.0, 2.0 };
    const input_shape = [_]usize{ 1, 5 };

    const result = try layer.forward(allocator, &input, &input_shape, true);
    defer allocator.free(result.data);
    defer allocator.free(result.shape);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.data[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result.data[4], 1e-6);
}
