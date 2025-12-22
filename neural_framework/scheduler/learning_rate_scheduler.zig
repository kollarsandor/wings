const std = @import("std");
const math = std.math;

pub const SchedulerType = enum(u8) {
    constant,
    step,
    multi_step,
    exponential,
    cosine_annealing,
    cosine_annealing_warm_restarts,
    linear,
    polynomial,
    one_cycle,
    cyclic,
    warmup_constant,
    warmup_linear,
    warmup_cosine,
    reduce_on_plateau,
    custom,
};

pub const AnnealStrategy = enum(u8) {
    linear,
    cosine,
};

pub const CyclicMode = enum(u8) {
    triangular,
    triangular2,
    exp_range,
};

pub const SchedulerConfig = struct {
    scheduler_type: SchedulerType,
    base_lr: f64,
    max_lr: f64,
    min_lr: f64,
    total_steps: u64,
    warmup_steps: u64,
    step_size: u64,
    gamma: f64,
    power: f64,
    t_max: u64,
    t_mult: f64,
    eta_min: f64,
    pct_start: f64,
    div_factor: f64,
    final_div_factor: f64,
    anneal_strategy: AnnealStrategy,
    cycle_momentum: bool,
    base_momentum: f64,
    max_momentum: f64,
    scale_fn: ?*const fn (f64) f64,
    milestones: []const u64,
    mode: []const u8,
    factor: f64,
    patience: u64,
    threshold: f64,
    cooldown: u64,
    verbose: bool,

    pub fn default() SchedulerConfig {
        return SchedulerConfig{
            .scheduler_type = .constant,
            .base_lr = 0.001,
            .max_lr = 0.01,
            .min_lr = 0.0,
            .total_steps = 10000,
            .warmup_steps = 1000,
            .step_size = 1000,
            .gamma = 0.1,
            .power = 1.0,
            .t_max = 10000,
            .t_mult = 1.0,
            .eta_min = 0.0,
            .pct_start = 0.3,
            .div_factor = 25.0,
            .final_div_factor = 10000.0,
            .anneal_strategy = .cosine,
            .cycle_momentum = true,
            .base_momentum = 0.85,
            .max_momentum = 0.95,
            .scale_fn = null,
            .milestones = &[_]u64{},
            .mode = "min",
            .factor = 0.1,
            .patience = 10,
            .threshold = 0.0001,
            .cooldown = 0,
            .verbose = false,
        };
    }
};

pub const SchedulerState = struct {
    current_step: u64,
    current_epoch: u64,
    current_lr: f64,
    current_momentum: f64,
    best_metric: f64,
    num_bad_epochs: u64,
    cooldown_counter: u64,
    t_cur: u64,
    cycle_count: u64,
    last_restart: u64,
    history: std.ArrayList(f64),

    pub fn init(allocator: std.mem.Allocator, initial_lr: f64) SchedulerState {
        return SchedulerState{
            .current_step = 0,
            .current_epoch = 0,
            .current_lr = initial_lr,
            .current_momentum = 0.9,
            .best_metric = math.inf(f64),
            .num_bad_epochs = 0,
            .cooldown_counter = 0,
            .t_cur = 0,
            .cycle_count = 0,
            .last_restart = 0,
            .history = std.ArrayList(f64).init(allocator),
        };
    }

    pub fn deinit(self: *SchedulerState) void {
        self.history.deinit();
    }

    pub fn recordLr(self: *SchedulerState, lr: f64) !void {
        try self.history.append(lr);
    }
};

pub const LearningRateScheduler = struct {
    config: SchedulerConfig,
    state: SchedulerState,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: SchedulerConfig) LearningRateScheduler {
        return LearningRateScheduler{
            .config = config,
            .state = SchedulerState.init(allocator, config.base_lr),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LearningRateScheduler) void {
        self.state.deinit();
    }

    pub fn step(self: *LearningRateScheduler) f64 {
        self.state.current_step += 1;
        
        const lr = switch (self.config.scheduler_type) {
            .constant => self.constantSchedule(),
            .step => self.stepSchedule(),
            .multi_step => self.multiStepSchedule(),
            .exponential => self.exponentialSchedule(),
            .cosine_annealing => self.cosineAnnealingSchedule(),
            .cosine_annealing_warm_restarts => self.cosineAnnealingWarmRestartsSchedule(),
            .linear => self.linearSchedule(),
            .polynomial => self.polynomialSchedule(),
            .one_cycle => self.oneCycleSchedule(),
            .cyclic => self.cyclicSchedule(),
            .warmup_constant => self.warmupConstantSchedule(),
            .warmup_linear => self.warmupLinearSchedule(),
            .warmup_cosine => self.warmupCosineSchedule(),
            .reduce_on_plateau => self.state.current_lr,
            .custom => self.customSchedule(),
        };

        self.state.current_lr = lr;
        self.state.recordLr(lr) catch {};
        
        return lr;
    }

    pub fn stepEpoch(self: *LearningRateScheduler) void {
        self.state.current_epoch += 1;
    }

    pub fn stepWithMetric(self: *LearningRateScheduler, metric: f64) f64 {
        if (self.config.scheduler_type != .reduce_on_plateau) {
            return self.step();
        }

        self.state.current_step += 1;
        
        if (self.state.cooldown_counter > 0) {
            self.state.cooldown_counter -= 1;
            self.state.num_bad_epochs = 0;
            return self.state.current_lr;
        }

        const is_better = if (std.mem.eql(u8, self.config.mode, "min"))
            metric < self.state.best_metric * (1.0 - self.config.threshold)
        else
            metric > self.state.best_metric * (1.0 + self.config.threshold);

        if (is_better) {
            self.state.best_metric = metric;
            self.state.num_bad_epochs = 0;
        } else {
            self.state.num_bad_epochs += 1;
        }

        if (self.state.num_bad_epochs > self.config.patience) {
            self.state.current_lr = @max(self.state.current_lr * self.config.factor, self.config.min_lr);
            self.state.cooldown_counter = self.config.cooldown;
            self.state.num_bad_epochs = 0;
        }

        self.state.recordLr(self.state.current_lr) catch {};
        return self.state.current_lr;
    }

    fn constantSchedule(self: *LearningRateScheduler) f64 {
        return self.config.base_lr;
    }

    fn stepSchedule(self: *LearningRateScheduler) f64 {
        const num_decays = self.state.current_step / self.config.step_size;
        return self.config.base_lr * math.pow(f64, self.config.gamma, @as(f64, @floatFromInt(num_decays)));
    }

    fn multiStepSchedule(self: *LearningRateScheduler) f64 {
        var lr = self.config.base_lr;
        for (self.config.milestones) |milestone| {
            if (self.state.current_step >= milestone) {
                lr *= self.config.gamma;
            }
        }
        return lr;
    }

    fn exponentialSchedule(self: *LearningRateScheduler) f64 {
        return self.config.base_lr * math.pow(f64, self.config.gamma, @as(f64, @floatFromInt(self.state.current_step)));
    }

    fn cosineAnnealingSchedule(self: *LearningRateScheduler) f64 {
        const t = @as(f64, @floatFromInt(self.state.current_step));
        const t_max = @as(f64, @floatFromInt(self.config.t_max));
        const cos_inner = math.pi * t / t_max;
        return self.config.eta_min + (self.config.base_lr - self.config.eta_min) * (1.0 + @cos(cos_inner)) / 2.0;
    }

    fn cosineAnnealingWarmRestartsSchedule(self: *LearningRateScheduler) f64 {
        var t_cur = self.state.t_cur;
        var t_i = self.config.t_max;

        while (t_cur >= t_i) {
            t_cur -= t_i;
            t_i = @as(u64, @intFromFloat(@as(f64, @floatFromInt(t_i)) * self.config.t_mult));
            self.state.cycle_count += 1;
        }

        self.state.t_cur = t_cur + 1;

        const t = @as(f64, @floatFromInt(t_cur));
        const t_max = @as(f64, @floatFromInt(t_i));
        const cos_inner = math.pi * t / t_max;
        return self.config.eta_min + (self.config.base_lr - self.config.eta_min) * (1.0 + @cos(cos_inner)) / 2.0;
    }

    fn linearSchedule(self: *LearningRateScheduler) f64 {
        const step = @as(f64, @floatFromInt(self.state.current_step));
        const total = @as(f64, @floatFromInt(self.config.total_steps));
        
        if (self.state.current_step >= self.config.total_steps) {
            return self.config.min_lr;
        }
        
        const progress = step / total;
        return self.config.base_lr * (1.0 - progress) + self.config.min_lr * progress;
    }

    fn polynomialSchedule(self: *LearningRateScheduler) f64 {
        const step = @min(self.state.current_step, self.config.total_steps);
        const step_f = @as(f64, @floatFromInt(step));
        const total = @as(f64, @floatFromInt(self.config.total_steps));
        
        const decay = math.pow(f64, 1.0 - step_f / total, self.config.power);
        return (self.config.base_lr - self.config.min_lr) * decay + self.config.min_lr;
    }

    fn oneCycleSchedule(self: *LearningRateScheduler) f64 {
        const step = @as(f64, @floatFromInt(self.state.current_step));
        const total = @as(f64, @floatFromInt(self.config.total_steps));
        const warmup = @as(f64, @floatFromInt(self.config.warmup_steps));

        const initial_lr = self.config.max_lr / self.config.div_factor;
        const final_lr = initial_lr / self.config.final_div_factor;

        const step1 = total * self.config.pct_start;
        const step2 = total - step1;

        if (step <= step1) {
            const pct = step / step1;
            return initial_lr + (self.config.max_lr - initial_lr) * pct;
        } else {
            const pct = (step - step1) / step2;
            return switch (self.config.anneal_strategy) {
                .cosine => blk: {
                    const cos_inner = math.pi * pct;
                    break :blk final_lr + (self.config.max_lr - final_lr) * (1.0 + @cos(cos_inner)) / 2.0;
                },
                .linear => self.config.max_lr - (self.config.max_lr - final_lr) * pct,
            };
        }
    }

    fn cyclicSchedule(self: *LearningRateScheduler) f64 {
        const step = @as(f64, @floatFromInt(self.state.current_step));
        const step_size = @as(f64, @floatFromInt(self.config.step_size));

        const cycle = @floor(1.0 + step / (2.0 * step_size));
        const x = @abs(step / step_size - 2.0 * cycle + 1.0);
        
        const base_lr = self.config.base_lr;
        const max_lr = self.config.max_lr;

        return base_lr + (max_lr - base_lr) * @max(0.0, 1.0 - x);
    }

    fn warmupConstantSchedule(self: *LearningRateScheduler) f64 {
        if (self.state.current_step < self.config.warmup_steps) {
            const step = @as(f64, @floatFromInt(self.state.current_step));
            const warmup = @as(f64, @floatFromInt(self.config.warmup_steps));
            return self.config.base_lr * step / warmup;
        }
        return self.config.base_lr;
    }

    fn warmupLinearSchedule(self: *LearningRateScheduler) f64 {
        if (self.state.current_step < self.config.warmup_steps) {
            const step = @as(f64, @floatFromInt(self.state.current_step));
            const warmup = @as(f64, @floatFromInt(self.config.warmup_steps));
            return self.config.base_lr * step / warmup;
        }

        const step = @as(f64, @floatFromInt(self.state.current_step - self.config.warmup_steps));
        const remaining = @as(f64, @floatFromInt(self.config.total_steps - self.config.warmup_steps));
        
        if (step >= remaining) {
            return self.config.min_lr;
        }
        
        const progress = step / remaining;
        return self.config.base_lr * (1.0 - progress) + self.config.min_lr * progress;
    }

    fn warmupCosineSchedule(self: *LearningRateScheduler) f64 {
        if (self.state.current_step < self.config.warmup_steps) {
            const step = @as(f64, @floatFromInt(self.state.current_step));
            const warmup = @as(f64, @floatFromInt(self.config.warmup_steps));
            return self.config.base_lr * step / warmup;
        }

        const step = @as(f64, @floatFromInt(self.state.current_step - self.config.warmup_steps));
        const remaining = @as(f64, @floatFromInt(self.config.total_steps - self.config.warmup_steps));
        
        const progress = step / remaining;
        const cos_inner = math.pi * progress;
        return self.config.eta_min + (self.config.base_lr - self.config.eta_min) * (1.0 + @cos(cos_inner)) / 2.0;
    }

    fn customSchedule(self: *LearningRateScheduler) f64 {
        if (self.config.scale_fn) |scale_fn| {
            const step = @as(f64, @floatFromInt(self.state.current_step));
            const total = @as(f64, @floatFromInt(self.config.total_steps));
            return self.config.base_lr * scale_fn(step / total);
        }
        return self.config.base_lr;
    }

    pub fn getLr(self: *const LearningRateScheduler) f64 {
        return self.state.current_lr;
    }

    pub fn getLastLr(self: *const LearningRateScheduler) ?f64 {
        if (self.state.history.items.len > 0) {
            return self.state.history.items[self.state.history.items.len - 1];
        }
        return null;
    }

    pub fn reset(self: *LearningRateScheduler) void {
        self.state.current_step = 0;
        self.state.current_epoch = 0;
        self.state.current_lr = self.config.base_lr;
        self.state.t_cur = 0;
        self.state.cycle_count = 0;
        self.state.last_restart = 0;
        self.state.best_metric = math.inf(f64);
        self.state.num_bad_epochs = 0;
        self.state.cooldown_counter = 0;
        self.state.history.clearRetainingCapacity();
    }
};

pub const MomentumScheduler = struct {
    base_momentum: f64,
    max_momentum: f64,
    total_steps: u64,
    warmup_steps: u64,
    current_step: u64,
    current_momentum: f64,
    cycle_momentum: bool,

    pub fn init(base_momentum: f64, max_momentum: f64, total_steps: u64, warmup_steps: u64, cycle_momentum: bool) MomentumScheduler {
        return MomentumScheduler{
            .base_momentum = base_momentum,
            .max_momentum = max_momentum,
            .total_steps = total_steps,
            .warmup_steps = warmup_steps,
            .current_step = 0,
            .current_momentum = base_momentum,
            .cycle_momentum = cycle_momentum,
        };
    }

    pub fn step(self: *MomentumScheduler) f64 {
        self.current_step += 1;

        if (!self.cycle_momentum) {
            return self.base_momentum;
        }

        const step = @as(f64, @floatFromInt(self.current_step));
        const total = @as(f64, @floatFromInt(self.total_steps));
        const pct = step / total;

        if (pct <= 0.3) {
            self.current_momentum = self.max_momentum - (self.max_momentum - self.base_momentum) * pct / 0.3;
        } else {
            const pct_rest = (pct - 0.3) / 0.7;
            self.current_momentum = self.base_momentum + (self.max_momentum - self.base_momentum) * pct_rest;
        }

        return self.current_momentum;
    }

    pub fn getMomentum(self: *const MomentumScheduler) f64 {
        return self.current_momentum;
    }

    pub fn reset(self: *MomentumScheduler) void {
        self.current_step = 0;
        self.current_momentum = self.base_momentum;
    }
};

pub const WarmupScheduler = struct {
    target_lr: f64,
    warmup_steps: u64,
    warmup_method: WarmupMethod,
    current_step: u64,

    pub const WarmupMethod = enum {
        linear,
        exponential,
        constant,
    };

    pub fn init(target_lr: f64, warmup_steps: u64, method: WarmupMethod) WarmupScheduler {
        return WarmupScheduler{
            .target_lr = target_lr,
            .warmup_steps = warmup_steps,
            .warmup_method = method,
            .current_step = 0,
        };
    }

    pub fn step(self: *WarmupScheduler) f64 {
        self.current_step += 1;

        if (self.current_step >= self.warmup_steps) {
            return self.target_lr;
        }

        const step = @as(f64, @floatFromInt(self.current_step));
        const warmup = @as(f64, @floatFromInt(self.warmup_steps));

        return switch (self.warmup_method) {
            .linear => self.target_lr * step / warmup,
            .exponential => self.target_lr * math.pow(f64, step / warmup, 2.0),
            .constant => self.target_lr * 0.1,
        };
    }

    pub fn isComplete(self: *const WarmupScheduler) bool {
        return self.current_step >= self.warmup_steps;
    }
};

pub const ChainedScheduler = struct {
    schedulers: std.ArrayList(*LearningRateScheduler),
    milestones: std.ArrayList(u64),
    current_scheduler_idx: usize,
    current_step: u64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ChainedScheduler {
        return ChainedScheduler{
            .schedulers = std.ArrayList(*LearningRateScheduler).init(allocator),
            .milestones = std.ArrayList(u64).init(allocator),
            .current_scheduler_idx = 0,
            .current_step = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ChainedScheduler) void {
        self.schedulers.deinit();
        self.milestones.deinit();
    }

    pub fn addScheduler(self: *ChainedScheduler, scheduler: *LearningRateScheduler, milestone: u64) !void {
        try self.schedulers.append(scheduler);
        try self.milestones.append(milestone);
    }

    pub fn step(self: *ChainedScheduler) f64 {
        self.current_step += 1;

        while (self.current_scheduler_idx < self.milestones.items.len - 1) {
            if (self.current_step < self.milestones.items[self.current_scheduler_idx + 1]) {
                break;
            }
            self.current_scheduler_idx += 1;
        }

        if (self.current_scheduler_idx < self.schedulers.items.len) {
            return self.schedulers.items[self.current_scheduler_idx].step();
        }

        return self.schedulers.items[self.schedulers.items.len - 1].getLr();
    }

    pub fn getLr(self: *const ChainedScheduler) f64 {
        if (self.current_scheduler_idx < self.schedulers.items.len) {
            return self.schedulers.items[self.current_scheduler_idx].getLr();
        }
        return 0.0;
    }
};

pub const LRRangeTest = struct {
    start_lr: f64,
    end_lr: f64,
    num_iter: u64,
    current_iter: u64,
    losses: std.ArrayList(f64),
    lrs: std.ArrayList(f64),
    smooth_factor: f64,
    diverge_threshold: f64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, start_lr: f64, end_lr: f64, num_iter: u64) LRRangeTest {
        return LRRangeTest{
            .start_lr = start_lr,
            .end_lr = end_lr,
            .num_iter = num_iter,
            .current_iter = 0,
            .losses = std.ArrayList(f64).init(allocator),
            .lrs = std.ArrayList(f64).init(allocator),
            .smooth_factor = 0.05,
            .diverge_threshold = 4.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LRRangeTest) void {
        self.losses.deinit();
        self.lrs.deinit();
    }

    pub fn step(self: *LRRangeTest, loss: f64) ?f64 {
        self.current_iter += 1;

        const iter = @as(f64, @floatFromInt(self.current_iter));
        const total = @as(f64, @floatFromInt(self.num_iter));
        const mult = (self.end_lr / self.start_lr);
        const lr = self.start_lr * math.pow(f64, mult, iter / total);

        self.losses.append(loss) catch {};
        self.lrs.append(lr) catch {};

        if (self.losses.items.len > 1) {
            const min_loss = self.findMinLoss();
            if (loss > self.diverge_threshold * min_loss) {
                return null;
            }
        }

        return lr;
    }

    fn findMinLoss(self: *const LRRangeTest) f64 {
        var min_val: f64 = math.inf(f64);
        for (self.losses.items) |loss| {
            if (loss < min_val) {
                min_val = loss;
            }
        }
        return min_val;
    }

    pub fn getSuggestedLr(self: *const LRRangeTest) f64 {
        if (self.losses.items.len < 10) {
            return self.start_lr;
        }

        var min_idx: usize = 0;
        var min_loss: f64 = math.inf(f64);
        
        for (self.losses.items, 0..) |loss, i| {
            if (loss < min_loss) {
                min_loss = loss;
                min_idx = i;
            }
        }

        const suggested_idx = min_idx / 10;
        if (suggested_idx < self.lrs.items.len) {
            return self.lrs.items[suggested_idx];
        }

        return self.lrs.items[0];
    }
};

pub const AdaptiveLRScheduler = struct {
    base_lr: f64,
    current_lr: f64,
    grad_history: std.ArrayList(f64),
    loss_history: std.ArrayList(f64),
    window_size: usize,
    increase_factor: f64,
    decrease_factor: f64,
    min_lr: f64,
    max_lr: f64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, base_lr: f64, min_lr: f64, max_lr: f64) AdaptiveLRScheduler {
        return AdaptiveLRScheduler{
            .base_lr = base_lr,
            .current_lr = base_lr,
            .grad_history = std.ArrayList(f64).init(allocator),
            .loss_history = std.ArrayList(f64).init(allocator),
            .window_size = 10,
            .increase_factor = 1.1,
            .decrease_factor = 0.5,
            .min_lr = min_lr,
            .max_lr = max_lr,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AdaptiveLRScheduler) void {
        self.grad_history.deinit();
        self.loss_history.deinit();
    }

    pub fn step(self: *AdaptiveLRScheduler, loss: f64, grad_norm: f64) f64 {
        self.loss_history.append(loss) catch {};
        self.grad_history.append(grad_norm) catch {};

        if (self.loss_history.items.len < self.window_size) {
            return self.current_lr;
        }

        const recent_losses = self.loss_history.items[self.loss_history.items.len - self.window_size ..];
        var trend: f64 = 0.0;
        for (recent_losses[1..], 0..) |curr_loss, i| {
            trend += curr_loss - recent_losses[i];
        }
        trend /= @as(f64, @floatFromInt(self.window_size - 1));

        if (trend > 0.0) {
            self.current_lr = @max(self.current_lr * self.decrease_factor, self.min_lr);
        } else if (trend < -0.001) {
            self.current_lr = @min(self.current_lr * self.increase_factor, self.max_lr);
        }

        return self.current_lr;
    }

    pub fn getLr(self: *const AdaptiveLRScheduler) f64 {
        return self.current_lr;
    }
};

pub fn createScheduler(allocator: std.mem.Allocator, config: SchedulerConfig) LearningRateScheduler {
    return LearningRateScheduler.init(allocator, config);
}

pub fn createWarmupCosineScheduler(allocator: std.mem.Allocator, base_lr: f64, warmup_steps: u64, total_steps: u64, min_lr: f64) LearningRateScheduler {
    var config = SchedulerConfig.default();
    config.scheduler_type = .warmup_cosine;
    config.base_lr = base_lr;
    config.warmup_steps = warmup_steps;
    config.total_steps = total_steps;
    config.eta_min = min_lr;
    return LearningRateScheduler.init(allocator, config);
}

pub fn createOneCycleScheduler(allocator: std.mem.Allocator, max_lr: f64, total_steps: u64, pct_start: f64, div_factor: f64, final_div_factor: f64) LearningRateScheduler {
    var config = SchedulerConfig.default();
    config.scheduler_type = .one_cycle;
    config.max_lr = max_lr;
    config.total_steps = total_steps;
    config.pct_start = pct_start;
    config.div_factor = div_factor;
    config.final_div_factor = final_div_factor;
    return LearningRateScheduler.init(allocator, config);
}

pub fn createCosineAnnealingScheduler(allocator: std.mem.Allocator, base_lr: f64, t_max: u64, eta_min: f64) LearningRateScheduler {
    var config = SchedulerConfig.default();
    config.scheduler_type = .cosine_annealing;
    config.base_lr = base_lr;
    config.t_max = t_max;
    config.eta_min = eta_min;
    return LearningRateScheduler.init(allocator, config);
}

pub fn createStepScheduler(allocator: std.mem.Allocator, base_lr: f64, step_size: u64, gamma: f64) LearningRateScheduler {
    var config = SchedulerConfig.default();
    config.scheduler_type = .step;
    config.base_lr = base_lr;
    config.step_size = step_size;
    config.gamma = gamma;
    return LearningRateScheduler.init(allocator, config);
}

pub fn createReduceOnPlateauScheduler(allocator: std.mem.Allocator, base_lr: f64, factor: f64, patience: u64, min_lr: f64) LearningRateScheduler {
    var config = SchedulerConfig.default();
    config.scheduler_type = .reduce_on_plateau;
    config.base_lr = base_lr;
    config.factor = factor;
    config.patience = patience;
    config.min_lr = min_lr;
    return LearningRateScheduler.init(allocator, config);
}

test "constant scheduler" {
    const allocator = std.testing.allocator;
    var config = SchedulerConfig.default();
    config.scheduler_type = .constant;
    config.base_lr = 0.01;
    
    var scheduler = LearningRateScheduler.init(allocator, config);
    defer scheduler.deinit();
    
    const lr1 = scheduler.step();
    const lr2 = scheduler.step();
    const lr3 = scheduler.step();
    
    try std.testing.expectApproxEqAbs(@as(f64, 0.01), lr1, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.01), lr2, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.01), lr3, 1e-10);
}

test "step scheduler" {
    const allocator = std.testing.allocator;
    var config = SchedulerConfig.default();
    config.scheduler_type = .step;
    config.base_lr = 0.01;
    config.step_size = 10;
    config.gamma = 0.1;
    
    var scheduler = LearningRateScheduler.init(allocator, config);
    defer scheduler.deinit();
    
    for (0..9) |_| {
        _ = scheduler.step();
    }
    const lr_before = scheduler.step();
    
    const lr_after = scheduler.step();
    
    try std.testing.expect(lr_after < lr_before);
}

test "cosine annealing scheduler" {
    const allocator = std.testing.allocator;
    var config = SchedulerConfig.default();
    config.scheduler_type = .cosine_annealing;
    config.base_lr = 0.01;
    config.t_max = 100;
    config.eta_min = 0.0001;
    
    var scheduler = LearningRateScheduler.init(allocator, config);
    defer scheduler.deinit();
    
    const lr_start = scheduler.step();
    
    for (0..49) |_| {
        _ = scheduler.step();
    }
    const lr_mid = scheduler.getLr();
    
    for (0..50) |_| {
        _ = scheduler.step();
    }
    const lr_end = scheduler.getLr();
    
    try std.testing.expect(lr_start > lr_mid);
    try std.testing.expect(lr_end < lr_mid);
}

test "warmup scheduler" {
    const warmup = WarmupScheduler.init(0.01, 10, .linear);
    _ = warmup;
}

test "momentum scheduler" {
    var momentum = MomentumScheduler.init(0.85, 0.95, 100, 10, true);
    
    const m1 = momentum.step();
    try std.testing.expect(m1 >= 0.85 and m1 <= 0.95);
}
