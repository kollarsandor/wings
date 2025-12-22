import std
import vec
import color

enum OptimizerType:
    sgd_optimizer
    momentum_optimizer
    nesterov_optimizer
    adagrad_optimizer
    rmsprop_optimizer
    adam_optimizer
    adamw_optimizer
    lamb_optimizer
    lars_optimizer
    radam_optimizer
    nadam_optimizer
    adafactor_optimizer

class OptimizerConfig:
    learning_rate: float
    weight_decay: float
    momentum: float
    beta1: float
    beta2: float
    epsilon: float
    dampening: float
    nesterov: bool
    amsgrad: bool
    maximize: bool
    foreach_enabled: bool
    fused: bool
    capturable: bool
    differentiable: bool

    def default() -> OptimizerConfig:
        OptimizerConfig {
            learning_rate: 0.001,
            weight_decay: 0.0,
            momentum: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            dampening: 0.0,
            nesterov: false,
            amsgrad: false,
            maximize: false,
            foreach_enabled: true,
            fused: false,
            capturable: false,
            differentiable: false
        }

    def with_learning_rate(lr: float) -> OptimizerConfig:
        learning_rate = lr
        return this

    def with_weight_decay(wd: float) -> OptimizerConfig:
        weight_decay = wd
        return this

    def with_momentum(m: float) -> OptimizerConfig:
        momentum = m
        return this

    def with_betas(b1: float, b2: float) -> OptimizerConfig:
        beta1 = b1
        beta2 = b2
        return this

class ParameterState:
    step: int
    exp_avg: [float]
    exp_avg_sq: [float]
    max_exp_avg_sq: [float]
    momentum_buffer: [float]
    sum_grad: [float]
    grad_avg: [float]
    grad_sq_sum: [float]

    def new(size: int) -> ParameterState:
        let zeros: [float] = []
        for i in size:
            push(zeros, 0.0)
        ParameterState {
            step: 0,
            exp_avg: copy(zeros),
            exp_avg_sq: copy(zeros),
            max_exp_avg_sq: copy(zeros),
            momentum_buffer: copy(zeros),
            sum_grad: copy(zeros),
            grad_avg: copy(zeros),
            grad_sq_sum: copy(zeros)
        }

    def reset():
        step = 0
        for i in length(exp_avg):
            exp_avg[i] = 0.0
            exp_avg_sq[i] = 0.0
            max_exp_avg_sq[i] = 0.0
            momentum_buffer[i] = 0.0

class ParameterGroup:
    params: [[float]]
    grads: [[float]]
    states: [ParameterState]
    config: OptimizerConfig
    group_id: int

    def new(group_id: int, config: OptimizerConfig) -> ParameterGroup:
        ParameterGroup {
            params: [],
            grads: [],
            states: [],
            config: config,
            group_id: group_id
        }

    def add_parameter(param: [float], grad: [float]):
        push(params, param)
        push(grads, grad)
        push(states, ParameterState.new(length(param)))

    def num_parameters() -> int:
        return length(params)

    def total_elements() -> int:
        var total = 0
        for p in params:
            total += length(p)
        return total

class BaseOptimizer:
    param_groups: [ParameterGroup]
    defaults: OptimizerConfig
    global_step: int
    optimizer_type: OptimizerType

    def new(config: OptimizerConfig, opt_type: OptimizerType) -> BaseOptimizer:
        BaseOptimizer {
            param_groups: [],
            defaults: config,
            global_step: 0,
            optimizer_type: opt_type
        }

    def add_param_group(group: ParameterGroup):
        push(param_groups, group)

    def zero_grad():
        for group in param_groups:
            for grad in group.grads:
                for i in length(grad):
                    grad[i] = 0.0

    def step():
        global_step += 1

    def state_dict() -> [[float]]:
        let states: [[float]] = []
        for group in param_groups:
            for state in group.states:
                push(states, copy(state.exp_avg))
        return states

    def load_state_dict(states: [[float]]):
        var idx = 0
        for group in param_groups:
            for state in group.states:
                if idx < length(states):
                    state.exp_avg = copy(states[idx])
                    idx += 1

class SGDOptimizer:
    base: BaseOptimizer

    def new(config: OptimizerConfig) -> SGDOptimizer:
        SGDOptimizer {
            base: BaseOptimizer.new(config, sgd_optimizer)
        }

    def step():
        base.step()
        
        for group in base.param_groups:
            let lr = group.config.learning_rate
            let wd = group.config.weight_decay
            let momentum = group.config.momentum
            let dampening = group.config.dampening
            let nesterov = group.config.nesterov
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                for i in length(param):
                    var d_p = grad[i]
                    
                    if wd != 0.0:
                        d_p += wd * param[i]
                    
                    if momentum != 0.0:
                        if state.step == 0:
                            state.momentum_buffer[i] = d_p
                        else:
                            state.momentum_buffer[i] = momentum * state.momentum_buffer[i] + (1.0 - dampening) * d_p
                        
                        if nesterov:
                            d_p = d_p + momentum * state.momentum_buffer[i]
                        else:
                            d_p = state.momentum_buffer[i]
                    
                    param[i] -= lr * d_p
                
                state.step += 1

    def zero_grad():
        base.zero_grad()

class AdamOptimizer:
    base: BaseOptimizer

    def new(config: OptimizerConfig) -> AdamOptimizer:
        AdamOptimizer {
            base: BaseOptimizer.new(config, adam_optimizer)
        }

    def step():
        base.step()
        
        for group in base.param_groups:
            let lr = group.config.learning_rate
            let beta1 = group.config.beta1
            let beta2 = group.config.beta2
            let eps = group.config.epsilon
            let wd = group.config.weight_decay
            let amsgrad = group.config.amsgrad
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                state.step += 1
                
                let bias_correction1 = 1.0 - pow(beta1, float(state.step))
                let bias_correction2 = 1.0 - pow(beta2, float(state.step))
                
                for i in length(param):
                    var g = grad[i]
                    
                    if wd != 0.0:
                        g += wd * param[i]
                    
                    state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g
                    state.exp_avg_sq[i] = beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * g * g
                    
                    var denom: float
                    if amsgrad:
                        state.max_exp_avg_sq[i] = max(state.max_exp_avg_sq[i], state.exp_avg_sq[i])
                        denom = sqrt(state.max_exp_avg_sq[i] / bias_correction2) + eps
                    else:
                        denom = sqrt(state.exp_avg_sq[i] / bias_correction2) + eps
                    
                    let step_size = lr / bias_correction1
                    param[i] -= step_size * state.exp_avg[i] / denom

    def zero_grad():
        base.zero_grad()

class AdamWOptimizer:
    base: BaseOptimizer

    def new(config: OptimizerConfig) -> AdamWOptimizer:
        AdamWOptimizer {
            base: BaseOptimizer.new(config, adamw_optimizer)
        }

    def step():
        base.step()
        
        for group in base.param_groups:
            let lr = group.config.learning_rate
            let beta1 = group.config.beta1
            let beta2 = group.config.beta2
            let eps = group.config.epsilon
            let wd = group.config.weight_decay
            let amsgrad = group.config.amsgrad
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                state.step += 1
                
                for i in length(param):
                    param[i] -= lr * wd * param[i]
                
                let bias_correction1 = 1.0 - pow(beta1, float(state.step))
                let bias_correction2 = 1.0 - pow(beta2, float(state.step))
                
                for i in length(param):
                    let g = grad[i]
                    
                    state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g
                    state.exp_avg_sq[i] = beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * g * g
                    
                    var denom: float
                    if amsgrad:
                        state.max_exp_avg_sq[i] = max(state.max_exp_avg_sq[i], state.exp_avg_sq[i])
                        denom = sqrt(state.max_exp_avg_sq[i] / bias_correction2) + eps
                    else:
                        denom = sqrt(state.exp_avg_sq[i] / bias_correction2) + eps
                    
                    let step_size = lr / bias_correction1
                    param[i] -= step_size * state.exp_avg[i] / denom

    def zero_grad():
        base.zero_grad()

class RMSpropOptimizer:
    base: BaseOptimizer
    alpha: float
    centered: bool

    def new(config: OptimizerConfig) -> RMSpropOptimizer:
        RMSpropOptimizer {
            base: BaseOptimizer.new(config, rmsprop_optimizer),
            alpha: 0.99,
            centered: false
        }

    def step():
        base.step()
        
        for group in base.param_groups:
            let lr = group.config.learning_rate
            let eps = group.config.epsilon
            let wd = group.config.weight_decay
            let momentum = group.config.momentum
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                state.step += 1
                
                for i in length(param):
                    var g = grad[i]
                    
                    if wd != 0.0:
                        g += wd * param[i]
                    
                    state.exp_avg_sq[i] = alpha * state.exp_avg_sq[i] + (1.0 - alpha) * g * g
                    
                    var avg: float
                    if centered:
                        state.grad_avg[i] = alpha * state.grad_avg[i] + (1.0 - alpha) * g
                        avg = state.exp_avg_sq[i] - state.grad_avg[i] * state.grad_avg[i]
                    else:
                        avg = state.exp_avg_sq[i]
                    
                    if momentum > 0.0:
                        state.momentum_buffer[i] = momentum * state.momentum_buffer[i] + g / (sqrt(avg) + eps)
                        param[i] -= lr * state.momentum_buffer[i]
                    else:
                        param[i] -= lr * g / (sqrt(avg) + eps)

    def zero_grad():
        base.zero_grad()

class AdagradOptimizer:
    base: BaseOptimizer
    lr_decay: float
    initial_accumulator_value: float

    def new(config: OptimizerConfig) -> AdagradOptimizer:
        AdagradOptimizer {
            base: BaseOptimizer.new(config, adagrad_optimizer),
            lr_decay: 0.0,
            initial_accumulator_value: 0.0
        }

    def step():
        base.step()
        
        for group in base.param_groups:
            let lr = group.config.learning_rate
            let eps = group.config.epsilon
            let wd = group.config.weight_decay
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                if state.step == 0:
                    for i in length(param):
                        state.sum_grad[i] = initial_accumulator_value
                
                state.step += 1
                
                let adjusted_lr = lr / (1.0 + float(state.step - 1) * lr_decay)
                
                for i in length(param):
                    var g = grad[i]
                    
                    if wd != 0.0:
                        g += wd * param[i]
                    
                    state.sum_grad[i] += g * g
                    param[i] -= adjusted_lr * g / (sqrt(state.sum_grad[i]) + eps)

    def zero_grad():
        base.zero_grad()

class LAMBOptimizer:
    base: BaseOptimizer
    trust_clip: bool

    def new(config: OptimizerConfig) -> LAMBOptimizer:
        LAMBOptimizer {
            base: BaseOptimizer.new(config, lamb_optimizer),
            trust_clip: true
        }

    def step():
        base.step()
        
        for group in base.param_groups:
            let lr = group.config.learning_rate
            let beta1 = group.config.beta1
            let beta2 = group.config.beta2
            let eps = group.config.epsilon
            let wd = group.config.weight_decay
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                state.step += 1
                
                let bias_correction1 = 1.0 - pow(beta1, float(state.step))
                let bias_correction2 = 1.0 - pow(beta2, float(state.step))
                
                var weight_norm = 0.0
                for i in length(param):
                    weight_norm += param[i] * param[i]
                weight_norm = sqrt(weight_norm)
                
                var adam_step: [float] = []
                var adam_norm = 0.0
                
                for i in length(param):
                    let g = grad[i]
                    
                    state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g
                    state.exp_avg_sq[i] = beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * g * g
                    
                    let m_hat = state.exp_avg[i] / bias_correction1
                    let v_hat = state.exp_avg_sq[i] / bias_correction2
                    
                    var step_val = m_hat / (sqrt(v_hat) + eps)
                    if wd > 0.0:
                        step_val += wd * param[i]
                    
                    push(adam_step, step_val)
                    adam_norm += step_val * step_val
                
                adam_norm = sqrt(adam_norm)
                
                var trust_ratio = 1.0
                if weight_norm > 0.0 and adam_norm > 0.0:
                    trust_ratio = weight_norm / adam_norm
                    if trust_clip:
                        trust_ratio = min(trust_ratio, 10.0)
                
                for i in length(param):
                    param[i] -= lr * trust_ratio * adam_step[i]

    def zero_grad():
        base.zero_grad()

class LARSOptimizer:
    base: BaseOptimizer
    trust_coefficient: float
    trust_clip: bool
    eps: float

    def new(config: OptimizerConfig) -> LARSOptimizer:
        LARSOptimizer {
            base: BaseOptimizer.new(config, lars_optimizer),
            trust_coefficient: 0.001,
            trust_clip: true,
            eps: 1e-8
        }

    def step():
        base.step()
        
        for group in base.param_groups:
            let lr = group.config.learning_rate
            let wd = group.config.weight_decay
            let momentum = group.config.momentum
            let dampening = group.config.dampening
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                state.step += 1
                
                var weight_norm = 0.0
                var grad_norm = 0.0
                for i in length(param):
                    weight_norm += param[i] * param[i]
                    grad_norm += grad[i] * grad[i]
                weight_norm = sqrt(weight_norm)
                grad_norm = sqrt(grad_norm)
                
                var local_lr = 1.0
                if weight_norm > eps and grad_norm > eps:
                    local_lr = trust_coefficient * weight_norm / (grad_norm + wd * weight_norm + eps)
                    if trust_clip:
                        local_lr = min(local_lr, 1.0)
                
                for i in length(param):
                    var d_p = grad[i]
                    
                    if wd != 0.0:
                        d_p += wd * param[i]
                    
                    d_p *= local_lr
                    
                    if momentum != 0.0:
                        if state.step == 1:
                            state.momentum_buffer[i] = d_p
                        else:
                            state.momentum_buffer[i] = momentum * state.momentum_buffer[i] + (1.0 - dampening) * d_p
                        d_p = state.momentum_buffer[i]
                    
                    param[i] -= lr * d_p

    def zero_grad():
        base.zero_grad()

class RAdamOptimizer:
    base: BaseOptimizer
    rho_inf: float

    def new(config: OptimizerConfig) -> RAdamOptimizer:
        RAdamOptimizer {
            base: BaseOptimizer.new(config, radam_optimizer),
            rho_inf: 0.0
        }

    def compute_rho_inf(beta2: float) -> float:
        return 2.0 / (1.0 - beta2) - 1.0

    def step():
        base.step()
        
        for group in base.param_groups:
            let lr = group.config.learning_rate
            let beta1 = group.config.beta1
            let beta2 = group.config.beta2
            let eps = group.config.epsilon
            let wd = group.config.weight_decay
            
            let rho_inf_val = compute_rho_inf(beta2)
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                state.step += 1
                
                let bias_correction1 = 1.0 - pow(beta1, float(state.step))
                let bias_correction2 = 1.0 - pow(beta2, float(state.step))
                
                let beta2_t = pow(beta2, float(state.step))
                let rho_t = rho_inf_val - 2.0 * float(state.step) * beta2_t / bias_correction2
                
                for i in length(param):
                    var g = grad[i]
                    
                    if wd != 0.0:
                        g += wd * param[i]
                    
                    state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g
                    state.exp_avg_sq[i] = beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * g * g
                    
                    let m_hat = state.exp_avg[i] / bias_correction1
                    
                    if rho_t > 5.0:
                        let v_hat = state.exp_avg_sq[i] / bias_correction2
                        let rect = sqrt((rho_t - 4.0) * (rho_t - 2.0) * rho_inf_val / ((rho_inf_val - 4.0) * (rho_inf_val - 2.0) * rho_t))
                        param[i] -= lr * rect * m_hat / (sqrt(v_hat) + eps)
                    else:
                        param[i] -= lr * m_hat

    def zero_grad():
        base.zero_grad()

class NAdamOptimizer:
    base: BaseOptimizer

    def new(config: OptimizerConfig) -> NAdamOptimizer:
        NAdamOptimizer {
            base: BaseOptimizer.new(config, nadam_optimizer)
        }

    def step():
        base.step()
        
        for group in base.param_groups:
            let lr = group.config.learning_rate
            let beta1 = group.config.beta1
            let beta2 = group.config.beta2
            let eps = group.config.epsilon
            let wd = group.config.weight_decay
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                state.step += 1
                
                let bias_correction1 = 1.0 - pow(beta1, float(state.step))
                let bias_correction2 = 1.0 - pow(beta2, float(state.step))
                let bias_correction1_next = 1.0 - pow(beta1, float(state.step + 1))
                
                for i in length(param):
                    var g = grad[i]
                    
                    if wd != 0.0:
                        param[i] -= lr * wd * param[i]
                    
                    state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g
                    state.exp_avg_sq[i] = beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * g * g
                    
                    let m_hat = state.exp_avg[i] / bias_correction1
                    let m_hat_next = state.exp_avg[i] / bias_correction1_next
                    let v_hat = state.exp_avg_sq[i] / bias_correction2
                    
                    let m_nadam = (1.0 - beta1) * g / bias_correction1 + beta1 * m_hat_next
                    
                    param[i] -= lr * m_nadam / (sqrt(v_hat) + eps)

    def zero_grad():
        base.zero_grad()

class AdafactorOptimizer:
    base: BaseOptimizer
    scale_parameter: bool
    relative_step: bool
    warmup_init: bool
    clip_threshold: float
    decay_rate: float
    beta1: float
    rms_eps: float

    def new(config: OptimizerConfig) -> AdafactorOptimizer:
        AdafactorOptimizer {
            base: BaseOptimizer.new(config, adafactor_optimizer),
            scale_parameter: true,
            relative_step: true,
            warmup_init: false,
            clip_threshold: 1.0,
            decay_rate: -0.8,
            beta1: 0.0,
            rms_eps: 1e-30
        }

    def compute_rms(tensor: [float]) -> float:
        var sum_sq = 0.0
        for v in tensor:
            sum_sq += v * v
        return sqrt(sum_sq / float(length(tensor)))

    def get_lr(base_lr: float, step: int) -> float:
        if relative_step:
            var min_step = 1e-6
            if warmup_init:
                min_step = 1e-2 * float(step)
            return min(base_lr, 1.0 / sqrt(float(step)))
        return base_lr

    def get_rho(step: int) -> float:
        return min(0.3, 1.0 / pow(float(step), 1.0 - decay_rate))

    def step():
        base.step()
        
        for group in base.param_groups:
            let base_lr = group.config.learning_rate
            
            for p_idx in length(group.params):
                let param = group.params[p_idx]
                let grad = group.grads[p_idx]
                var state = group.states[p_idx]
                
                state.step += 1
                
                let lr = get_lr(base_lr, state.step)
                let rho = get_rho(state.step)
                
                var alpha = lr
                if scale_parameter:
                    let rms = compute_rms(param)
                    alpha = lr * max(1e-3, rms)
                
                for i in length(param):
                    let g = grad[i]
                    
                    state.exp_avg_sq[i] = rho * state.exp_avg_sq[i] + (1.0 - rho) * g * g
                    
                    let v_hat = state.exp_avg_sq[i]
                    let update_scale = 1.0 / (sqrt(v_hat) + rms_eps)
                    
                    var update = g * update_scale
                    
                    let rms_update = compute_rms([update])
                    if rms_update > clip_threshold:
                        update = update * clip_threshold / rms_update
                    
                    if beta1 > 0.0:
                        state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * update
                        update = state.exp_avg[i]
                    
                    param[i] -= alpha * update

    def zero_grad():
        base.zero_grad()

class LRScheduler:
    base_lr: float
    current_lr: float
    step_count: int
    last_epoch: int

    def new(base_lr: float) -> LRScheduler:
        LRScheduler {
            base_lr: base_lr,
            current_lr: base_lr,
            step_count: 0,
            last_epoch: -1
        }

    def get_lr() -> float:
        return current_lr

    def step():
        step_count += 1
        last_epoch += 1

class StepLR:
    scheduler: LRScheduler
    step_size: int
    gamma: float

    def new(base_lr: float, step_size: int, gamma: float) -> StepLR:
        StepLR {
            scheduler: LRScheduler.new(base_lr),
            step_size: step_size,
            gamma: gamma
        }

    def step():
        scheduler.step()
        let epoch = scheduler.last_epoch
        let num_decays = epoch / step_size
        scheduler.current_lr = scheduler.base_lr * pow(gamma, float(num_decays))

    def get_lr() -> float:
        return scheduler.get_lr()

class ExponentialLR:
    scheduler: LRScheduler
    gamma: float

    def new(base_lr: float, gamma: float) -> ExponentialLR:
        ExponentialLR {
            scheduler: LRScheduler.new(base_lr),
            gamma: gamma
        }

    def step():
        scheduler.step()
        scheduler.current_lr = scheduler.base_lr * pow(gamma, float(scheduler.last_epoch))

    def get_lr() -> float:
        return scheduler.get_lr()

class CosineAnnealingLR:
    scheduler: LRScheduler
    t_max: int
    eta_min: float

    def new(base_lr: float, t_max: int, eta_min: float) -> CosineAnnealingLR:
        CosineAnnealingLR {
            scheduler: LRScheduler.new(base_lr),
            t_max: t_max,
            eta_min: eta_min
        }

    def step():
        scheduler.step()
        let epoch = scheduler.last_epoch
        let cos_inner = 3.14159265359 * float(epoch) / float(t_max)
        scheduler.current_lr = eta_min + (scheduler.base_lr - eta_min) * (1.0 + cos(cos_inner)) / 2.0

    def get_lr() -> float:
        return scheduler.get_lr()

class WarmupLR:
    scheduler: LRScheduler
    warmup_steps: int
    target_lr: float

    def new(target_lr: float, warmup_steps: int) -> WarmupLR:
        WarmupLR {
            scheduler: LRScheduler.new(0.0),
            warmup_steps: warmup_steps,
            target_lr: target_lr
        }

    def step():
        scheduler.step()
        let step = scheduler.step_count
        if step <= warmup_steps:
            scheduler.current_lr = target_lr * float(step) / float(warmup_steps)
        else:
            scheduler.current_lr = target_lr

    def get_lr() -> float:
        return scheduler.get_lr()

class WarmupCosineAnnealingLR:
    scheduler: LRScheduler
    warmup_steps: int
    total_steps: int
    eta_min: float

    def new(base_lr: float, warmup_steps: int, total_steps: int, eta_min: float) -> WarmupCosineAnnealingLR:
        WarmupCosineAnnealingLR {
            scheduler: LRScheduler.new(base_lr),
            warmup_steps: warmup_steps,
            total_steps: total_steps,
            eta_min: eta_min
        }

    def step():
        scheduler.step()
        let step = scheduler.step_count
        
        if step <= warmup_steps:
            scheduler.current_lr = scheduler.base_lr * float(step) / float(warmup_steps)
        else:
            let progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
            let cos_inner = 3.14159265359 * progress
            scheduler.current_lr = eta_min + (scheduler.base_lr - eta_min) * (1.0 + cos(cos_inner)) / 2.0

    def get_lr() -> float:
        return scheduler.get_lr()

class PolynomialLR:
    scheduler: LRScheduler
    total_iters: int
    power: float
    end_lr: float

    def new(base_lr: float, total_iters: int, power: float, end_lr: float) -> PolynomialLR:
        PolynomialLR {
            scheduler: LRScheduler.new(base_lr),
            total_iters: total_iters,
            power: power,
            end_lr: end_lr
        }

    def step():
        scheduler.step()
        let step = min(scheduler.step_count, total_iters)
        let decay = pow(1.0 - float(step) / float(total_iters), power)
        scheduler.current_lr = (scheduler.base_lr - end_lr) * decay + end_lr

    def get_lr() -> float:
        return scheduler.get_lr()

class OneCycleLR:
    scheduler: LRScheduler
    max_lr: float
    total_steps: int
    pct_start: float
    div_factor: float
    final_div_factor: float
    anneal_strategy: string

    def new(max_lr: float, total_steps: int) -> OneCycleLR:
        OneCycleLR {
            scheduler: LRScheduler.new(max_lr / 25.0),
            max_lr: max_lr,
            total_steps: total_steps,
            pct_start: 0.3,
            div_factor: 25.0,
            final_div_factor: 10000.0,
            anneal_strategy: "cos"
        }

    def step():
        scheduler.step()
        let step = scheduler.step_count
        let initial_lr = max_lr / div_factor
        let final_lr = initial_lr / final_div_factor
        
        let step1 = int(float(total_steps) * pct_start)
        let step2 = total_steps - step1
        
        if step <= step1:
            let pct = float(step) / float(step1)
            scheduler.current_lr = initial_lr + (max_lr - initial_lr) * pct
        else:
            let pct = float(step - step1) / float(step2)
            if anneal_strategy == "cos":
                let cos_inner = 3.14159265359 * pct
                scheduler.current_lr = final_lr + (max_lr - final_lr) * (1.0 + cos(cos_inner)) / 2.0
            else:
                scheduler.current_lr = max_lr - (max_lr - final_lr) * pct

    def get_lr() -> float:
        return scheduler.get_lr()

class ReduceLROnPlateau:
    scheduler: LRScheduler
    mode: string
    factor: float
    patience: int
    threshold: float
    threshold_mode: string
    cooldown: int
    min_lr: float
    best: float
    num_bad_epochs: int
    cooldown_counter: int

    def new(base_lr: float, mode: string, factor: float, patience: int) -> ReduceLROnPlateau:
        let initial_best = if mode == "min": 1e38 else: -1e38
        ReduceLROnPlateau {
            scheduler: LRScheduler.new(base_lr),
            mode: mode,
            factor: factor,
            patience: patience,
            threshold: 1e-4,
            threshold_mode: "rel",
            cooldown: 0,
            min_lr: 0.0,
            best: initial_best,
            num_bad_epochs: 0,
            cooldown_counter: 0
        }

    def is_better(current: float) -> bool:
        if mode == "min":
            if threshold_mode == "rel":
                return current < best * (1.0 - threshold)
            else:
                return current < best - threshold
        else:
            if threshold_mode == "rel":
                return current > best * (1.0 + threshold)
            else:
                return current > best + threshold

    def step_with_metric(metric: float):
        scheduler.step()
        
        if cooldown_counter > 0:
            cooldown_counter -= 1
            num_bad_epochs = 0
            return
        
        if is_better(metric):
            best = metric
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
        
        if num_bad_epochs > patience:
            scheduler.current_lr = max(scheduler.current_lr * factor, min_lr)
            cooldown_counter = cooldown
            num_bad_epochs = 0

    def get_lr() -> float:
        return scheduler.get_lr()

def create_optimizer(opt_type: OptimizerType, config: OptimizerConfig):
    switch opt_type:
        case sgd_optimizer:
            return SGDOptimizer.new(config)
        case adam_optimizer:
            return AdamOptimizer.new(config)
        case adamw_optimizer:
            return AdamWOptimizer.new(config)
        case rmsprop_optimizer:
            return RMSpropOptimizer.new(config)
        case adagrad_optimizer:
            return AdagradOptimizer.new(config)
        case lamb_optimizer:
            return LAMBOptimizer.new(config)
        case lars_optimizer:
            return LARSOptimizer.new(config)
        case radam_optimizer:
            return RAdamOptimizer.new(config)
        case nadam_optimizer:
            return NAdamOptimizer.new(config)
        case adafactor_optimizer:
            return AdafactorOptimizer.new(config)
        default:
            return AdamOptimizer.new(config)

def run_optimizer_test():
    let config = OptimizerConfig.default()
    let optimizer = AdamOptimizer.new(config)
    
    let params = [1.0, 2.0, 3.0, 4.0]
    let grads = [0.1, 0.2, 0.3, 0.4]
    
    var group = ParameterGroup.new(0, config)
    group.add_parameter(params, grads)
    optimizer.base.add_param_group(group)
    
    for i in 100:
        optimizer.step()
    
    print("Optimizer test completed")
    print("Final params: " + string(params))

def main():
    run_optimizer_test()
