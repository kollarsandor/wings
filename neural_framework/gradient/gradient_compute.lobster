import std
import vec
import color
import gl

enum GradientMode:
    forward_mode
    reverse_mode
    mixed_mode

enum OperationType:
    add_op
    sub_op
    mul_op
    div_op
    matmul_op
    relu_op
    sigmoid_op
    tanh_op
    softmax_op
    exp_op
    log_op
    pow_op
    sqrt_op
    sum_op
    mean_op
    max_op
    min_op
    reshape_op
    transpose_op
    concat_op
    slice_op
    conv2d_op
    maxpool_op
    batchnorm_op
    layernorm_op
    dropout_op
    embedding_op
    attention_op

class TensorShape:
    dims: [int]
    ndim: int
    size: int

    def new(dimensions: [int]) -> TensorShape:
        var total = 1
        for d in dimensions:
            total *= d
        TensorShape { dims: dimensions, ndim: length(dimensions), size: total }

    def equals(other: TensorShape) -> bool:
        if ndim != other.ndim:
            return false
        for i in ndim:
            if dims[i] != other.dims[i]:
                return false
        return true

    def broadcast_compatible(other: TensorShape) -> bool:
        let max_ndim = max(ndim, other.ndim)
        for i in max_ndim:
            let dim_a = if i < ndim: dims[ndim - 1 - i] else: 1
            let dim_b = if i < other.ndim: other.dims[other.ndim - 1 - i] else: 1
            if dim_a != dim_b and dim_a != 1 and dim_b != 1:
                return false
        return true

class DualNumber:
    value: float
    tangent: float

    def new(v: float, t: float) -> DualNumber:
        DualNumber { value: v, tangent: t }

    def add(other: DualNumber) -> DualNumber:
        DualNumber { value: value + other.value, tangent: tangent + other.tangent }

    def sub(other: DualNumber) -> DualNumber:
        DualNumber { value: value - other.value, tangent: tangent - other.tangent }

    def mul(other: DualNumber) -> DualNumber:
        let new_val = value * other.value
        let new_tan = value * other.tangent + tangent * other.value
        DualNumber { value: new_val, tangent: new_tan }

    def div(other: DualNumber) -> DualNumber:
        let new_val = value / other.value
        let new_tan = (tangent * other.value - value * other.tangent) / (other.value * other.value)
        DualNumber { value: new_val, tangent: new_tan }

    def exp_dual() -> DualNumber:
        let e = exp(value)
        DualNumber { value: e, tangent: e * tangent }

    def log_dual() -> DualNumber:
        DualNumber { value: log(value), tangent: tangent / value }

    def sqrt_dual() -> DualNumber:
        let s = sqrt(value)
        DualNumber { value: s, tangent: tangent / (2.0 * s) }

    def pow_dual(n: float) -> DualNumber:
        let p = pow(value, n)
        DualNumber { value: p, tangent: n * pow(value, n - 1.0) * tangent }

    def sin_dual() -> DualNumber:
        DualNumber { value: sin(value), tangent: cos(value) * tangent }

    def cos_dual() -> DualNumber:
        DualNumber { value: cos(value), tangent: -sin(value) * tangent }

    def tanh_dual() -> DualNumber:
        let t = tanh(value)
        DualNumber { value: t, tangent: (1.0 - t * t) * tangent }

class GradientTape:
    operations: [TapeEntry]
    variable_count: int
    is_recording: bool
    persistent: bool

    def new() -> GradientTape:
        GradientTape {
            operations: [],
            variable_count: 0,
            is_recording: true,
            persistent: false
        }

    def start_recording():
        is_recording = true

    def stop_recording():
        is_recording = false

    def record(entry: TapeEntry):
        if is_recording:
            push(operations, entry)

    def clear():
        operations = []
        variable_count = 0

    def num_operations() -> int:
        return length(operations)

class TapeEntry:
    op_type: OperationType
    input_ids: [int]
    output_id: int
    input_shapes: [TensorShape]
    output_shape: TensorShape
    cached_values: [[float]]
    extra_params: [float]

    def new(op: OperationType, inputs: [int], output: int, in_shapes: [TensorShape], out_shape: TensorShape) -> TapeEntry:
        TapeEntry {
            op_type: op,
            input_ids: inputs,
            output_id: output,
            input_shapes: in_shapes,
            output_shape: out_shape,
            cached_values: [],
            extra_params: []
        }

class Variable:
    id: int
    data: [float]
    grad: [float]
    shape: TensorShape
    requires_grad: bool
    is_leaf: bool
    name: string

    def new(tensor_data: [float], tensor_shape: TensorShape, track_grad: bool, var_name: string, var_id: int) -> Variable:
        let grad_data = map(tensor_data): 0.0
        Variable {
            id: var_id,
            data: tensor_data,
            grad: grad_data,
            shape: tensor_shape,
            requires_grad: track_grad,
            is_leaf: true,
            name: var_name
        }

    def zero_grad():
        for i in length(grad):
            grad[i] = 0.0

    def detach() -> Variable:
        Variable {
            id: id,
            data: copy(data),
            grad: map(data): 0.0,
            shape: shape,
            requires_grad: false,
            is_leaf: true,
            name: name + "_detached"
        }

class ComputationGraph:
    tape: GradientTape
    variables: [Variable]
    next_id: int
    gradient_checkpoints: [[int]]
    memory_efficient: bool

    def new() -> ComputationGraph:
        ComputationGraph {
            tape: GradientTape.new(),
            variables: [],
            next_id: 0,
            gradient_checkpoints: [],
            memory_efficient: false
        }

    def create_variable(data: [float], shape: TensorShape, requires_grad: bool, name: string) -> Variable:
        let var = Variable.new(data, shape, requires_grad, name, next_id)
        next_id += 1
        push(variables, var)
        return var

    def get_variable(id: int) -> Variable:
        for v in variables:
            if v.id == id:
                return v
        return variables[0]

    def add_checkpoint(var_ids: [int]):
        push(gradient_checkpoints, var_ids)

class GradientComputer:
    graph: ComputationGraph
    mode: GradientMode
    gradient_accumulation: bool
    retain_graph: bool

    def new() -> GradientComputer:
        GradientComputer {
            graph: ComputationGraph.new(),
            mode: reverse_mode,
            gradient_accumulation: false,
            retain_graph: false
        }

    def forward_add(a: Variable, b: Variable) -> Variable:
        let result_data: [float] = []
        for i in length(a.data):
            push(result_data, a.data[i] + b.data[i])
        
        let result = graph.create_variable(result_data, a.shape, a.requires_grad or b.requires_grad, "add_result")
        result.is_leaf = false

        let entry = TapeEntry.new(add_op, [a.id, b.id], result.id, [a.shape, b.shape], result.shape)
        graph.tape.record(entry)

        return result

    def forward_sub(a: Variable, b: Variable) -> Variable:
        let result_data: [float] = []
        for i in length(a.data):
            push(result_data, a.data[i] - b.data[i])
        
        let result = graph.create_variable(result_data, a.shape, a.requires_grad or b.requires_grad, "sub_result")
        result.is_leaf = false

        let entry = TapeEntry.new(sub_op, [a.id, b.id], result.id, [a.shape, b.shape], result.shape)
        graph.tape.record(entry)

        return result

    def forward_mul(a: Variable, b: Variable) -> Variable:
        let result_data: [float] = []
        for i in length(a.data):
            push(result_data, a.data[i] * b.data[i])
        
        let result = graph.create_variable(result_data, a.shape, a.requires_grad or b.requires_grad, "mul_result")
        result.is_leaf = false

        var entry = TapeEntry.new(mul_op, [a.id, b.id], result.id, [a.shape, b.shape], result.shape)
        entry.cached_values = [copy(a.data), copy(b.data)]
        graph.tape.record(entry)

        return result

    def forward_div(a: Variable, b: Variable) -> Variable:
        let result_data: [float] = []
        for i in length(a.data):
            if abs(b.data[i]) > 1e-10:
                push(result_data, a.data[i] / b.data[i])
            else:
                push(result_data, 0.0)
        
        let result = graph.create_variable(result_data, a.shape, a.requires_grad or b.requires_grad, "div_result")
        result.is_leaf = false

        var entry = TapeEntry.new(div_op, [a.id, b.id], result.id, [a.shape, b.shape], result.shape)
        entry.cached_values = [copy(a.data), copy(b.data)]
        graph.tape.record(entry)

        return result

    def forward_matmul(a: Variable, b: Variable) -> Variable:
        let m = a.shape.dims[0]
        let k = a.shape.dims[1]
        let n = b.shape.dims[1]

        let result_data: [float] = []
        for i in m:
            for j in n:
                var sum = 0.0
                for l in k:
                    sum += a.data[i * k + l] * b.data[l * n + j]
                push(result_data, sum)

        let result_shape = TensorShape.new([m, n])
        let result = graph.create_variable(result_data, result_shape, a.requires_grad or b.requires_grad, "matmul_result")
        result.is_leaf = false

        var entry = TapeEntry.new(matmul_op, [a.id, b.id], result.id, [a.shape, b.shape], result.shape)
        entry.cached_values = [copy(a.data), copy(b.data)]
        graph.tape.record(entry)

        return result

    def forward_relu(x: Variable) -> Variable:
        let result_data: [float] = []
        for i in length(x.data):
            push(result_data, max(0.0, x.data[i]))
        
        let result = graph.create_variable(result_data, x.shape, x.requires_grad, "relu_result")
        result.is_leaf = false

        var entry = TapeEntry.new(relu_op, [x.id], result.id, [x.shape], result.shape)
        entry.cached_values = [copy(x.data)]
        graph.tape.record(entry)

        return result

    def forward_sigmoid(x: Variable) -> Variable:
        let result_data: [float] = []
        for i in length(x.data):
            let sig = 1.0 / (1.0 + exp(-x.data[i]))
            push(result_data, sig)
        
        let result = graph.create_variable(result_data, x.shape, x.requires_grad, "sigmoid_result")
        result.is_leaf = false

        var entry = TapeEntry.new(sigmoid_op, [x.id], result.id, [x.shape], result.shape)
        entry.cached_values = [copy(result_data)]
        graph.tape.record(entry)

        return result

    def forward_tanh_op(x: Variable) -> Variable:
        let result_data: [float] = []
        for i in length(x.data):
            push(result_data, tanh(x.data[i]))
        
        let result = graph.create_variable(result_data, x.shape, x.requires_grad, "tanh_result")
        result.is_leaf = false

        var entry = TapeEntry.new(tanh_op, [x.id], result.id, [x.shape], result.shape)
        entry.cached_values = [copy(result_data)]
        graph.tape.record(entry)

        return result

    def forward_softmax(x: Variable, axis: int) -> Variable:
        let batch_size = x.shape.dims[0]
        let num_classes = x.shape.dims[1]
        
        let result_data: [float] = []
        
        for b in batch_size:
            var max_val = x.data[b * num_classes]
            for c in num_classes:
                max_val = max(max_val, x.data[b * num_classes + c])
            
            var sum_exp = 0.0
            for c in num_classes:
                sum_exp += exp(x.data[b * num_classes + c] - max_val)
            
            for c in num_classes:
                let softmax_val = exp(x.data[b * num_classes + c] - max_val) / sum_exp
                push(result_data, softmax_val)
        
        let result = graph.create_variable(result_data, x.shape, x.requires_grad, "softmax_result")
        result.is_leaf = false

        var entry = TapeEntry.new(softmax_op, [x.id], result.id, [x.shape], result.shape)
        entry.cached_values = [copy(result_data)]
        entry.extra_params = [float(axis)]
        graph.tape.record(entry)

        return result

    def forward_exp(x: Variable) -> Variable:
        let result_data: [float] = []
        for i in length(x.data):
            push(result_data, exp(x.data[i]))
        
        let result = graph.create_variable(result_data, x.shape, x.requires_grad, "exp_result")
        result.is_leaf = false

        var entry = TapeEntry.new(exp_op, [x.id], result.id, [x.shape], result.shape)
        entry.cached_values = [copy(result_data)]
        graph.tape.record(entry)

        return result

    def forward_log(x: Variable) -> Variable:
        let result_data: [float] = []
        for i in length(x.data):
            if x.data[i] > 0.0:
                push(result_data, log(x.data[i]))
            else:
                push(result_data, -1000.0)
        
        let result = graph.create_variable(result_data, x.shape, x.requires_grad, "log_result")
        result.is_leaf = false

        var entry = TapeEntry.new(log_op, [x.id], result.id, [x.shape], result.shape)
        entry.cached_values = [copy(x.data)]
        graph.tape.record(entry)

        return result

    def forward_sum(x: Variable, axis: int) -> Variable:
        var result_data: [float] = []
        var result_shape: TensorShape
        
        if axis < 0:
            var total = 0.0
            for v in x.data:
                total += v
            push(result_data, total)
            result_shape = TensorShape.new([1])
        else:
            let outer_size = x.shape.dims[0]
            let inner_size = x.shape.dims[1]
            
            if axis == 0:
                for j in inner_size:
                    var sum = 0.0
                    for i in outer_size:
                        sum += x.data[i * inner_size + j]
                    push(result_data, sum)
                result_shape = TensorShape.new([inner_size])
            else:
                for i in outer_size:
                    var sum = 0.0
                    for j in inner_size:
                        sum += x.data[i * inner_size + j]
                    push(result_data, sum)
                result_shape = TensorShape.new([outer_size])
        
        let result = graph.create_variable(result_data, result_shape, x.requires_grad, "sum_result")
        result.is_leaf = false

        var entry = TapeEntry.new(sum_op, [x.id], result.id, [x.shape], result.shape)
        entry.extra_params = [float(axis)]
        graph.tape.record(entry)

        return result

    def forward_mean(x: Variable, axis: int) -> Variable:
        var result_data: [float] = []
        var result_shape: TensorShape
        
        if axis < 0:
            var total = 0.0
            for v in x.data:
                total += v
            push(result_data, total / float(length(x.data)))
            result_shape = TensorShape.new([1])
        else:
            let outer_size = x.shape.dims[0]
            let inner_size = x.shape.dims[1]
            
            if axis == 0:
                for j in inner_size:
                    var sum = 0.0
                    for i in outer_size:
                        sum += x.data[i * inner_size + j]
                    push(result_data, sum / float(outer_size))
                result_shape = TensorShape.new([inner_size])
            else:
                for i in outer_size:
                    var sum = 0.0
                    for j in inner_size:
                        sum += x.data[i * inner_size + j]
                    push(result_data, sum / float(inner_size))
                result_shape = TensorShape.new([outer_size])
        
        let result = graph.create_variable(result_data, result_shape, x.requires_grad, "mean_result")
        result.is_leaf = false

        var entry = TapeEntry.new(mean_op, [x.id], result.id, [x.shape], result.shape)
        entry.extra_params = [float(axis), float(x.shape.size)]
        graph.tape.record(entry)

        return result

    def backward_add(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let grad_a = copy(grad_output)
        let grad_b = copy(grad_output)
        return [grad_a, grad_b]

    def backward_sub(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let grad_a = copy(grad_output)
        let grad_b: [float] = []
        for g in grad_output:
            push(grad_b, -g)
        return [grad_a, grad_b]

    def backward_mul(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let a_data = entry.cached_values[0]
        let b_data = entry.cached_values[1]
        
        let grad_a: [float] = []
        let grad_b: [float] = []
        
        for i in length(grad_output):
            push(grad_a, grad_output[i] * b_data[i])
            push(grad_b, grad_output[i] * a_data[i])
        
        return [grad_a, grad_b]

    def backward_div(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let a_data = entry.cached_values[0]
        let b_data = entry.cached_values[1]
        
        let grad_a: [float] = []
        let grad_b: [float] = []
        
        for i in length(grad_output):
            let b = b_data[i]
            if abs(b) > 1e-10:
                push(grad_a, grad_output[i] / b)
                push(grad_b, -grad_output[i] * a_data[i] / (b * b))
            else:
                push(grad_a, 0.0)
                push(grad_b, 0.0)
        
        return [grad_a, grad_b]

    def backward_matmul(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let a_data = entry.cached_values[0]
        let b_data = entry.cached_values[1]
        let a_shape = entry.input_shapes[0]
        let b_shape = entry.input_shapes[1]
        
        let m = a_shape.dims[0]
        let k = a_shape.dims[1]
        let n = b_shape.dims[1]
        
        let grad_a: [float] = []
        for i in m:
            for j in k:
                var sum = 0.0
                for l in n:
                    sum += grad_output[i * n + l] * b_data[j * n + l]
                push(grad_a, sum)
        
        let grad_b: [float] = []
        for i in k:
            for j in n:
                var sum = 0.0
                for l in m:
                    sum += a_data[l * k + i] * grad_output[l * n + j]
                push(grad_b, sum)
        
        return [grad_a, grad_b]

    def backward_relu(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let x_data = entry.cached_values[0]
        let grad_x: [float] = []
        
        for i in length(grad_output):
            if x_data[i] > 0.0:
                push(grad_x, grad_output[i])
            else:
                push(grad_x, 0.0)
        
        return [grad_x]

    def backward_sigmoid(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let sig_output = entry.cached_values[0]
        let grad_x: [float] = []
        
        for i in length(grad_output):
            let s = sig_output[i]
            push(grad_x, grad_output[i] * s * (1.0 - s))
        
        return [grad_x]

    def backward_tanh(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let tanh_output = entry.cached_values[0]
        let grad_x: [float] = []
        
        for i in length(grad_output):
            let t = tanh_output[i]
            push(grad_x, grad_output[i] * (1.0 - t * t))
        
        return [grad_x]

    def backward_softmax(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let softmax_output = entry.cached_values[0]
        let input_shape = entry.input_shapes[0]
        let batch_size = input_shape.dims[0]
        let num_classes = input_shape.dims[1]
        
        let grad_x: [float] = []
        
        for b in batch_size:
            for i in num_classes:
                var sum = 0.0
                for j in num_classes:
                    let s_i = softmax_output[b * num_classes + i]
                    let s_j = softmax_output[b * num_classes + j]
                    let g_j = grad_output[b * num_classes + j]
                    if i == j:
                        sum += g_j * s_i * (1.0 - s_i)
                    else:
                        sum += g_j * (-s_i * s_j)
                push(grad_x, sum)
        
        return [grad_x]

    def backward_exp(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let exp_output = entry.cached_values[0]
        let grad_x: [float] = []
        
        for i in length(grad_output):
            push(grad_x, grad_output[i] * exp_output[i])
        
        return [grad_x]

    def backward_log(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let x_data = entry.cached_values[0]
        let grad_x: [float] = []
        
        for i in length(grad_output):
            if x_data[i] > 0.0:
                push(grad_x, grad_output[i] / x_data[i])
            else:
                push(grad_x, 0.0)
        
        return [grad_x]

    def backward_sum(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let input_shape = entry.input_shapes[0]
        let axis = int(entry.extra_params[0])
        
        let grad_x: [float] = []
        
        if axis < 0:
            for i in input_shape.size:
                push(grad_x, grad_output[0])
        else:
            let outer_size = input_shape.dims[0]
            let inner_size = input_shape.dims[1]
            
            if axis == 0:
                for i in outer_size:
                    for j in inner_size:
                        push(grad_x, grad_output[j])
            else:
                for i in outer_size:
                    for j in inner_size:
                        push(grad_x, grad_output[i])
        
        return [grad_x]

    def backward_mean(entry: TapeEntry, grad_output: [float]) -> [[float]]:
        let input_shape = entry.input_shapes[0]
        let axis = int(entry.extra_params[0])
        let total_size = int(entry.extra_params[1])
        
        let grad_x: [float] = []
        
        if axis < 0:
            let scale = 1.0 / float(total_size)
            for i in input_shape.size:
                push(grad_x, grad_output[0] * scale)
        else:
            let outer_size = input_shape.dims[0]
            let inner_size = input_shape.dims[1]
            
            if axis == 0:
                let scale = 1.0 / float(outer_size)
                for i in outer_size:
                    for j in inner_size:
                        push(grad_x, grad_output[j] * scale)
            else:
                let scale = 1.0 / float(inner_size)
                for i in outer_size:
                    for j in inner_size:
                        push(grad_x, grad_output[i] * scale)
        
        return [grad_x]

    def backward(loss_var: Variable):
        loss_var.grad[0] = 1.0
        
        var grad_table: [[float]] = []
        for v in graph.variables:
            push(grad_table, map(v.data): 0.0)
        
        grad_table[loss_var.id][0] = 1.0
        
        let ops = graph.tape.operations
        var i = length(ops) - 1
        while i >= 0:
            let entry = ops[i]
            let output_grad = grad_table[entry.output_id]
            
            var input_grads: [[float]] = []
            
            switch entry.op_type:
                case add_op:
                    input_grads = backward_add(entry, output_grad)
                case sub_op:
                    input_grads = backward_sub(entry, output_grad)
                case mul_op:
                    input_grads = backward_mul(entry, output_grad)
                case div_op:
                    input_grads = backward_div(entry, output_grad)
                case matmul_op:
                    input_grads = backward_matmul(entry, output_grad)
                case relu_op:
                    input_grads = backward_relu(entry, output_grad)
                case sigmoid_op:
                    input_grads = backward_sigmoid(entry, output_grad)
                case tanh_op:
                    input_grads = backward_tanh(entry, output_grad)
                case softmax_op:
                    input_grads = backward_softmax(entry, output_grad)
                case exp_op:
                    input_grads = backward_exp(entry, output_grad)
                case log_op:
                    input_grads = backward_log(entry, output_grad)
                case sum_op:
                    input_grads = backward_sum(entry, output_grad)
                case mean_op:
                    input_grads = backward_mean(entry, output_grad)
                default:
                    pass
            
            for j in length(entry.input_ids):
                let input_id = entry.input_ids[j]
                let input_grad = input_grads[j]
                for k in length(input_grad):
                    grad_table[input_id][k] += input_grad[k]
            
            i -= 1
        
        for v in graph.variables:
            if v.requires_grad:
                v.grad = grad_table[v.id]

    def compute_jacobian(inputs: [Variable], outputs: [Variable]) -> [[float]]:
        let num_inputs = length(inputs)
        let num_outputs = length(outputs)
        
        var total_input_size = 0
        for inp in inputs:
            total_input_size += inp.shape.size
        
        var total_output_size = 0
        for out in outputs:
            total_output_size += out.shape.size
        
        let jacobian: [[float]] = []
        for i in total_output_size:
            let row: [float] = []
            for j in total_input_size:
                push(row, 0.0)
            push(jacobian, row)
        
        var output_offset = 0
        for out_idx in num_outputs:
            let out_var = outputs[out_idx]
            for out_elem in out_var.shape.size:
                for v in graph.variables:
                    v.zero_grad()
                
                out_var.grad[out_elem] = 1.0
                
                let ops = graph.tape.operations
                var i = length(ops) - 1
                while i >= 0:
                    let entry = ops[i]
                    let output_grad = graph.get_variable(entry.output_id).grad
                    
                    var input_grads: [[float]] = []
                    
                    switch entry.op_type:
                        case add_op:
                            input_grads = backward_add(entry, output_grad)
                        case mul_op:
                            input_grads = backward_mul(entry, output_grad)
                        case matmul_op:
                            input_grads = backward_matmul(entry, output_grad)
                        default:
                            input_grads = []
                            for id in entry.input_ids:
                                push(input_grads, map(output_grad): 0.0)
                    
                    for j in length(entry.input_ids):
                        let input_id = entry.input_ids[j]
                        let input_var = graph.get_variable(input_id)
                        for k in length(input_grads[j]):
                            input_var.grad[k] += input_grads[j][k]
                    
                    i -= 1
                
                var input_offset = 0
                for inp_idx in num_inputs:
                    let inp_var = inputs[inp_idx]
                    for inp_elem in inp_var.shape.size:
                        jacobian[output_offset + out_elem][input_offset + inp_elem] = inp_var.grad[inp_elem]
                    input_offset += inp_var.shape.size
            
            output_offset += out_var.shape.size
        
        return jacobian

    def gradient_check(func, inputs: [Variable], epsilon: float) -> float:
        var max_diff = 0.0
        
        for inp in inputs:
            for i in length(inp.data):
                let original = inp.data[i]
                
                inp.data[i] = original + epsilon
                let loss_plus = compute_loss(func, inputs)
                
                inp.data[i] = original - epsilon
                let loss_minus = compute_loss(func, inputs)
                
                inp.data[i] = original
                
                let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
                let analytical_grad = inp.grad[i]
                
                let diff = abs(numerical_grad - analytical_grad)
                let scale = max(abs(numerical_grad), abs(analytical_grad)) + 1e-8
                let relative_diff = diff / scale
                
                max_diff = max(max_diff, relative_diff)
        
        return max_diff

    def compute_loss(func, inputs: [Variable]) -> float:
        return 0.0

    def clear_graph():
        graph.tape.clear()
        graph.variables = []
        graph.next_id = 0

class GradientAccumulator:
    accumulated_grads: [[float]]
    accumulation_steps: int
    current_step: int

    def new(num_params: int) -> GradientAccumulator:
        let grads: [[float]] = []
        for i in num_params:
            push(grads, [])
        GradientAccumulator {
            accumulated_grads: grads,
            accumulation_steps: 1,
            current_step: 0
        }

    def set_accumulation_steps(steps: int):
        accumulation_steps = steps

    def accumulate(param_idx: int, grads: [float]):
        if length(accumulated_grads[param_idx]) == 0:
            accumulated_grads[param_idx] = copy(grads)
        else:
            for i in length(grads):
                accumulated_grads[param_idx][i] += grads[i]

    def get_averaged_grads(param_idx: int) -> [float]:
        let scale = 1.0 / float(accumulation_steps)
        let result: [float] = []
        for g in accumulated_grads[param_idx]:
            push(result, g * scale)
        return result

    def step() -> bool:
        current_step += 1
        if current_step >= accumulation_steps:
            current_step = 0
            return true
        return false

    def zero_accumulated():
        for i in length(accumulated_grads):
            accumulated_grads[i] = []

class MixedPrecisionGradient:
    master_weights: [[float]]
    fp16_grads: [[float]]
    loss_scale: float
    scale_growth_factor: float
    scale_backoff_factor: float
    scale_growth_interval: int
    steps_since_scale_change: int
    overflow_detected: bool

    def new() -> MixedPrecisionGradient:
        MixedPrecisionGradient {
            master_weights: [],
            fp16_grads: [],
            loss_scale: 65536.0,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: 2000,
            steps_since_scale_change: 0,
            overflow_detected: false
        }

    def scale_gradients(grads: [float]) -> [float]:
        let scaled: [float] = []
        for g in grads:
            push(scaled, g * loss_scale)
        return scaled

    def unscale_gradients(grads: [float]) -> [float]:
        let unscaled: [float] = []
        overflow_detected = false
        for g in grads:
            let val = g / loss_scale
            if abs(val) > 65504.0 or val != val:
                overflow_detected = true
            push(unscaled, val)
        return unscaled

    def update_scale():
        if overflow_detected:
            loss_scale *= scale_backoff_factor
            steps_since_scale_change = 0
        else:
            steps_since_scale_change += 1
            if steps_since_scale_change >= scale_growth_interval:
                loss_scale *= scale_growth_factor
                steps_since_scale_change = 0

def run_gradient_tests():
    let computer = GradientComputer.new()
    
    let a_data = [1.0, 2.0, 3.0, 4.0]
    let a_shape = TensorShape.new([2, 2])
    let a = computer.graph.create_variable(a_data, a_shape, true, "a")
    
    let b_data = [5.0, 6.0, 7.0, 8.0]
    let b_shape = TensorShape.new([2, 2])
    let b = computer.graph.create_variable(b_data, b_shape, true, "b")
    
    let c = computer.forward_add(a, b)
    let d = computer.forward_mul(c, b)
    let loss = computer.forward_sum(d, -1)
    
    computer.backward(loss)
    
    print("Gradient computation test completed")
    print("a.grad: " + string(a.grad))
    print("b.grad: " + string(b.grad))

def main():
    run_gradient_tests()
