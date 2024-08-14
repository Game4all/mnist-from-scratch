// Copyright 2024, Lucas Arriesse (Game4all)
// Please see the license file at root of repository for more information.

const brainz = @import("brainz");
const std = @import("std");

const Dense = brainz.Dense;
const Device = brainz.Device;
const Allocator = std.mem.Allocator;
const Tensor = brainz.Tensor;

/// A MLP-based neural network for handwritten digit classification.
pub fn MNISTClassifier(comptime num_batches: usize) type {
    return struct {
        layer_1: Dense(784, 256, num_batches, brainz.activation.Sigmoid) = undefined,
        layer_2: Dense(256, 192, num_batches, brainz.activation.Sigmoid) = undefined,
        layer_3: Dense(192, 48, num_batches, brainz.activation.Sigmoid) = undefined,
        layer_4: Dense(48, 10, num_batches, brainz.activation.Sigmoid) = undefined,

        // weight gradients
        weight_grad_1: Tensor(f32),
        weight_grad_2: Tensor(f32),
        weight_grad_3: Tensor(f32),
        weight_grad_4: Tensor(f32),

        // weight gradients flattened
        weight_grad_1_f: Tensor(f32),
        weight_grad_2_f: Tensor(f32),
        weight_grad_3_f: Tensor(f32),
        weight_grad_4_f: Tensor(f32),

        bias_grad_1: Tensor(f32),
        bias_grad_2: Tensor(f32),
        bias_grad_3: Tensor(f32),
        bias_grad_4: Tensor(f32),

        pub fn forward(self: *@This(), device: Device, input: *const Tensor(f32)) !*Tensor(f32) {
            const a = try self.layer_1.forward(device, input);
            const b = try self.layer_2.forward(device, a);
            const c = try self.layer_3.forward(device, b);
            return self.layer_4.forward(device, c);
        }

        pub fn backwards(self: *@This(), device: Device, loss_grad: *const Tensor(f32)) !void {
            const D = try self.layer_4.backwards(device, loss_grad);
            const C = try self.layer_3.backwards(device, D);
            const B = try self.layer_2.backwards(device, C);
            _ = try self.layer_1.backwards(device, B);
        }

        pub inline fn inputShape(self: *@This()) struct { usize, usize, usize } {
            return self.layer_1.inputShape();
        }

        pub inline fn outputShape(self: *@This()) struct { usize, usize, usize } {
            return self.layer_4.outputShape();
        }

        pub fn step(self: *@This(), device: Device, ins: *const Tensor(f32), lr: f32) !void {
            const layer4_inputs = self.layer_3.activation_outputs.transpose();
            const layer3_inputs = self.layer_2.activation_outputs.transpose();
            const layer2_inputs = self.layer_1.activation_outputs.transpose();
            const layer1_inputs = ins.transpose();

            try brainz.ops.matMul(f32, device, &self.layer_1.grad, &layer1_inputs, &self.weight_grad_1);
            try brainz.ops.matMul(f32, device, &self.layer_2.grad, &layer2_inputs, &self.weight_grad_2);
            try brainz.ops.matMul(f32, device, &self.layer_3.grad, &layer3_inputs, &self.weight_grad_3);
            try brainz.ops.matMul(f32, device, &self.layer_4.grad, &layer4_inputs, &self.weight_grad_4);

            try device.barrier();

            // batched weights reduction

            try brainz.ops.reduce(f32, device, .Sum, &self.weight_grad_1, 0, &self.weight_grad_1_f);
            try brainz.ops.reduce(f32, device, .Sum, &self.weight_grad_2, 0, &self.weight_grad_2_f);
            try brainz.ops.reduce(f32, device, .Sum, &self.weight_grad_3, 0, &self.weight_grad_3_f);
            try brainz.ops.reduce(f32, device, .Sum, &self.weight_grad_4, 0, &self.weight_grad_4_f);

            try brainz.ops.reduce(f32, device, .Sum, &self.layer_1.grad, 0, &self.bias_grad_1);
            try brainz.ops.reduce(f32, device, .Sum, &self.layer_2.grad, 0, &self.bias_grad_2);
            try brainz.ops.reduce(f32, device, .Sum, &self.layer_3.grad, 0, &self.bias_grad_3);
            try brainz.ops.reduce(f32, device, .Sum, &self.layer_4.grad, 0, &self.bias_grad_4);

            try device.barrier();

            // weight updates

            const alpha = lr * @as(f32, @floatFromInt(num_batches)) * (1.0 / @as(f32, @floatFromInt(num_batches)));

            try brainz.ops.sub(f32, device, &self.layer_1.weights, &self.weight_grad_1_f, &self.layer_1.weights, .{ .alpha = alpha });
            try brainz.ops.sub(f32, device, &self.layer_1.biases, &self.bias_grad_1, &self.layer_1.biases, .{ .alpha = alpha });

            try brainz.ops.sub(f32, device, &self.layer_2.weights, &self.weight_grad_2_f, &self.layer_2.weights, .{ .alpha = alpha });
            try brainz.ops.sub(f32, device, &self.layer_2.biases, &self.bias_grad_2, &self.layer_2.biases, .{ .alpha = alpha });

            try brainz.ops.sub(f32, device, &self.layer_3.weights, &self.weight_grad_3_f, &self.layer_3.weights, .{ .alpha = alpha });
            try brainz.ops.sub(f32, device, &self.layer_3.biases, &self.bias_grad_3, &self.layer_3.biases, .{ .alpha = alpha });

            try brainz.ops.sub(f32, device, &self.layer_4.weights, &self.weight_grad_4_f, &self.layer_4.weights, .{ .alpha = alpha });
            try brainz.ops.sub(f32, device, &self.layer_4.biases, &self.bias_grad_4, &self.layer_4.biases, .{ .alpha = alpha });

            try device.barrier();
        }

        pub fn init(self: *@This(), alloc: Allocator) !void {
            try self.layer_1.init(alloc);
            try self.layer_2.init(alloc);
            try self.layer_3.init(alloc);
            try self.layer_4.init(alloc);

            const wg1_shape = try brainz.ops.opShape(
                .MatMul,
                self.layer_1.grad.shape,
                try brainz.ops.opShape(.Transpose, self.layer_1.inputShape(), null),
            );

            const wg2_shape = try brainz.ops.opShape(
                .MatMul,
                self.layer_2.grad.shape,
                try brainz.ops.opShape(.Transpose, self.layer_1.outputShape(), null),
            );

            const wg3_shape = try brainz.ops.opShape(
                .MatMul,
                self.layer_3.grad.shape,
                try brainz.ops.opShape(.Transpose, self.layer_2.outputShape(), null),
            );

            const wg4_shape = try brainz.ops.opShape(
                .MatMul,
                self.layer_4.grad.shape,
                try brainz.ops.opShape(.Transpose, self.layer_3.outputShape(), null),
            );

            self.weight_grad_1 = try Tensor(f32).init(wg1_shape, alloc);
            self.weight_grad_2 = try Tensor(f32).init(wg2_shape, alloc);
            self.weight_grad_3 = try Tensor(f32).init(wg3_shape, alloc);
            self.weight_grad_4 = try Tensor(f32).init(wg4_shape, alloc);

            self.weight_grad_1_f = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, wg1_shape, 0), alloc);
            self.weight_grad_2_f = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, wg2_shape, 0), alloc);
            self.weight_grad_3_f = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, wg3_shape, 0), alloc);
            self.weight_grad_4_f = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, wg4_shape, 0), alloc);

            self.bias_grad_1 = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.layer_1.outputShape(), 0), alloc);
            self.bias_grad_2 = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.layer_2.outputShape(), 0), alloc);
            self.bias_grad_3 = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.layer_3.outputShape(), 0), alloc);
            self.bias_grad_4 = try Tensor(f32).init(try brainz.ops.opShape(.Reduce, self.layer_4.outputShape(), 0), alloc);
        }

        /// Saves this model weights to disk.
        pub fn saveModelToFile(self: *@This(), path: [:0]const u8) !void {
            var file = try std.fs.cwd().createFile(path, .{});
            var writer = file.writer();

            // save layer 1 weights + biases
            try writer.writeAll(std.mem.sliceAsBytes(self.layer_1.weights.constSlice()));
            try writer.writeAll(std.mem.sliceAsBytes(self.layer_1.biases.constSlice()));

            // save layer 2 weights + biases
            try writer.writeAll(std.mem.sliceAsBytes(self.layer_2.weights.constSlice()));
            try writer.writeAll(std.mem.sliceAsBytes(self.layer_2.biases.constSlice()));

            // save layer 3 weights + biases
            try writer.writeAll(std.mem.sliceAsBytes(self.layer_3.weights.constSlice()));
            try writer.writeAll(std.mem.sliceAsBytes(self.layer_3.biases.constSlice()));

            // save layer 4 weights + biases
            try writer.writeAll(std.mem.sliceAsBytes(self.layer_4.weights.constSlice()));
            try writer.writeAll(std.mem.sliceAsBytes(self.layer_4.biases.constSlice()));
        }

        /// Loads the model weights from a reader.
        pub fn load(self: *@This(), reader: anytype) !void {
            _ = try reader.read(std.mem.sliceAsBytes(self.layer_1.weights.slice()));
            _ = try reader.read(std.mem.sliceAsBytes(self.layer_1.biases.slice()));

            _ = try reader.read(std.mem.sliceAsBytes(self.layer_2.weights.slice()));
            _ = try reader.read(std.mem.sliceAsBytes(self.layer_2.biases.slice()));

            _ = try reader.read(std.mem.sliceAsBytes(self.layer_3.weights.slice()));
            _ = try reader.read(std.mem.sliceAsBytes(self.layer_3.biases.slice()));

            _ = try reader.read(std.mem.sliceAsBytes(self.layer_4.weights.slice()));
            _ = try reader.read(std.mem.sliceAsBytes(self.layer_4.biases.slice()));
        }

        /// Loads the model weights from a file on disk.
        pub fn loadFromFile(self: *@This(), path: [:0]const u8) !void {
            const file = try std.fs.cwd().openFile(path, .{});
            const reader = file.reader();
            return try self.load(reader);
        }
    };
}
