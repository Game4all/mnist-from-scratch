// Copyright 2024, Lucas Arriesse (Game4all)
// Please see the license file at root of repository for more information.

const std = @import("std");
const brainz = @import("brainz");

const Tensor = brainz.Tensor;
const MNISTClassifier = @import("model.zig").MNISTClassifier;
const DummyDevice = brainz.Device.DummyDevice;

/// The model weights
const MODEL_WEIGHTS = @embedFile("model.bin");

/// Model itself.
var model: MNISTClassifier(1) = undefined;
/// Tensor containing the inputs.
var input_tensor: Tensor(f32) = undefined;
/// Results tensor.
var argmax_result: Tensor(usize) = undefined;

/// Initializes the model and loads its weights for inferrence.
pub export fn init() u32 {
    const allocator = std.heap.wasm_allocator;

    model.init(allocator) catch return 1;
    var fixedStream = std.io.fixedBufferStream(MODEL_WEIGHTS);
    const reader = fixedStream.reader();
    model.load(reader) catch return 1;

    input_tensor = Tensor(f32).init(model.inputShape(), allocator) catch return 1;
    argmax_result = Tensor(usize).init(.{ 1, 0, 1 }, allocator) catch return 1;

    return 0;
}

/// Returns the pointer to the image data.
pub export fn getInputDataPointer() usize {
    return @intFromPtr(input_tensor.data.ptr);
}

/// Returns the pointer to the raw output logits of the model.
pub export fn getOutputLogitsPointer() usize {
    return @intFromPtr(model.layer_4.activation_outputs.data.ptr);
}

/// Performs inference using the specified data and returns the result
/// The return value indicates the number class with the highest score.
pub export fn runInference() u32 {
    const result = model.forward(DummyDevice, &input_tensor) catch return 0;
    brainz.ops.argmax(f32, result, 1, &argmax_result);
    return @truncate(argmax_result.get(.{ 0, 0, 0 }));
}
