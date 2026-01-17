// Copyright 2024, Lucas Arriesse (Game4all)
// Please see the license file at root of repository for more information.

const std = @import("std");
const brainz = @import("brainz");

const TensorArena = brainz.TensorArena;
const LinearPlan = brainz.LinearPlan;
const ExecutionPlan = brainz.ExecutionPlan;
const Tensor = brainz.Tensor;
const MNISTClassifier = @import("model.zig").MNISTClassifier;
const MNISTClassifierNet = brainz.nn.Sequential(MNISTClassifier);

/// The model weights
const MODEL_WEIGHTS = @embedFile("model.bin");

/// Model data (weights)
var model: MNISTClassifierNet = undefined;

/// Tensor Arena and finalized execution plan
var tensorArena: TensorArena = undefined;
var execPlan: ExecutionPlan = undefined;

/// Tensor containing the inputs.
var inputTensor: ?*const Tensor = null;
/// Softmax logits tensor
var modelLogits: ?*const Tensor = null;
/// Result indices tensor
var argmaxResults: ?*const Tensor = null;

/// Loss gradient tensor.
var lossTensor: ?*const Tensor = null;
/// One hot encoded tensor with the expected class logit.
var targetLabelTensor: ?*const Tensor = null;

/// Initializes the model and loads its weights for inferrence.
pub export fn init() u32 {
    const allocator = std.heap.wasm_allocator;

    // setup tensor arena + plan builder
    tensorArena = TensorArena.init(allocator);
    var planBuilder: LinearPlan = .init(&tensorArena, allocator);
    defer planBuilder.deinit();

    // create an input tensor
    inputTensor = planBuilder.createInput("input", .float32, .fromSlice(&.{ 1, MNISTClassifier.IN_FEATURES }), false) catch return 1;
    targetLabelTensor = planBuilder.createInput("target_label", .float32, .fromSlice(&.{ 1, MNISTClassifier.IN_FEATURES }), false) catch return 1;

    // building model + writing forward pass
    model = MNISTClassifierNet.init(.{&planBuilder}) catch return 1;

    // model forward -> model output -> softmaxed model logits -> final label by argmax

    // raw model outputs
    const rawLogits = model.forward(&planBuilder, inputTensor.?) catch return 1;
    // transform raw model outputs into usable logits
    modelLogits = brainz.ops.softmax(&planBuilder, rawLogits, 1) catch return 1;
    // compute the predicted label by argmax-ing
    argmaxResults = brainz.ops.argMax(&planBuilder, modelLogits.?, 1) catch return 1;

    // compute model loss on the side for training on mis-predictions
    lossTensor = brainz.ops.crossEntropyLoss(&planBuilder, modelLogits.?, targetLabelTensor.?) catch return 1;

    // finalize execution plan and allocate tensor storage
    execPlan = planBuilder.finalize(true) catch return 1;
    tensorArena.allocateStorage() catch return 1;

    // read the model weights
    var weightReader = std.Io.Reader.fixed(MODEL_WEIGHTS);
    model.layers.loadWeights(&weightReader) catch return 1;

    return 0;
}

/// Returns the pointer to the image data.
pub export fn getInputDataPointer() usize {
    return @intFromPtr(inputTensor.?.storage.?);
}

/// Returns the pointer to the raw output logits of the model.
pub export fn getOutputLogitsPointer() usize {
    return @intFromPtr(modelLogits.?.storage.?);
}

/// Performs inference using the specified data and returns the result
/// The return value indicates the number class with the highest score.
pub export fn runInference() u32 {
    execPlan.forward() catch return 0;
    return @intCast(argmaxResults.?.slice(usize).?[0]);
}

/// Trains the model on the current drawing in the input tensor
pub export fn train(expected: usize) u32 {
    // set the target label in the target label tensor
    const tgtDataSlice = targetLabelTensor.?.slice(f32).?;
    @memset(tgtDataSlice, 0);
    tgtDataSlice[expected] = 1.0;

    // create optimizer
    var sgd: brainz.optim.SGD = .init(execPlan.getParams(), 0.1);

    // perform forward pass to compute loss first
    execPlan.forward() catch return 0;
    execPlan.zeroGrad(); // zero gradients
    lossTensor.?.grad.?.slice(f32).?[0] = 1.0; // seed the loss gradient for backprop
    // perform backprop
    execPlan.backward() catch return 0;

    // update weights with SGD
    sgd.step();

    return 0;
}
