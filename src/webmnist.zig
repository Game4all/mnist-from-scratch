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
    initInner() catch |e| return @intFromError(e);
    return 0;
}

fn initInner() !void {
    const allocator = std.heap.wasm_allocator;

    // setup tensor arena + plan builder
    tensorArena = TensorArena.init(allocator);
    var planBuilder: LinearPlan = .init(&tensorArena, allocator);
    defer planBuilder.deinit();

    // create an input tensor
    inputTensor = try planBuilder.createInput("input", .float32, .fromSlice(&.{ 1, MNISTClassifier.IN_FEATURES }), false);

    // create the tensor containing the one hot encoded target label for retraining on a misclassified input
    targetLabelTensor = try planBuilder.createInput("target_label", .float32, .fromSlice(&.{ 1, MNISTClassifier.OUT_FEATURES }), false);

    // building model + writing forward pass
    model = try MNISTClassifierNet.init(.{&planBuilder});

    // model forward -> model output -> softmaxed model logits -> final label by argmax

    // raw model outputs
    const rawLogits = try model.forward(&planBuilder, inputTensor.?);
    // transform raw model outputs into usable logits
    modelLogits = try brainz.ops.softmax(&planBuilder, rawLogits, 1);
    // compute the predicted label by argmax-ing
    argmaxResults = try brainz.ops.argMax(&planBuilder, modelLogits.?, 1);

    // compute model loss on the side for training on mis-predictions
    lossTensor = try brainz.ops.crossEntropyLoss(&planBuilder, modelLogits.?, targetLabelTensor.?);

    // finalize execution plan and allocate tensor storage
    execPlan = try planBuilder.finalize(true);
    try tensorArena.allocateStorage();

    // read the model weights
    var weightReader = std.Io.Reader.fixed(MODEL_WEIGHTS);
    model.layers.loadWeights(&weightReader) catch return; //FIXME: this shouldn't error out, but idk why
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
    var sgd: brainz.optim.SGD = .init(execPlan.getParams(), 0.001);

    // perform forward pass to compute loss first
    execPlan.forward() catch return 0;
    // reread the loss on the sample
    execPlan.zeroGrad(); // zero gradients
    lossTensor.?.grad.?.slice(f32).?[0] = 1.0; // seed the loss gradient for backprop
    // perform backprop
    execPlan.backward() catch return 0;

    // update weights with SGD
    sgd.step();

    return 0;
}
