// Copyright 2024, Lucas Arriesse (Game4all)
// Please see the license file at root of repository for more information.

const std = @import("std");
const brainz = @import("brainz");
const mnist = @import("mnist.zig");
const model = @import("model.zig");

const MNISTDataset = mnist.MNISTDataset;
const MNISTClassifier = model.MNISTClassifier;

const Tensor = brainz.Tensor;
const TensorArena = brainz.TensorArena;
const LinearPlan = brainz.LinearPlan;
const ExecutionPlan = brainz.ExecutionPlan;

const MNISTClassifierNet = brainz.nn.Sequential(MNISTClassifier);

const BATCH_SIZE: usize = 32;
const BASE_LEARNING_RATE: f32 = 0.1;
const NUM_EPOCHS: usize = 15;

pub fn main() !void {
    // getting handle to stdout
    var stdoutBuffer: [1024]u8 = undefined;
    var stdoutWriter = std.fs.File.stdout().writer(&stdoutBuffer);
    const stdout = &stdoutWriter.interface;

    // setup gpa for training
    var gpa: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer gpa.deinit();

    const allocator = gpa.allocator();

    // setup tensor allocator + compute graph builder
    var tensorArena: TensorArena = .init(allocator);
    var planBuilder: LinearPlan = .init(&tensorArena, allocator);

    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();

    var network: MNISTClassifierNet = try .init(.{&planBuilder});

    // defining the compute plan
    const inputs = try planBuilder.createInput("input", .float32, .fromSlice(&.{ BATCH_SIZE, MNISTClassifier.IN_FEATURES }), false); // the tensor with input data
    const targets = try planBuilder.createInput("targets", .float32, .fromSlice(&.{ BATCH_SIZE, MNISTClassifier.OUT_FEATURES }), false); // the tensor with OHE'ed target labels

    const y_pred = try network.forward(&planBuilder, inputs);
    const y_softmax = try brainz.ops.softmax(&planBuilder, y_pred, 1);
    const y_indices = try brainz.ops.argMax(&planBuilder, y_softmax, 1); // argMax of y_pred

    try planBuilder.registerOutput("y_pred", y_pred);
    try planBuilder.registerOutput("y_indices", y_indices);

    const loss = try brainz.ops.crossEntropyLoss(&planBuilder, y_softmax, targets);
    try planBuilder.registerOutput("loss", loss);

    // finalize the plan and allocate backing memory for tensors.
    var plan = try planBuilder.finalize(true);
    try tensorArena.allocateStorage();

    // randomly initialize net weights (now that we've got physical storage for our tensors)
    network.initializeWeights(rnd);

    // load the train and the test datasets
    var trainDataset: MNISTDataset = try .init(allocator, "mnist/train-labels.idx1-ubyte", "mnist/train-images.idx3-ubyte");
    var testDataset: MNISTDataset = try .init(allocator, "mnist/t10k-labels.idx1-ubyte", "mnist/t10k-images.idx3-ubyte");

    // init the optimizer
    var sgd: brainz.optim.SGD = .init(plan.getParams(), BASE_LEARNING_RATE);
    const lossGrad = loss.grad.?.slice(f32).?;

    try stdout.print("Starting training", .{});
    try stdout.flush();

    for (0..NUM_EPOCHS) |epoch| {
        var epochLoss: f32 = 0.0;
        var iterator = trainDataset.iterator(BATCH_SIZE);
        var batchNum: usize = 0;

        const labelDataSlice = targets.slice(f32).?;
        const inputDataSlice = inputs.slice(f32).?;
        while (iterator.next()) {

            // memset label data to zero then OHE encode it
            iterator.copyLabelData(labelDataSlice);

            // copy image data to the input tensor
            iterator.copyImageData(inputDataSlice);

            // perform forward pass and log loss
            try plan.forward();
            epochLoss += loss.scalar(f32).?;

            plan.zeroGrad(); // zero stored gradients
            lossGrad[0] = 1.0; // seed gradient for backprop
            try plan.backward(); // perform back prop

            // optimize
            sgd.step();

            if (batchNum % 100 == 0) {
                try stdout.print("\rEpoch: {} | Loss: {} | Batch: {} / {} ", .{ epoch, epochLoss, iterator.pos, trainDataset.count_images });
                try stdout.flush();
            }
            batchNum += 1;
        }

        // evaluate the model on the test dataset
        const testAccuracy = try evaluateModelAccuracy(&testDataset, &plan, inputs, y_indices);

        try stdout.print("\rEpoch: {} | Train loss: {} | Test Accuracy: {d:.2}%  \n\r", .{ epoch, epochLoss, testAccuracy * 100.0 });
        try stdout.flush();

        try stdout.print("Saving model to disk", .{});
        try stdout.flush();

        var writeBuffer: [1024]u8 = undefined;
        var weightsFile = try std.fs.cwd().createFile("src/model.bin", .{});
        var weightWriter = weightsFile.writer(&writeBuffer);
        const weightWriterIo = &weightWriter.interface;

        try network.getInner().saveWeights(weightWriterIo);

        try stdout.print("Model saved to disk.", .{});
        try stdout.flush();
    }
}

fn evaluateModelAccuracy(
    evaluation_dataset: *const MNISTDataset,
    plan: *ExecutionPlan,
    input: *const Tensor,
    y_indices: *const Tensor,
) !f32 {
    var correct_predictions: usize = 0;
    var total_predictions: usize = 0;
    var iterator = evaluation_dataset.iterator(BATCH_SIZE);
    while (iterator.next()) {
        const inputDataSlice = input.slice(f32).?;
        const predictedIndices = y_indices.slice(usize).?;
        const labels = iterator.getLabels() orelse continue;

        iterator.copyImageData(inputDataSlice);

        // perform forward pass
        try plan.forward();

        // compute model accuracy on MNIST test dataset
        for (labels, 0..) |label, idx| {
            if (predictedIndices[idx] == label) {
                correct_predictions += 1;
            }
            total_predictions += 1;
        }
    }
    return @as(f32, @floatFromInt(correct_predictions)) / @as(f32, @floatFromInt(total_predictions));
}
