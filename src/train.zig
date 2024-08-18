// Copyright 2024, Lucas Arriesse (Game4all)
// Please see the license file at root of repository for more information.

const std = @import("std");
const brainz = @import("brainz");
const mnist = @import("mnist.zig");

const MNISTDataset = mnist.MNISTDataset;
const MNISTClassifier = @import("model.zig").MNISTClassifier;
const Tensor = brainz.Tensor;

const BASE_LEARNING_RATE: f32 = 5e-2;
const NUM_EPOCHS: usize = 15;

pub fn main() !void {
    const out = std.io.getStdOut();
    const writer = out.writer();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var cpudev: brainz.preferred_device_type = .{};
    try cpudev.init(gpa.allocator(), null);
    defer cpudev.deinit();

    const device = cpudev.device();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const arena_alloc = arena.allocator();

    var net: MNISTClassifier(128) = undefined;
    try net.init(arena_alloc);
    net.loadFromFile("src/model.bin") catch |err| {
        try writer.print("Failed to load a model checkpoint: {} \n", .{err});
        try writer.print("Starting from scratch ... \n", .{});
    };

    var training_dataset = try MNISTDataset.init(arena_alloc, "train-labels.idx1-ubyte", "train-images.idx3-ubyte", &net);
    var evaluation_dataset = try MNISTDataset.init(arena_alloc, "t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte", &net);

    const num_training_images = training_dataset.image_data.len / MNISTDataset.IMAGE_SIZE;

    var softmaxed_logits = try Tensor(f32).init(net.outputShape(), arena_alloc);
    var loss = try Tensor(f32).init(net.outputShape(), arena_alloc);

    const tb = try std.time.Instant.now();

    for (0..NUM_EPOCHS) |e| {
        // cyclical cosine learning rate scheduling
        const lr: f32 = 1e-5 + BASE_LEARNING_RATE * @abs(@cos(@as(f32, @floatFromInt(e)) / std.math.pi));

        var total_loss: f32 = 0.0;
        var batch_iterator = training_dataset.iterator();

        while (try batch_iterator.next_train(device)) |data| {
            const inputs, const labels = data;
            const result = try net.forward(device, inputs);

            // apply softmax to the output logits
            try brainz.ops.softmax(f32, device, result, 1, &softmaxed_logits);
            try device.barrier();

            try brainz.ops.categoricalCrossEntropyLossBackprop(f32, device, &softmaxed_logits, labels, &loss);
            total_loss += brainz.ops.categoricalCrossEntropyLoss(f32, device, &softmaxed_logits, labels);

            try device.barrier();

            try writer.print("\r=> epoch: {} | loss: {d:.3} | {}/{} | lr={e:.3}    ", .{ e, total_loss, batch_iterator.pos / MNISTDataset.IMAGE_SIZE, num_training_images, lr });
            try net.backwards(device, &loss);
            try net.step(device, inputs, BASE_LEARNING_RATE);
        }

        const accuracy = try evaluate_model_accuracy(&evaluation_dataset, device, &net);
        try writer.print("\r epoch: {} | loss: {d:.3} | lr={e:.3} | Validation Acc: {d:.2}% \n", .{ e, total_loss, lr, accuracy * 100.0 });

        if (e % 10 == 0) {
            try net.saveModelToFile("src/model.bin");
            try writer.print("Saved model checkpoint to disk. \n", .{});
        }
    }

    const after = try std.time.Instant.now();
    const diff = after.since(tb) / std.time.ns_per_s;

    try net.saveModelToFile("src/model.bin");
    try writer.print("\n Training took {}s / {}min", .{ diff, diff / std.time.s_per_min });
}

fn evaluate_model_accuracy(
    evaluation_dataset: *MNISTDataset,
    device: anytype,
    net: anytype,
) !f32 {
    const total_batches: usize = (evaluation_dataset.label_data.len / evaluation_dataset.batch_size) * evaluation_dataset.batch_size;
    var num_correct: usize = 0;

    var iter = evaluation_dataset.iterator();

    // local allocator
    var buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const fba_alloc = fba.allocator();

    // tensors labels from output
    var out_labels = try Tensor(usize).init(.{ net.outputShape().@"0", 0, 1 }, fba_alloc);
    var expected_labels = try Tensor(usize).init(.{ net.outputShape().@"0", 0, 1 }, fba_alloc);

    while (try iter.next_eval(device)) |data| {
        const inputs, const labels = data;

        const result = try net.forward(device, inputs);
        brainz.ops.argmax(f32, result, 1, &out_labels); //argmax the guessed labels

        try brainz.ops.cast(u8, usize, device, &labels, &expected_labels);
        try device.barrier();

        try brainz.ops.elementWiseEq(usize, device, &out_labels, &expected_labels, &expected_labels);
        try device.barrier();

        num_correct += brainz.ops.sum(usize, &expected_labels);
    }

    return @as(f32, @as(f32, @floatFromInt(num_correct)) / @as(f32, @floatFromInt(total_batches)));
}
