// Copyright 2024, Lucas Arriesse (Game4all)
// Please see the license file at root of repository for more information.

const brainz = @import("brainz");
const std = @import("std");

const Linear = brainz.nn.Linear;
const Activation = brainz.nn.Activation;
const LinearPlan = brainz.LinearPlan;

const Allocator = std.mem.Allocator;
const Tensor = brainz.Tensor;

/// A MLP-based neural network for handwritten digit classification.
pub const MNISTClassifier = struct {
    const Self = @This();

    pub const IN_FEATURES = 784;
    pub const OUT_FEATURES = 10;

    layer_1: Linear(f32, true),
    act_1: Activation(.relu),
    layer_2: Linear(f32, true),
    act_2: Activation(.relu),
    layer_3: Linear(f32, true),
    act_3: Activation(.relu),
    layer_4: Linear(f32, true),

    pub fn init(plan: *LinearPlan) !Self {
        return .{
            .layer_1 = try .init(plan, IN_FEATURES, 256),
            .act_1 = .init,
            .layer_2 = try .init(plan, 256, 192),
            .act_2 = .init,
            .layer_3 = try .init(plan, 192, 48),
            .act_3 = .init,
            .layer_4 = try .init(plan, 48, OUT_FEATURES),
        };
    }

    /// Saves the model weights to the specified path.
    pub fn saveWeights(self: *const Self, writer: *std.Io.Writer) !void {
        inline for (.{ "layer_1", "layer_2", "layer_3", "layer_4" }) |layer_name| {
            const layer = @field(self, layer_name);
            const wSlice = layer.weights.slice(f32).?;
            const bSlice = layer.biases.?.slice(f32).?;

            try writer.writeAll(std.mem.sliceAsBytes(wSlice));
            try writer.writeAll(std.mem.sliceAsBytes(bSlice));
        }
    }

    /// Loads the model weights from the specified path.
    pub fn loadWeights(self: *Self, reader: *std.Io.Reader) !void {
        inline for (.{ "layer_1", "layer_2", "layer_3", "layer_4" }) |layer_name| {
            const layer = &@field(self, layer_name);
            const wSlice = layer.weights.slice(f32).?;
            const bSlice = layer.biases.?.slice(f32).?;

            try reader.readSliceAll(std.mem.sliceAsBytes(wSlice));
            try reader.readSliceAll(std.mem.sliceAsBytes(bSlice));
        }
    }
};
