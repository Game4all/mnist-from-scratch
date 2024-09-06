// Copyright 2024, Lucas Arriesse (Game4all)
// Please see the license file at root of repository for more information.

const std = @import("std");
const brainz = @import("brainz");

const Tensor = brainz.Tensor;
const Allocator = std.mem.Allocator;
const Device = brainz.Device;

/// MNIST dataset
pub const MNISTDataset = struct {
    alloc: Allocator,

    /// dataset itself.
    label_data: []u8,
    image_data: []u8,

    batch_size: usize,

    // tensors
    f_inputs: Tensor(f32),
    label_encoded_tensor: Tensor(f32),

    pub const IMAGE_SIZE: comptime_int = 784;

    const Self = @This();

    pub fn init(alloc: Allocator, labels_path: [:0]const u8, images_path: [:0]const u8, network: anytype) !@This() {
        const labels = try load(alloc, labels_path);
        errdefer alloc.free(labels);

        const images = try load(alloc, images_path);
        errdefer alloc.free(images);

        return @This(){
            .alloc = alloc,
            .image_data = images,
            .batch_size = network.inputShape().@"0",
            .label_data = labels,
            .label_encoded_tensor = try Tensor(f32).init(network.outputShape(), alloc),
            .f_inputs = try Tensor(f32).init(network.inputShape(), alloc),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.alloc.free(self.image_data);
        self.alloc.free(self.label_data);

        self.label_encoded_tensor.deinit(self.alloc);
        self.f_inputs.deinit(self.alloc);
    }

    fn load(alloc: Allocator, path: [:0]const u8) ![]u8 {
        var file = try std.fs.cwd().openFile(path, .{});
        var reader = file.reader();
        defer file.close();

        _ = try reader.readInt(u16, .big); // magic bytes
        const ty = try reader.readInt(u8, .big);

        const data_type: usize = switch (ty) {
            0x08 => @sizeOf(u8),
            0x09 => @sizeOf(i8),
            0x0B => @sizeOf(i16),
            0x0C => @sizeOf(i32),
            0x0D => @sizeOf(f32),
            0x0E => @sizeOf(f64),
            else => return error.InvalidDataFormat,
        };

        const n_dims = try reader.readInt(u8, .big); // numbers of dimensions

        var total_size: usize = 1;
        for (0..n_dims) |_| // reading dimensions
            total_size *= try reader.readInt(u32, .big);

        total_size *= data_type;

        const data = try alloc.alloc(u8, total_size);
        errdefer alloc.free(data);

        _ = try reader.read(data);
        return data;
    }

    pub inline fn iterator(self: *@This()) BatchIterator {
        return .{
            .pos = 0,
            .ptr = self,
        };
    }

    const BatchIterator = struct {
        pos: usize,
        ptr: *Self,

        pub fn nextBatchTraining(self: *@This(), dev: Device) !?struct { *const Tensor(f32), *const Tensor(f32) } {
            if (self.pos < self.ptr.image_data.len) {
                // preprocess and load the inputs
                const inputU = try Tensor(u8).initFromSlice(self.ptr.f_inputs.shape, self.ptr.image_data[self.pos..(self.pos + IMAGE_SIZE * self.ptr.batch_size)]);

                try brainz.ops.cast(u8, f32, dev, &inputU, &self.ptr.f_inputs);
                try dev.barrier();

                try brainz.ops.mulScalar(f32, dev, &self.ptr.f_inputs, 1.0 / 256.0, &self.ptr.f_inputs);
                try dev.barrier();

                // one hot encoding
                encodeOneHot(f32, self.ptr.label_data[(self.pos / IMAGE_SIZE)..((self.pos / IMAGE_SIZE) + self.ptr.batch_size)], &self.ptr.label_encoded_tensor);

                self.pos += IMAGE_SIZE * self.ptr.batch_size;

                return .{ &self.ptr.f_inputs, &self.ptr.label_encoded_tensor };
            }

            return null;
        }

        pub fn nextBatchEval(self: *@This(), dev: Device) !?struct { *const Tensor(f32), Tensor(u8) } {
            if (self.pos < self.ptr.image_data.len) {
                // preprocess and load the inputs
                const inputU = try Tensor(u8).initFromSlice(self.ptr.f_inputs.shape, self.ptr.image_data[self.pos..(self.pos + IMAGE_SIZE * self.ptr.batch_size)]);
                try brainz.ops.cast(u8, f32, dev, &inputU, &self.ptr.f_inputs);
                try dev.barrier();

                try brainz.ops.mulScalar(f32, dev, &self.ptr.f_inputs, 1.0 / 256.0, &self.ptr.f_inputs);
                try dev.barrier();

                const labels = try Tensor(u8).initFromSlice(.{ self.ptr.batch_size, 0, 1 }, self.ptr.label_data[(self.pos / IMAGE_SIZE)..((self.pos / IMAGE_SIZE) + self.ptr.batch_size)]);

                self.pos += IMAGE_SIZE * self.ptr.batch_size;

                return .{ &self.ptr.f_inputs, labels };
            }

            return null;
        }
    };
};

fn encodeOneHot(comptime ty: type, indices: []const u8, result: *Tensor(ty)) void {
    std.debug.assert(indices.len == result.shape.@"0");
    result.fill(0);

    for (indices, 0..) |value, i|
        result.set(.{ i, value, 0 }, 1);
}
