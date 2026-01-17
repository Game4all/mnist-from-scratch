// Copyright 2024, Lucas Arriesse (Game4all)
// Please see the license file at root of repository for more information.

const std = @import("std");
const brainz = @import("brainz");

const Allocator = std.mem.Allocator;

/// A reader for the MNIST Handwritten digits dataset.
pub const MNISTDataset = struct {
    alloc: Allocator,

    /// labels
    label_data: []usize,
    /// raw image data
    image_data: []f32,
    /// total number of images in the split
    count_images: usize,

    /// The size of an image
    pub const IMAGE_SIZE: comptime_int = 784;
    pub const MAX_LABEL_TYPES = 10; // 0 to 9

    const Self = @This();

    /// Loads a split of the dataset from disk from the given labels and images path.
    pub fn init(alloc: Allocator, labels_path: [:0]const u8, images_path: [:0]const u8) !@This() {
        // ensuring the files are accessible
        try std.fs.cwd().access(labels_path, .{ .mode = .read_only });
        try std.fs.cwd().access(images_path, .{ .mode = .read_only });

        // load both labels and image data from disk
        const labels_u8 = try load(alloc, labels_path);
        defer alloc.free(labels_u8);

        const images_u8 = try load(alloc, images_path);
        defer alloc.free(images_u8);

        const label_data = try alloc.alloc(usize, labels_u8.len);
        errdefer alloc.free(label_data);

        for (labels_u8, 0..) |label, i|
            label_data[i] = @as(usize, label);

        const image_data = try alloc.alloc(f32, images_u8.len);
        errdefer alloc.free(image_data);
        for (images_u8, 0..) |pixel, i|
            image_data[i] = @as(f32, @floatFromInt(pixel)) / 255.0;

        const image_count = image_data.len / Self.IMAGE_SIZE;

        std.debug.assert(image_count == label_data.len);

        return @This(){
            .alloc = alloc,
            .label_data = label_data,
            .image_data = image_data,
            .count_images = image_count,
        };
    }

    pub fn deinit(self: *Self) void {
        self.alloc.free(self.image_data);
        self.alloc.free(self.label_data);
    }

    fn load(alloc: Allocator, path: [:0]const u8) ![]u8 {
        var file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        var buffer: [1024]u8 = undefined;
        var fileReader = file.reader(&buffer);
        var reader = &fileReader.interface;

        // tossing header magic bytes
        const headerMagic = try reader.peekInt(u16, .big);
        reader.toss(@sizeOf(u16));

        if (headerMagic != 0) return error.HeaderCheckFailed;

        // reading the data format (0x08 = unsigned byte)
        const dataType = try reader.peekInt(u8, .big);
        reader.toss(@sizeOf(u8));

        if (dataType != 0x08) return error.UnsupportedDataType;

        const nDims = try reader.peekInt(u8, .big);
        reader.toss(@sizeOf(u8));

        var totalSize: usize = 1;
        for (0..nDims) |_| {
            totalSize *= try reader.peekInt(u32, .big);
            reader.toss(@sizeOf(u32));
        }

        const data = try reader.readAlloc(alloc, totalSize);

        return data;
    }

    pub const Iterator = struct {
        dataset: *const MNISTDataset,
        batch_size: usize,
        pos: usize = 0,

        /// Copies the image data to the specified slice.
        pub fn copyImageData(self: *const Iterator, dest: []f32) void {
            std.debug.assert(dest.len == self.batch_size * MNISTDataset.IMAGE_SIZE);
            if (self.pos >= self.dataset.label_data.len) return;
            const end = @min(self.pos + self.batch_size, self.dataset.label_data.len);
            const images = self.dataset.image_data[self.pos * IMAGE_SIZE .. end * IMAGE_SIZE];
            @memcpy(dest, images);
        }

        /// Returns the labels for the current iteration.
        pub fn getLabels(self: *const Iterator) ?[]const usize {
            if (self.pos >= self.dataset.label_data.len) return null;
            const end = @min(self.pos + self.batch_size, self.dataset.label_data.len);
            const labels = self.dataset.label_data[self.pos..end];
            return labels;
        }

        /// Copies the one hot encoded label data to the specified slice.
        pub fn copyLabelData(self: *const Iterator, dest: []f32) void {
            std.debug.assert(dest.len == self.batch_size * MNISTDataset.MAX_LABEL_TYPES);
            const labelData = self.getLabels() orelse return;
            @memset(dest, 0);
            for (labelData, 0..) |label, idx|
                dest[idx * MNISTDataset.MAX_LABEL_TYPES + label] = 1.0;
        }

        pub fn next(self: *Iterator) bool {
            if (self.pos >= self.dataset.label_data.len) return false;

            const end = @min(self.pos + self.batch_size, self.dataset.label_data.len);
            const actual_batch_size = end - self.pos;

            self.pos += actual_batch_size;

            return true;
        }
    };

    /// Returns a dataset iterator with the given batch size.
    pub fn iterator(self: *const Self, batch_size: usize) Iterator {
        return .{
            .dataset = self,
            .batch_size = batch_size,
        };
    }
};
