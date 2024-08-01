const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const brainz = b.dependency("brainz", .{
        .optimize = .ReleaseFast,
        .target = target,
    });

    // code for model training
    const train = b.addExecutable(.{
        .name = "train",
        .root_source_file = b.path("src/train.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });

    // code for inference on web with WASM
    const webmnist = b.addExecutable(.{
        .name = "webmnist",
        .optimize = .ReleaseFast,
        .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
        .root_source_file = b.path("src/webmnist.zig"),
    });

    webmnist.rdynamic = true;
    webmnist.entry = .disabled;
    webmnist.export_memory = true;

    train.root_module.addImport("brainz", brainz.module("brainz"));
    webmnist.root_module.addImport("brainz", brainz.module("brainz"));

    b.installArtifact(train);

    // copy html + wasm to the web directory
    b.installDirectory(.{ .source_dir = b.path("public"), .install_dir = .{ .custom = "web" }, .install_subdir = "public" });
    b.getInstallStep().dependOn(&b.addInstallArtifact(webmnist, .{
        .dest_dir = .{
            .override = .{ .custom = "web" },
        },
        .dest_sub_path = "public/webmnist.wasm",
    }).step);

    const run_train_step = b.addRunArtifact(train);

    const run_train_cmd = b.step("train", "Runs model training");
    run_train_cmd.dependOn(&run_train_step.step);

    // run model training before compiling the WASM executable.
    webmnist.step.dependOn(&run_train_step.step);
}
