const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const brainz = b.dependency("brainz", .{
        .optimize = optimize,
        .target = target,
    });

    // code for model training
    const train_module = b.addModule("train_module", .{
        .root_source_file = b.path("src/train.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });

    const train_executable = b.addExecutable(.{
        .name = "train",
        .root_module = train_module,
    });

    // code for inference on web with WASM
    const webmnist = b.addExecutable(.{
        .name = "webmnist",
        .root_module = b.addModule("webmnist", .{
            .root_source_file = b.path("src/webmnist.zig"),
            .optimize = optimize,
            .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
        }),
    });

    webmnist.rdynamic = true;
    webmnist.entry = .disabled;
    webmnist.export_memory = true;

    // handle imports
    train_executable.root_module.addImport("brainz", brainz.module("brainz"));
    webmnist.root_module.addImport("brainz", brainz.module("brainz"));

    b.installArtifact(train_executable);

    // copy html + wasm to the web directory
    b.installDirectory(.{ .source_dir = b.path("public"), .install_dir = .{ .custom = "web" }, .install_subdir = "public" });
    b.getInstallStep().dependOn(&b.addInstallArtifact(webmnist, .{
        .dest_dir = .{
            .override = .{ .custom = "web" },
        },
        .dest_sub_path = "public/webmnist.wasm",
    }).step);

    // step to train model locally
    const run_train_cmd = b.step("train", "Runs model training");
    run_train_cmd.dependOn(&b.addRunArtifact(train_executable).step);

    // run model training if model checkpoints aren't found.
    std.fs.cwd().access("src/model.bin", .{}) catch |err| {
        std.log.warn("Failed to open base model checkpoints ({}). Running model training.", .{err});
        webmnist.step.dependOn(run_train_cmd);
    };
}
