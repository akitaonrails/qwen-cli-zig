const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe_module = b.createModule(.{
        .root_source_file = b.path("src/qwen_cli.zig"),
        .target = target,
        .optimize = optimize,
    });

    const llama_include_path = b.path("vendor/llama.cpp/include");
    const ggml_include_path = b.path("vendor/llama.cpp/ggml/include");
    const llama_lib_path = b.path("vendor/llama.cpp/build/bin");

    exe_module.addIncludePath(llama_include_path);
    exe_module.addIncludePath(ggml_include_path);
    exe_module.addLibraryPath(llama_lib_path);

    const exe = b.addExecutable(.{
        .name = "qwen_cli",
        .root_module = exe_module,
    });

    switch (target.result.os.tag) {
        .windows => {},
        else => {
            exe.root_module.addRPath(llama_lib_path);
            exe.root_module.linkSystemLibrary("llama", .{});
            exe.root_module.linkSystemLibrary("c", .{});
            exe.root_module.linkSystemLibrary("m", .{});
            exe.root_module.linkSystemLibrary("dl", .{});
            exe.root_module.linkSystemLibrary("pthread", .{});
            exe.root_module.linkSystemLibrary("cuda", .{});
            exe.root_module.linkSystemLibrary("cudart", .{});
            exe.root_module.linkSystemLibrary("cublas", .{});
        },
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);
}
