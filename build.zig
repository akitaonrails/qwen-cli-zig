const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Paths for llama.cpp (built by build.sh)
    const llama_include = b.path("vendor/llama.cpp/include");
    const llama_lib = b.path("vendor/llama.cpp/build/bin");

    // Create root module
    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add include paths
    root_module.addIncludePath(llama_include);

    // Main executable
    const exe = b.addExecutable(.{
        .name = "qwen_cli",
        .root_module = root_module,
    });

    // Library paths and linking
    exe.addLibraryPath(llama_lib);

    if (target.result.os.tag != .windows) {
        // Unix: RPATH for runtime library discovery
        exe.addRPath(llama_lib);

        // Link llama.cpp and system libraries
        exe.linkSystemLibrary("llama");
        exe.linkSystemLibrary("c");
        exe.linkSystemLibrary("m");
        exe.linkSystemLibrary("dl");
        exe.linkSystemLibrary("pthread");

        // CUDA libraries (when built with CUDA support)
        exe.linkSystemLibrary("cuda");
        exe.linkSystemLibrary("cudart");
        exe.linkSystemLibrary("cublas");
    }

    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);

    // Unit tests
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/chat.zig"),
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .root_module = test_module,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
