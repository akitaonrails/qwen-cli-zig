const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/qwen_cli.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "qwen_cli",
        .root_module = root_module,
    });

    // --- Llama.cpp Configuration ---
    // Paths are hardcoded assuming build.sh has run successfully
    // and created the vendor/llama.cpp directory structure.
    const llama_include_path = "vendor/llama.cpp/include"; // Directory containing llama.h
    const llama_lib_path = "vendor/llama.cpp/build/bin"; // Directory containing libllama.so

    exe.root_module.addIncludePath(b.path(llama_include_path));
    exe.root_module.addLibraryPath(b.path(llama_lib_path));

    // --- System Libraries, RPATH, and Linker Flags ---
    if (target.result.os.tag == .windows) {
        // Windows specific system libraries might be needed depending on llama.cpp build
        // RPATH is not typically used on Windows
    } else {
        exe.linker_enable_new_dtags = false;

        // Add RPATH so the executable knows where to find libllama.so and its deps at runtime
        exe.root_module.addRPath(b.path(llama_lib_path));

        exe.addObjectFile(b.path(llama_lib_path ++ "/libllama.so.0"));
        exe.addObjectFile(b.path(llama_lib_path ++ "/libggml.so.0"));
        exe.addObjectFile(b.path(llama_lib_path ++ "/libggml-base.so.0"));
        exe.addObjectFile(b.path(llama_lib_path ++ "/libggml-cpu.so.0"));
        exe.addObjectFile(b.path(llama_lib_path ++ "/libggml-cuda.so.0"));

        // Link essential system libraries (still needed by llama.cpp code)
        exe.linkSystemLibrary("c");
        exe.linkSystemLibrary("m");
        exe.linkSystemLibrary("dl");
        exe.linkSystemLibrary("pthread");

        // Link CUDA libraries since build.sh enables CUDA.
        exe.linkSystemLibrary("cuda");
        exe.linkSystemLibrary("cudart");
        exe.linkSystemLibrary("cublas");
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
