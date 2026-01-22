const std = @import("std");

pub fn build(b: *std.Build) anyerror!void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "qwen_cli",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/qwen_cli.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // --- Llama.cpp Configuration ---
    // Paths are hardcoded assuming build.sh has run successfully
    // and created the vendor/llama.cpp directory structure.
    const llama_include_path = "vendor/llama.cpp/include"; // Correct directory containing llama.h
    // *** Path for shared library ***
    const llama_lib_path = "vendor/llama.cpp/build/bin"; // Directory containing libllama.so

    // Add include path for C headers to the executable step
    exe.addIncludePath(b.path(llama_include_path));

    // Keep addLibraryPath - it might help the linker find dependencies
    exe.addLibraryPath(b.path(llama_lib_path)); // Keep for dependencies if needed

    // --- Linking ---
    // Remove linkSystemLibrary("llama") and use explicit linker flags instead


    // --- System Libraries, RPATH, and Linker Flags ---
    if (target.result.os.tag == .windows) {
        // Windows specific system libraries might be needed depending on llama.cpp build
        // exe.linkSystemLibrary("kernel32"); // Example
        // RPATH is not typically used on Windows or for static linking
    } else {
        // Add RPATH so the executable knows where to find libllama.so and its deps at runtime
        exe.addRPath(b.path(llama_lib_path));

        // Link the shared llama library. addLibraryPath tells the linker where to find it.
        exe.linkSystemLibrary("llama");

        // Link essential system libraries (still needed by llama.cpp code)
        exe.linkSystemLibrary("c");
        exe.linkSystemLibrary("m");
        exe.linkSystemLibrary("dl"); // Often needed for dynamic linking features used by C libs
        exe.linkSystemLibrary("pthread"); // Often needed by llama.cpp

        // Add other necessary system libraries depending on llama.cpp build options (e.g., cuda, rocm)
        // Link CUDA libraries since we enabled it in build.sh
        // Ensure CUDA toolkit is installed and these libs are findable by the system linker
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
