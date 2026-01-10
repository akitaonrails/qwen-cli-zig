// chat.zig - Chat formatting and history management
//
// Handles Qwen3 chat format and conversation history.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// A single message in the conversation.
pub const Message = struct {
    role: Role,
    content: []const u8,

    pub const Role = enum {
        system,
        user,
        assistant,

        pub fn toString(self: Role) []const u8 {
            return switch (self) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
            };
        }
    };
};

/// Manages conversation history with automatic memory management.
pub const History = struct {
    messages: std.ArrayList(Message),
    allocator: Allocator,
    max_pairs: usize,

    const Self = @This();

    pub fn init(allocator: Allocator, max_pairs: usize) Self {
        return .{
            .messages = .{},
            .allocator = allocator,
            .max_pairs = max_pairs,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.messages.items) |msg| {
            self.allocator.free(msg.content);
        }
        self.messages.deinit(self.allocator);
    }

    /// Add a message to history, duplicating the content.
    pub fn add(self: *Self, role: Message.Role, content: []const u8) !void {
        const content_copy = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(content_copy);
        try self.messages.append(self.allocator, .{ .role = role, .content = content_copy });
    }

    /// Remove the last message (useful for error recovery).
    pub fn removeLast(self: *Self) void {
        if (self.messages.pop()) |msg| {
            self.allocator.free(msg.content);
        }
    }

    /// Get messages for prompt building, respecting max_pairs limit.
    pub fn getRecentMessages(self: *const Self) []const Message {
        const max_messages = self.max_pairs * 2;
        if (self.messages.items.len <= max_messages) {
            return self.messages.items;
        }
        return self.messages.items[self.messages.items.len - max_messages ..];
    }
};

/// Qwen3 chat format constants.
pub const Format = struct {
    pub const im_start = "<|im_start|>";
    pub const im_end = "<|im_end|>";

    /// Build a complete prompt from system prompt, history, and current user input.
    pub fn buildPrompt(
        allocator: Allocator,
        system_prompt: []const u8,
        history: *const History,
        user_input: []const u8,
    ) ![]const u8 {
        var builder: std.ArrayList(u8) = .{};
        errdefer builder.deinit(allocator);

        // System message
        try builder.appendSlice(allocator, im_start);
        try builder.appendSlice(allocator, "system\n");
        try builder.appendSlice(allocator, system_prompt);
        try builder.appendSlice(allocator, im_end);
        try builder.appendSlice(allocator, "\n");

        // History messages
        for (history.getRecentMessages()) |msg| {
            try builder.appendSlice(allocator, im_start);
            try builder.appendSlice(allocator, msg.role.toString());
            try builder.appendSlice(allocator, "\n");
            try builder.appendSlice(allocator, msg.content);
            try builder.appendSlice(allocator, im_end);
            try builder.appendSlice(allocator, "\n");
        }

        // Current user input
        try builder.appendSlice(allocator, im_start);
        try builder.appendSlice(allocator, "user\n");
        try builder.appendSlice(allocator, user_input);
        try builder.appendSlice(allocator, im_end);
        try builder.appendSlice(allocator, "\n");

        // Assistant start
        try builder.appendSlice(allocator, im_start);
        try builder.appendSlice(allocator, "assistant\n");

        return builder.toOwnedSlice(allocator);
    }
};

test "History basic operations" {
    const allocator = std.testing.allocator;
    var history = History.init(allocator, 3);
    defer history.deinit();

    try history.add(.user, "Hello");
    try history.add(.assistant, "Hi there!");

    try std.testing.expectEqual(@as(usize, 2), history.messages.items.len);
    try std.testing.expectEqualStrings("Hello", history.messages.items[0].content);
}

test "Format buildPrompt" {
    const allocator = std.testing.allocator;
    var history = History.init(allocator, 3);
    defer history.deinit();

    const prompt = try Format.buildPrompt(allocator, "You are helpful.", &history, "Hi");
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>system") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>user") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|im_start|>assistant") != null);
}
