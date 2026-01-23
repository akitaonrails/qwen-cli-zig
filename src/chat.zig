const std = @import("std");
const llama = @import("llama.zig");

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

pub const ChatHistory = struct {
    messages: std.ArrayListUnmanaged(Message),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ChatHistory {
        return .{
            .messages = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ChatHistory) void {
        for (self.messages.items) |msg| {
            // Only free content, roles are static strings
            self.allocator.free(msg.content);
        }
        self.messages.deinit(self.allocator);
    }

    pub fn append(self: *ChatHistory, role: []const u8, content: []const u8) !void {
        const content_copy = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(content_copy);
        try self.messages.append(self.allocator, .{ .role = role, .content = content_copy });
    }
};

// Helper to convert Zig slice to null-terminated C string
pub fn toCString(allocator: std.mem.Allocator, slice: []const u8) ![:0]const u8 {
    const c_str = try allocator.allocSentinel(u8, slice.len, 0);
    @memcpy(c_str[0..slice.len], slice);
    return c_str;
}

// Temporary C string storage for building chat messages
pub const TempCStrings = struct {
    strings: std.ArrayListUnmanaged([:0]const u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TempCStrings {
        return .{
            .strings = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TempCStrings) void {
        for (self.strings.items) |s| {
            self.allocator.free(s);
        }
        self.strings.deinit(self.allocator);
    }

    pub fn add(self: *TempCStrings, slice: []const u8) ![:0]const u8 {
        const c_str = try toCString(self.allocator, slice);
        errdefer self.allocator.free(c_str);
        try self.strings.append(self.allocator, c_str);
        return c_str;
    }
};

// Build llama_chat_message array from history
pub fn buildChatMessages(
    allocator: std.mem.Allocator,
    system_prompt: []const u8,
    history: *const ChatHistory,
    user_input: []const u8,
    temp_strings: *TempCStrings,
) !std.ArrayListUnmanaged(llama.llama_chat_message) {
    var chat_messages = std.ArrayListUnmanaged(llama.llama_chat_message){};
    errdefer chat_messages.deinit(allocator);

    // Add system prompt
    const c_system_role = try temp_strings.add("system");
    const c_system_prompt = try temp_strings.add(system_prompt);
    try chat_messages.append(allocator, .{ .role = c_system_role, .content = c_system_prompt });

    // Add history messages
    for (history.messages.items) |msg| {
        const c_role = try temp_strings.add(msg.role);
        const c_content = try temp_strings.add(msg.content);
        try chat_messages.append(allocator, .{ .role = c_role, .content = c_content });
    }

    // Add current user input
    const c_user_role = try temp_strings.add("user");
    const c_user_input = try temp_strings.add(user_input);
    try chat_messages.append(allocator, .{ .role = c_user_role, .content = c_user_input });

    return chat_messages;
}

// Apply chat template and return formatted prompt
pub fn applyTemplate(
    allocator: std.mem.Allocator,
    model: ?*const llama.llama_model,
    chat_messages: []const llama.llama_chat_message,
) ![]u8 {
    // Get the model's chat template
    const tmpl = llama.llama_model_chat_template(model, null);

    var formatted_buf = std.ArrayListUnmanaged(u8){};
    defer formatted_buf.deinit(allocator);

    const initial_size: usize = 2048;
    try formatted_buf.resize(allocator, initial_size);

    var required_len = llama.llama_chat_apply_template(
        tmpl,
        chat_messages.ptr,
        chat_messages.len,
        true, // add_ass
        formatted_buf.items.ptr,
        @intCast(initial_size),
    );

    if (required_len < 0) {
        return error.TemplateError;
    }

    const required_usize: usize = @intCast(required_len);
    if (required_usize > initial_size) {
        try formatted_buf.resize(allocator, required_usize);
        required_len = llama.llama_chat_apply_template(
            tmpl,
            chat_messages.ptr,
            chat_messages.len,
            true,
            formatted_buf.items.ptr,
            required_len,
        );
        if (required_len < 0) {
            return error.TemplateError;
        }
    }

    // Return owned slice of the actual content
    const result = try allocator.alloc(u8, @intCast(required_len));
    @memcpy(result, formatted_buf.items[0..@intCast(required_len)]);
    return result;
}
