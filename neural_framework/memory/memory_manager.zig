const std = @import("std");
const builtin = @import("builtin");

pub const AllocationError = error{
    OutOfMemory,
    InvalidAlignment,
    InvalidSize,
    DoubleFree,
    UseAfterFree,
    BufferOverflow,
    AllocationNotFound,
    PoolExhausted,
    ArenaFull,
    InvalidPointer,
    CorruptedMetadata,
    AlignmentMismatch,
    SizeMismatch,
    LeakDetected,
};

pub const AllocationFlags = packed struct {
    zeroed: bool = false,
    pinned: bool = false,
    executable: bool = false,
    read_only: bool = false,
    no_cache: bool = false,
    huge_pages: bool = false,
    locked: bool = false,
    shared: bool = false,
    reserved1: u8 = 0,
    reserved2: u16 = 0,
};

pub const AllocationMetadata = struct {
    address: usize,
    size: usize,
    alignment: usize,
    timestamp: i64,
    flags: AllocationFlags,
    stack_trace: [32]usize,
    stack_depth: usize,
    thread_id: u64,
    allocation_id: u64,
    source_file: [256]u8,
    source_line: u32,
    source_column: u32,
    tag: [64]u8,
    previous: ?*AllocationMetadata,
    next: ?*AllocationMetadata,
    checksum: u64,
};

pub const MemoryPool = struct {
    base_address: [*]u8,
    total_size: usize,
    block_size: usize,
    block_count: usize,
    free_bitmap: []u64,
    allocation_count: usize,
    peak_allocation: usize,
    total_allocated: usize,
    total_freed: usize,
    lock: std.Thread.Mutex,
    
    pub fn init(allocator: std.mem.Allocator, block_size: usize, block_count: usize) !*MemoryPool {
        const total_size = block_size * block_count;
        const bitmap_size = (block_count + 63) / 64;
        
        const pool = try allocator.create(MemoryPool);
        errdefer allocator.destroy(pool);
        
        const memory = try allocator.alignedAlloc(u8, 4096, total_size);
        errdefer allocator.free(memory);
        
        const bitmap = try allocator.alloc(u64, bitmap_size);
        @memset(bitmap, 0);
        
        pool.* = MemoryPool{
            .base_address = memory.ptr,
            .total_size = total_size,
            .block_size = block_size,
            .block_count = block_count,
            .free_bitmap = bitmap,
            .allocation_count = 0,
            .peak_allocation = 0,
            .total_allocated = 0,
            .total_freed = 0,
            .lock = std.Thread.Mutex{},
        };
        
        return pool;
    }
    
    pub fn allocate(self: *MemoryPool) !*anyopaque {
        self.lock.lock();
        defer self.lock.unlock();
        
        for (self.free_bitmap, 0..) |*word, word_idx| {
            if (word.* != ~@as(u64, 0)) {
                const bit_idx = @ctz(~word.*);
                const block_idx = word_idx * 64 + bit_idx;
                
                if (block_idx >= self.block_count) {
                    return AllocationError.PoolExhausted;
                }
                
                word.* |= @as(u64, 1) << @intCast(bit_idx);
                
                self.allocation_count += 1;
                self.total_allocated += self.block_size;
                
                if (self.allocation_count > self.peak_allocation) {
                    self.peak_allocation = self.allocation_count;
                }
                
                const offset = block_idx * self.block_size;
                return @ptrCast(self.base_address + offset);
            }
        }
        
        return AllocationError.PoolExhausted;
    }
    
    pub fn deallocate(self: *MemoryPool, ptr: *anyopaque) !void {
        self.lock.lock();
        defer self.lock.unlock();
        
        const address = @intFromPtr(ptr);
        const base = @intFromPtr(self.base_address);
        
        if (address < base or address >= base + self.total_size) {
            return AllocationError.InvalidPointer;
        }
        
        const offset = address - base;
        if (offset % self.block_size != 0) {
            return AllocationError.InvalidPointer;
        }
        
        const block_idx = offset / self.block_size;
        const word_idx = block_idx / 64;
        const bit_idx: u6 = @intCast(block_idx % 64);
        
        if ((self.free_bitmap[word_idx] & (@as(u64, 1) << bit_idx)) == 0) {
            return AllocationError.DoubleFree;
        }
        
        self.free_bitmap[word_idx] &= ~(@as(u64, 1) << bit_idx);
        self.allocation_count -= 1;
        self.total_freed += self.block_size;
    }
    
    pub fn deinit(self: *MemoryPool, allocator: std.mem.Allocator) void {
        const slice = self.base_address[0..self.total_size];
        allocator.free(slice);
        allocator.free(self.free_bitmap);
        allocator.destroy(self);
    }
};

pub const Arena = struct {
    buffer: []u8,
    offset: usize,
    capacity: usize,
    alignment: usize,
    allocation_count: usize,
    parent_allocator: std.mem.Allocator,
    child_arenas: std.ArrayList(*Arena),
    markers: std.ArrayList(usize),
    
    pub fn init(parent: std.mem.Allocator, capacity: usize) !*Arena {
        const arena = try parent.create(Arena);
        errdefer parent.destroy(arena);
        
        const buffer = try parent.alloc(u8, capacity);
        errdefer parent.free(buffer);
        
        arena.* = Arena{
            .buffer = buffer,
            .offset = 0,
            .capacity = capacity,
            .alignment = 16,
            .allocation_count = 0,
            .parent_allocator = parent,
            .child_arenas = std.ArrayList(*Arena).init(parent),
            .markers = std.ArrayList(usize).init(parent),
        };
        
        return arena;
    }
    
    pub fn alloc(self: *Arena, comptime T: type, count: usize) ![]T {
        const size = @sizeOf(T) * count;
        const alignment = @alignOf(T);
        
        const aligned_offset = (self.offset + alignment - 1) & ~(alignment - 1);
        
        if (aligned_offset + size > self.capacity) {
            return AllocationError.ArenaFull;
        }
        
        const ptr: [*]T = @ptrCast(@alignCast(self.buffer.ptr + aligned_offset));
        self.offset = aligned_offset + size;
        self.allocation_count += 1;
        
        return ptr[0..count];
    }
    
    pub fn allocBytes(self: *Arena, size: usize, alignment: usize) ![]u8 {
        const aligned_offset = (self.offset + alignment - 1) & ~(alignment - 1);
        
        if (aligned_offset + size > self.capacity) {
            return AllocationError.ArenaFull;
        }
        
        const result = self.buffer[aligned_offset..aligned_offset + size];
        self.offset = aligned_offset + size;
        self.allocation_count += 1;
        
        return result;
    }
    
    pub fn pushMarker(self: *Arena) !usize {
        const marker = self.offset;
        try self.markers.append(marker);
        return marker;
    }
    
    pub fn popMarker(self: *Arena) !void {
        if (self.markers.items.len == 0) {
            return AllocationError.InvalidPointer;
        }
        
        const marker = self.markers.pop();
        self.offset = marker;
    }
    
    pub fn reset(self: *Arena) void {
        self.offset = 0;
        self.allocation_count = 0;
        self.markers.clearRetainingCapacity();
    }
    
    pub fn createChild(self: *Arena, capacity: usize) !*Arena {
        const child = try Arena.init(self.parent_allocator, capacity);
        try self.child_arenas.append(child);
        return child;
    }
    
    pub fn deinit(self: *Arena) void {
        for (self.child_arenas.items) |child| {
            child.deinit();
        }
        self.child_arenas.deinit();
        self.markers.deinit();
        self.parent_allocator.free(self.buffer);
        self.parent_allocator.destroy(self);
    }
};

pub const SlabAllocator = struct {
    slabs: [16]*MemoryPool,
    size_classes: [16]usize,
    fallback_allocator: std.mem.Allocator,
    total_allocations: u64,
    total_deallocations: u64,
    cache_hits: u64,
    cache_misses: u64,
    
    pub fn init(allocator: std.mem.Allocator) !*SlabAllocator {
        const slab_alloc = try allocator.create(SlabAllocator);
        errdefer allocator.destroy(slab_alloc);
        
        const size_classes = [16]usize{
            16, 32, 64, 128, 256, 512, 1024, 2048,
            4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
        };
        
        slab_alloc.size_classes = size_classes;
        slab_alloc.fallback_allocator = allocator;
        slab_alloc.total_allocations = 0;
        slab_alloc.total_deallocations = 0;
        slab_alloc.cache_hits = 0;
        slab_alloc.cache_misses = 0;
        
        for (&slab_alloc.slabs, 0..) |*slab, i| {
            const block_count = 1024 * 1024 / size_classes[i];
            slab.* = try MemoryPool.init(allocator, size_classes[i], block_count);
        }
        
        return slab_alloc;
    }
    
    fn findSizeClass(self: *SlabAllocator, size: usize) ?usize {
        for (self.size_classes, 0..) |class_size, i| {
            if (size <= class_size) {
                return i;
            }
        }
        return null;
    }
    
    pub fn allocate(self: *SlabAllocator, size: usize) !*anyopaque {
        self.total_allocations += 1;
        
        if (self.findSizeClass(size)) |class_idx| {
            self.cache_hits += 1;
            return self.slabs[class_idx].allocate();
        }
        
        self.cache_misses += 1;
        const memory = try self.fallback_allocator.alloc(u8, size);
        return @ptrCast(memory.ptr);
    }
    
    pub fn deallocate(self: *SlabAllocator, ptr: *anyopaque, size: usize) !void {
        self.total_deallocations += 1;
        
        if (self.findSizeClass(size)) |class_idx| {
            return self.slabs[class_idx].deallocate(ptr);
        }
        
        const slice: [*]u8 = @ptrCast(ptr);
        self.fallback_allocator.free(slice[0..size]);
    }
    
    pub fn deinit(self: *SlabAllocator) void {
        for (&self.slabs) |slab| {
            slab.deinit(self.fallback_allocator);
        }
        self.fallback_allocator.destroy(self);
    }
};

pub const AllocationTracker = struct {
    allocations: std.AutoHashMap(usize, AllocationMetadata),
    allocation_order: std.ArrayList(usize),
    next_id: u64,
    total_bytes_allocated: usize,
    total_bytes_freed: usize,
    peak_bytes: usize,
    current_bytes: usize,
    allocation_count: usize,
    deallocation_count: usize,
    lock: std.Thread.Mutex,
    parent_allocator: std.mem.Allocator,
    leak_check_enabled: bool,
    bounds_check_enabled: bool,
    
    pub fn init(allocator: std.mem.Allocator) !*AllocationTracker {
        const tracker = try allocator.create(AllocationTracker);
        
        tracker.* = AllocationTracker{
            .allocations = std.AutoHashMap(usize, AllocationMetadata).init(allocator),
            .allocation_order = std.ArrayList(usize).init(allocator),
            .next_id = 0,
            .total_bytes_allocated = 0,
            .total_bytes_freed = 0,
            .peak_bytes = 0,
            .current_bytes = 0,
            .allocation_count = 0,
            .deallocation_count = 0,
            .lock = std.Thread.Mutex{},
            .parent_allocator = allocator,
            .leak_check_enabled = true,
            .bounds_check_enabled = true,
        };
        
        return tracker;
    }
    
    fn captureStackTrace(self: *AllocationTracker) struct { trace: [32]usize, depth: usize } {
        _ = self;
        var trace: [32]usize = [_]usize{0} ** 32;
        var depth: usize = 0;
        
        var addresses: [32]usize = undefined;
        const captured = std.debug.captureStackTrace(@returnAddress(), &addresses, &trace, 32);
        depth = captured;
        
        return .{ .trace = trace, .depth = depth };
    }
    
    fn computeChecksum(meta: *const AllocationMetadata) u64 {
        var hash: u64 = 0xcbf29ce484222325;
        
        const bytes = std.mem.asBytes(meta);
        for (bytes[0..@offsetOf(AllocationMetadata, "checksum")]) |byte| {
            hash ^= byte;
            hash *%= 0x100000001b3;
        }
        
        return hash;
    }
    
    pub fn trackAllocation(self: *AllocationTracker, address: usize, size: usize, alignment: usize) !void {
        self.lock.lock();
        defer self.lock.unlock();
        
        const stack_info = self.captureStackTrace();
        
        var meta = AllocationMetadata{
            .address = address,
            .size = size,
            .alignment = alignment,
            .timestamp = std.time.timestamp(),
            .flags = AllocationFlags{},
            .stack_trace = stack_info.trace,
            .stack_depth = stack_info.depth,
            .thread_id = std.Thread.getCurrentId(),
            .allocation_id = self.next_id,
            .source_file = [_]u8{0} ** 256,
            .source_line = 0,
            .source_column = 0,
            .tag = [_]u8{0} ** 64,
            .previous = null,
            .next = null,
            .checksum = 0,
        };
        
        meta.checksum = computeChecksum(&meta);
        
        try self.allocations.put(address, meta);
        try self.allocation_order.append(address);
        
        self.next_id += 1;
        self.allocation_count += 1;
        self.total_bytes_allocated += size;
        self.current_bytes += size;
        
        if (self.current_bytes > self.peak_bytes) {
            self.peak_bytes = self.current_bytes;
        }
    }
    
    pub fn trackDeallocation(self: *AllocationTracker, address: usize) !AllocationMetadata {
        self.lock.lock();
        defer self.lock.unlock();
        
        const meta = self.allocations.get(address) orelse {
            return AllocationError.AllocationNotFound;
        };
        
        if (meta.checksum != computeChecksum(&meta)) {
            return AllocationError.CorruptedMetadata;
        }
        
        _ = self.allocations.remove(address);
        
        for (self.allocation_order.items, 0..) |addr, i| {
            if (addr == address) {
                _ = self.allocation_order.orderedRemove(i);
                break;
            }
        }
        
        self.deallocation_count += 1;
        self.total_bytes_freed += meta.size;
        self.current_bytes -= meta.size;
        
        return meta;
    }
    
    pub fn checkForLeaks(self: *AllocationTracker) ![]AllocationMetadata {
        self.lock.lock();
        defer self.lock.unlock();
        
        if (self.allocations.count() == 0) {
            return &[_]AllocationMetadata{};
        }
        
        var leaks = try self.parent_allocator.alloc(AllocationMetadata, self.allocations.count());
        var idx: usize = 0;
        
        var iter = self.allocations.iterator();
        while (iter.next()) |entry| {
            leaks[idx] = entry.value_ptr.*;
            idx += 1;
        }
        
        return leaks;
    }
    
    pub fn getStatistics(self: *AllocationTracker) struct {
        total_allocated: usize,
        total_freed: usize,
        current: usize,
        peak: usize,
        allocation_count: usize,
        deallocation_count: usize,
        active_allocations: usize,
    } {
        self.lock.lock();
        defer self.lock.unlock();
        
        return .{
            .total_allocated = self.total_bytes_allocated,
            .total_freed = self.total_bytes_freed,
            .current = self.current_bytes,
            .peak = self.peak_bytes,
            .allocation_count = self.allocation_count,
            .deallocation_count = self.deallocation_count,
            .active_allocations = self.allocations.count(),
        };
    }
    
    pub fn deinit(self: *AllocationTracker) void {
        self.allocations.deinit();
        self.allocation_order.deinit();
        self.parent_allocator.destroy(self);
    }
};

pub const BuddyAllocator = struct {
    memory: []u8,
    min_block_size: usize,
    max_order: usize,
    free_lists: []std.ArrayList(usize),
    block_orders: []u8,
    parent_allocator: std.mem.Allocator,
    total_size: usize,
    allocated_size: usize,
    
    pub fn init(allocator: std.mem.Allocator, total_size: usize, min_block_size: usize) !*BuddyAllocator {
        const buddy = try allocator.create(BuddyAllocator);
        errdefer allocator.destroy(buddy);
        
        var max_order: usize = 0;
        var size = min_block_size;
        while (size < total_size) {
            size *= 2;
            max_order += 1;
        }
        
        const memory = try allocator.alignedAlloc(u8, 4096, total_size);
        errdefer allocator.free(memory);
        
        const block_count = total_size / min_block_size;
        const block_orders = try allocator.alloc(u8, block_count);
        @memset(block_orders, 0xFF);
        
        const free_lists = try allocator.alloc(std.ArrayList(usize), max_order + 1);
        for (free_lists) |*list| {
            list.* = std.ArrayList(usize).init(allocator);
        }
        
        try free_lists[max_order].append(0);
        block_orders[0] = @intCast(max_order);
        
        buddy.* = BuddyAllocator{
            .memory = memory,
            .min_block_size = min_block_size,
            .max_order = max_order,
            .free_lists = free_lists,
            .block_orders = block_orders,
            .parent_allocator = allocator,
            .total_size = total_size,
            .allocated_size = 0,
        };
        
        return buddy;
    }
    
    fn getOrder(self: *BuddyAllocator, size: usize) usize {
        var order: usize = 0;
        var block_size = self.min_block_size;
        
        while (block_size < size and order < self.max_order) {
            block_size *= 2;
            order += 1;
        }
        
        return order;
    }
    
    fn getBlockIndex(self: *BuddyAllocator, offset: usize) usize {
        return offset / self.min_block_size;
    }
    
    fn getBlockOffset(self: *BuddyAllocator, index: usize) usize {
        return index * self.min_block_size;
    }
    
    fn getBuddyIndex(self: *BuddyAllocator, index: usize, order: usize) usize {
        const blocks_at_order = @as(usize, 1) << @intCast(order);
        return index ^ blocks_at_order;
    }
    
    pub fn allocate(self: *BuddyAllocator, size: usize) ![]u8 {
        const order = self.getOrder(size);
        
        var current_order = order;
        while (current_order <= self.max_order) : (current_order += 1) {
            if (self.free_lists[current_order].items.len > 0) {
                const block_index = self.free_lists[current_order].pop();
                
                while (current_order > order) {
                    current_order -= 1;
                    const buddy_index = self.getBuddyIndex(block_index, current_order);
                    try self.free_lists[current_order].append(buddy_index);
                    self.block_orders[buddy_index] = @intCast(current_order);
                }
                
                self.block_orders[block_index] = @intCast(order);
                const block_size = self.min_block_size << @intCast(order);
                self.allocated_size += block_size;
                
                const offset = self.getBlockOffset(block_index);
                return self.memory[offset..offset + block_size];
            }
        }
        
        return AllocationError.OutOfMemory;
    }
    
    pub fn deallocate(self: *BuddyAllocator, ptr: []u8) !void {
        const offset = @intFromPtr(ptr.ptr) - @intFromPtr(self.memory.ptr);
        var block_index = self.getBlockIndex(offset);
        var order: usize = self.block_orders[block_index];
        
        const block_size = self.min_block_size << @intCast(order);
        self.allocated_size -= block_size;
        
        while (order < self.max_order) {
            const buddy_index = self.getBuddyIndex(block_index, order);
            
            if (buddy_index >= self.block_orders.len) break;
            if (self.block_orders[buddy_index] != order) break;
            
            var found = false;
            for (self.free_lists[order].items, 0..) |idx, i| {
                if (idx == buddy_index) {
                    _ = self.free_lists[order].orderedRemove(i);
                    found = true;
                    break;
                }
            }
            
            if (!found) break;
            
            self.block_orders[buddy_index] = 0xFF;
            block_index = @min(block_index, buddy_index);
            order += 1;
        }
        
        try self.free_lists[order].append(block_index);
        self.block_orders[block_index] = @intCast(order);
    }
    
    pub fn deinit(self: *BuddyAllocator) void {
        for (self.free_lists) |*list| {
            list.deinit();
        }
        self.parent_allocator.free(self.free_lists);
        self.parent_allocator.free(self.block_orders);
        self.parent_allocator.free(self.memory);
        self.parent_allocator.destroy(self);
    }
};

pub const TensorMemoryManager = struct {
    arena: *Arena,
    pool_small: *MemoryPool,
    pool_medium: *MemoryPool,
    pool_large: *MemoryPool,
    tracker: *AllocationTracker,
    buddy: *BuddyAllocator,
    parent_allocator: std.mem.Allocator,
    tensor_count: usize,
    total_tensor_memory: usize,
    
    pub fn init(allocator: std.mem.Allocator) !*TensorMemoryManager {
        const manager = try allocator.create(TensorMemoryManager);
        errdefer allocator.destroy(manager);
        
        manager.arena = try Arena.init(allocator, 64 * 1024 * 1024);
        manager.pool_small = try MemoryPool.init(allocator, 1024, 16384);
        manager.pool_medium = try MemoryPool.init(allocator, 16384, 4096);
        manager.pool_large = try MemoryPool.init(allocator, 262144, 256);
        manager.tracker = try AllocationTracker.init(allocator);
        manager.buddy = try BuddyAllocator.init(allocator, 256 * 1024 * 1024, 4096);
        manager.parent_allocator = allocator;
        manager.tensor_count = 0;
        manager.total_tensor_memory = 0;
        
        return manager;
    }
    
    pub fn allocateTensor(self: *TensorMemoryManager, comptime T: type, shape: []const usize) ![]T {
        var total_elements: usize = 1;
        for (shape) |dim| {
            total_elements *= dim;
        }
        
        const size = total_elements * @sizeOf(T);
        self.total_tensor_memory += size;
        self.tensor_count += 1;
        
        if (size <= 1024) {
            const ptr = try self.pool_small.allocate();
            try self.tracker.trackAllocation(@intFromPtr(ptr), size, @alignOf(T));
            const typed: [*]T = @ptrCast(@alignCast(ptr));
            return typed[0..total_elements];
        } else if (size <= 16384) {
            const ptr = try self.pool_medium.allocate();
            try self.tracker.trackAllocation(@intFromPtr(ptr), size, @alignOf(T));
            const typed: [*]T = @ptrCast(@alignCast(ptr));
            return typed[0..total_elements];
        } else if (size <= 262144) {
            const ptr = try self.pool_large.allocate();
            try self.tracker.trackAllocation(@intFromPtr(ptr), size, @alignOf(T));
            const typed: [*]T = @ptrCast(@alignCast(ptr));
            return typed[0..total_elements];
        } else {
            const memory = try self.buddy.allocate(size);
            try self.tracker.trackAllocation(@intFromPtr(memory.ptr), size, @alignOf(T));
            const typed: [*]T = @ptrCast(@alignCast(memory.ptr));
            return typed[0..total_elements];
        }
    }
    
    pub fn freeTensor(self: *TensorMemoryManager, comptime T: type, tensor: []T) !void {
        const size = tensor.len * @sizeOf(T);
        const address = @intFromPtr(tensor.ptr);
        
        _ = try self.tracker.trackDeallocation(address);
        
        self.total_tensor_memory -= size;
        self.tensor_count -= 1;
        
        if (size <= 1024) {
            try self.pool_small.deallocate(@ptrCast(tensor.ptr));
        } else if (size <= 16384) {
            try self.pool_medium.deallocate(@ptrCast(tensor.ptr));
        } else if (size <= 262144) {
            try self.pool_large.deallocate(@ptrCast(tensor.ptr));
        } else {
            const byte_ptr: [*]u8 = @ptrCast(tensor.ptr);
            try self.buddy.deallocate(byte_ptr[0..size]);
        }
    }
    
    pub fn allocateTemporary(self: *TensorMemoryManager, comptime T: type, count: usize) ![]T {
        return self.arena.alloc(T, count);
    }
    
    pub fn resetTemporary(self: *TensorMemoryManager) void {
        self.arena.reset();
    }
    
    pub fn getStatistics(self: *TensorMemoryManager) struct {
        tensor_count: usize,
        tensor_memory: usize,
        tracker_stats: @TypeOf(self.tracker.getStatistics()),
    } {
        return .{
            .tensor_count = self.tensor_count,
            .tensor_memory = self.total_tensor_memory,
            .tracker_stats = self.tracker.getStatistics(),
        };
    }
    
    pub fn deinit(self: *TensorMemoryManager) void {
        self.arena.deinit();
        self.pool_small.deinit(self.parent_allocator);
        self.pool_medium.deinit(self.parent_allocator);
        self.pool_large.deinit(self.parent_allocator);
        self.tracker.deinit();
        self.buddy.deinit();
        self.parent_allocator.destroy(self);
    }
};

pub const GradientMemoryPool = struct {
    pools: [8]*MemoryPool,
    layer_allocations: std.AutoHashMap(usize, std.ArrayList(usize)),
    parent_allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) !*GradientMemoryPool {
        const pool = try allocator.create(GradientMemoryPool);
        errdefer allocator.destroy(pool);
        
        const sizes = [8]usize{ 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304 };
        
        for (&pool.pools, 0..) |*p, i| {
            const block_count = 16 * 1024 * 1024 / sizes[i];
            p.* = try MemoryPool.init(allocator, sizes[i], block_count);
        }
        
        pool.layer_allocations = std.AutoHashMap(usize, std.ArrayList(usize)).init(allocator);
        pool.parent_allocator = allocator;
        
        return pool;
    }
    
    fn findPool(self: *GradientMemoryPool, size: usize) ?usize {
        const sizes = [8]usize{ 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304 };
        for (sizes, 0..) |pool_size, i| {
            if (size <= pool_size) {
                _ = self;
                return i;
            }
        }
        return null;
    }
    
    pub fn allocateGradient(self: *GradientMemoryPool, layer_id: usize, size: usize) !*anyopaque {
        if (self.findPool(size)) |pool_idx| {
            const ptr = try self.pools[pool_idx].allocate();
            
            const result = try self.layer_allocations.getOrPut(layer_id);
            if (!result.found_existing) {
                result.value_ptr.* = std.ArrayList(usize).init(self.parent_allocator);
            }
            try result.value_ptr.append(@intFromPtr(ptr));
            
            return ptr;
        }
        
        return AllocationError.InvalidSize;
    }
    
    pub fn freeLayerGradients(self: *GradientMemoryPool, layer_id: usize) !void {
        if (self.layer_allocations.get(layer_id)) |addresses| {
            for (addresses.items) |address| {
                const ptr: *anyopaque = @ptrFromInt(address);
                
                for (&self.pools) |pool| {
                    pool.deallocate(ptr) catch continue;
                    break;
                }
            }
            
            var list = self.layer_allocations.fetchRemove(layer_id).?.value;
            list.deinit();
        }
    }
    
    pub fn deinit(self: *GradientMemoryPool) void {
        var iter = self.layer_allocations.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.layer_allocations.deinit();
        
        for (&self.pools) |pool| {
            pool.deinit(self.parent_allocator);
        }
        
        self.parent_allocator.destroy(self);
    }
};

pub const CacheAlignedAllocator = struct {
    backing_allocator: std.mem.Allocator,
    cache_line_size: usize,
    
    pub fn init(backing: std.mem.Allocator, cache_line_size: usize) CacheAlignedAllocator {
        return CacheAlignedAllocator{
            .backing_allocator = backing,
            .cache_line_size = cache_line_size,
        };
    }
    
    pub fn allocator(self: *CacheAlignedAllocator) std.mem.Allocator {
        return std.mem.Allocator{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }
    
    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *CacheAlignedAllocator = @ptrCast(@alignCast(ctx));
        _ = ret_addr;
        
        const alignment = @max(@as(usize, 1) << @intCast(ptr_align), self.cache_line_size);
        const aligned_len = (len + self.cache_line_size - 1) & ~(self.cache_line_size - 1);
        
        return self.backing_allocator.rawAlloc(aligned_len, @intCast(@ctz(alignment)), @returnAddress());
    }
    
    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *CacheAlignedAllocator = @ptrCast(@alignCast(ctx));
        _ = ret_addr;
        
        const aligned_new_len = (new_len + self.cache_line_size - 1) & ~(self.cache_line_size - 1);
        
        return self.backing_allocator.rawResize(buf, buf_align, aligned_new_len, @returnAddress());
    }
    
    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const self: *CacheAlignedAllocator = @ptrCast(@alignCast(ctx));
        _ = ret_addr;
        
        self.backing_allocator.rawFree(buf, buf_align, @returnAddress());
    }
};

pub const MemoryStatistics = struct {
    total_system_memory: usize,
    available_memory: usize,
    process_memory: usize,
    heap_size: usize,
    stack_size: usize,
    mapped_regions: usize,
    page_faults: usize,
    
    pub fn capture() MemoryStatistics {
        return MemoryStatistics{
            .total_system_memory = 0,
            .available_memory = 0,
            .process_memory = 0,
            .heap_size = 0,
            .stack_size = 0,
            .mapped_regions = 0,
            .page_faults = 0,
        };
    }
};

pub const MemoryManagerConfig = struct {
    arena_size: usize = 64 * 1024 * 1024,
    small_pool_block_size: usize = 1024,
    small_pool_block_count: usize = 16384,
    medium_pool_block_size: usize = 16384,
    medium_pool_block_count: usize = 4096,
    large_pool_block_size: usize = 262144,
    large_pool_block_count: usize = 256,
    buddy_total_size: usize = 256 * 1024 * 1024,
    buddy_min_block_size: usize = 4096,
    enable_tracking: bool = true,
    enable_leak_detection: bool = true,
    enable_bounds_checking: bool = true,
    cache_line_size: usize = 64,
};

pub const GlobalMemoryManager = struct {
    tensor_manager: *TensorMemoryManager,
    gradient_pool: *GradientMemoryPool,
    slab_allocator: *SlabAllocator,
    config: MemoryManagerConfig,
    parent_allocator: std.mem.Allocator,
    initialized: bool,
    
    var instance: ?*GlobalMemoryManager = null;
    var instance_lock: std.Thread.Mutex = .{};
    
    pub fn getInstance(allocator: std.mem.Allocator, config: MemoryManagerConfig) !*GlobalMemoryManager {
        instance_lock.lock();
        defer instance_lock.unlock();
        
        if (instance) |inst| {
            return inst;
        }
        
        const manager = try allocator.create(GlobalMemoryManager);
        errdefer allocator.destroy(manager);
        
        manager.tensor_manager = try TensorMemoryManager.init(allocator);
        manager.gradient_pool = try GradientMemoryPool.init(allocator);
        manager.slab_allocator = try SlabAllocator.init(allocator);
        manager.config = config;
        manager.parent_allocator = allocator;
        manager.initialized = true;
        
        instance = manager;
        return manager;
    }
    
    pub fn deinitInstance() void {
        instance_lock.lock();
        defer instance_lock.unlock();
        
        if (instance) |manager| {
            manager.tensor_manager.deinit();
            manager.gradient_pool.deinit();
            manager.slab_allocator.deinit();
            manager.parent_allocator.destroy(manager);
            instance = null;
        }
    }
};

test "memory pool basic allocation" {
    const allocator = std.testing.allocator;
    
    const pool = try MemoryPool.init(allocator, 64, 1024);
    defer pool.deinit(allocator);
    
    const ptr1 = try pool.allocate();
    const ptr2 = try pool.allocate();
    
    try std.testing.expect(@intFromPtr(ptr1) != @intFromPtr(ptr2));
    
    try pool.deallocate(ptr1);
    try pool.deallocate(ptr2);
}

test "arena allocation" {
    const allocator = std.testing.allocator;
    
    const arena = try Arena.init(allocator, 4096);
    defer arena.deinit();
    
    const ints = try arena.alloc(i32, 100);
    try std.testing.expect(ints.len == 100);
    
    const floats = try arena.alloc(f32, 50);
    try std.testing.expect(floats.len == 50);
    
    arena.reset();
    try std.testing.expect(arena.offset == 0);
}

test "buddy allocator" {
    const allocator = std.testing.allocator;
    
    const buddy = try BuddyAllocator.init(allocator, 1024 * 1024, 4096);
    defer buddy.deinit();
    
    const block1 = try buddy.allocate(8192);
    try std.testing.expect(block1.len >= 8192);
    
    const block2 = try buddy.allocate(4096);
    try std.testing.expect(block2.len >= 4096);
    
    try buddy.deallocate(block1);
    try buddy.deallocate(block2);
}

test "allocation tracker" {
    const allocator = std.testing.allocator;
    
    const tracker = try AllocationTracker.init(allocator);
    defer tracker.deinit();
    
    try tracker.trackAllocation(0x1000, 256, 16);
    try tracker.trackAllocation(0x2000, 512, 16);
    
    var stats = tracker.getStatistics();
    try std.testing.expect(stats.active_allocations == 2);
    try std.testing.expect(stats.current == 768);
    
    _ = try tracker.trackDeallocation(0x1000);
    stats = tracker.getStatistics();
    try std.testing.expect(stats.active_allocations == 1);
}

test "tensor memory manager" {
    const allocator = std.testing.allocator;
    
    const manager = try TensorMemoryManager.init(allocator);
    defer manager.deinit();
    
    const shape = [_]usize{ 32, 32 };
    const tensor = try manager.allocateTensor(f32, &shape);
    try std.testing.expect(tensor.len == 1024);
    
    try manager.freeTensor(f32, tensor);
}

test "slab allocator" {
    const allocator = std.testing.allocator;
    
    const slab = try SlabAllocator.init(allocator);
    defer slab.deinit();
    
    const ptr1 = try slab.allocate(32);
    const ptr2 = try slab.allocate(128);
    const ptr3 = try slab.allocate(1024);
    
    try slab.deallocate(ptr1, 32);
    try slab.deallocate(ptr2, 128);
    try slab.deallocate(ptr3, 1024);
}

test "gradient memory pool" {
    const allocator = std.testing.allocator;
    
    const pool = try GradientMemoryPool.init(allocator);
    defer pool.deinit();
    
    _ = try pool.allocateGradient(0, 256);
    _ = try pool.allocateGradient(0, 512);
    _ = try pool.allocateGradient(1, 1024);
    
    try pool.freeLayerGradients(0);
    try pool.freeLayerGradients(1);
}
