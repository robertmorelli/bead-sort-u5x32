const std = @import("std");
const testing = std.testing;

pub fn gravity_sort32(rows: *[32]u32) void {
    inline for (0..32) |row| {
        rows[row] = @truncate((@as(u64, 1) << @as(u6, @intCast(@popCount(rows[row])))) - 1);
    }
}

inline fn swap(a: *u32, b: *u32, j: u5, m: u32) void {
    const t = (a.* ^ (b.* >> j)) & m;
    a.* ^= t;
    b.* ^= t << j;
}

pub inline fn transpose_32x32_unrolled(A: *[32]u32) void {
    const m1: u32 = 0x0000FFFF;
    inline for (0..16) |i| swap(&A[i], &A[i + 16], 16, m1);

    const m2: u32 = 0x00FF00FF;
    inline for (0..8) |i| {
        swap(&A[i], &A[i + 8], 8, m2);
        swap(&A[i + 16], &A[i + 24], 8, m2);
    }

    const m3: u32 = 0x0F0F0F0F;
    inline for (0..4) |i| {
        swap(&A[i], &A[i + 4], 4, m3);
        swap(&A[i + 8], &A[i + 12], 4, m3);
        swap(&A[i + 16], &A[i + 20], 4, m3);
        swap(&A[i + 24], &A[i + 28], 4, m3);
    }

    const m4: u32 = 0x33333333;
    inline for (0..2) |i| {
        swap(&A[i], &A[i + 2], 2, m4);
        swap(&A[i + 4], &A[i + 6], 2, m4);
        swap(&A[i + 8], &A[i + 10], 2, m4);
        swap(&A[i + 12], &A[i + 14], 2, m4);
        swap(&A[i + 16], &A[i + 18], 2, m4);
        swap(&A[i + 20], &A[i + 22], 2, m4);
        swap(&A[i + 24], &A[i + 26], 2, m4);
        swap(&A[i + 28], &A[i + 30], 2, m4);
    }

    const m5: u32 = 0x55555555;
    inline for (0..16) |i| {
        swap(&A[i * 2], &A[i * 2 + 1], 1, m5);
    }
}

pub fn hybrid_merge_sort(values: []u5, threshold: usize) void {
    if (values.len <= 1) return;

    if (values.len <= threshold) {
        bead_sort(values);
        return;
    }

    const mid = values.len / 2;
    hybrid_merge_sort(values[0..mid], threshold);
    hybrid_merge_sort(values[mid..], threshold);

    var temp = std.heap.page_allocator.alloc(u5, values.len) catch return;
    defer std.heap.page_allocator.free(temp);

    var i: usize = 0;
    var j: usize = mid;
    var k: usize = 0;

    while (i < mid and j < values.len) {
        if (values[i] <= values[j]) {
            temp[k] = values[i];
            i += 1;
        } else {
            temp[k] = values[j];
            j += 1;
        }
        k += 1;
    }

    while (i < mid) {
        temp[k] = values[i];
        i += 1;
        k += 1;
    }

    while (j < values.len) {
        temp[k] = values[j];
        j += 1;
        k += 1;
    }

    @memcpy(values, temp);
}

pub fn hybrid_quick_sort(values: []u5, threshold: usize) void {
    if (values.len <= 1) return;

    if (values.len <= threshold) {
        bead_sort(values);
        return;
    }

    const pivot_idx = median_of_three(values);
    std.mem.swap(u5, &values[pivot_idx], &values[values.len - 1]);
    const pivot = values[values.len - 1];

    var i: usize = 0;
    for (0..values.len - 1) |j| {
        if (values[j] < pivot) {
            std.mem.swap(u5, &values[i], &values[j]);
            i += 1;
        }
    }
    std.mem.swap(u5, &values[i], &values[values.len - 1]);

    if (i > 0) hybrid_quick_sort(values[0..i], threshold);
    if (i + 1 < values.len) hybrid_quick_sort(values[i + 1 ..], threshold);
}

fn median_of_three(values: []u5) usize {
    const len = values.len;
    const a = 0;
    const b = len / 2;
    const c = len - 1;

    if (values[a] <= values[b]) {
        if (values[b] <= values[c]) return b;
        if (values[a] <= values[c]) return c;
        return a;
    } else {
        if (values[a] <= values[c]) return a;
        if (values[b] <= values[c]) return c;
        return b;
    }
}

pub fn hybrid_intro_sort(values: []u5, threshold: usize) void {
    const max_depth = 2 * std.math.log2_int(usize, values.len + 1);
    intro_sort_impl(values, threshold, max_depth);
}

fn intro_sort_impl(values: []u5, threshold: usize, depth_limit: usize) void {
    if (values.len <= 1) return;

    if (values.len <= threshold) {
        bead_sort(values);
        return;
    }

    if (depth_limit == 0) {
        std.sort.heap(u5, values, {}, std.sort.asc(u5));
        return;
    }

    const pivot_idx = median_of_three(values);
    std.mem.swap(u5, &values[pivot_idx], &values[values.len - 1]);
    const pivot = values[values.len - 1];

    var i: usize = 0;
    for (0..values.len - 1) |j| {
        if (values[j] < pivot) {
            std.mem.swap(u5, &values[i], &values[j]);
            i += 1;
        }
    }
    std.mem.swap(u5, &values[i], &values[values.len - 1]);

    if (i > 0) intro_sort_impl(values[0..i], threshold, depth_limit - 1);
    if (i + 1 < values.len) intro_sort_impl(values[i + 1 ..], threshold, depth_limit - 1);
}

pub inline fn bead_sort(values: []u5) void {
    if (values.len == 0) return;
    std.debug.assert(values.len <= 32);
    var matrix: [32]u32 = [_]u32{0} ** 32;
    for (values, 0..) |val, i| {
        if (val > 0) {
            const shift = @as(u6, 32) - @as(u6, val);
            matrix[i] = std.math.shl(u32, 0xFFFFFFFF, shift);
        }
    }
    transpose_32x32_unrolled(&matrix);
    for (0..32) |i| {
        const count = @popCount(matrix[i]);
        if (count > 0) {
            const shift = @as(u6, 32) - @as(u6, count);
            matrix[i] = std.math.shl(u32, 0xFFFFFFFF, shift);
        } else {
            matrix[i] = 0;
        }
    }
    transpose_32x32_unrolled(&matrix);
    for (values, 0..) |*val, i| {
        val.* = @intCast(@popCount(matrix[values.len - 1 - i]));
    }
}

test "beadSort - empty array" {
    var values: [0]u5 = undefined;
    bead_sort(&values);
}

test "beadSort - single element" {
    var values = [_]u5{15};
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{15}, &values);
}

test "beadSort - two elements sorted" {
    var values = [_]u5{ 5, 10 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 5, 10 }, &values);
}

test "beadSort - two elements unsorted" {
    var values = [_]u5{ 10, 5 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 5, 10 }, &values);
}

test "beadSort - all zeros" {
    var values = [_]u5{ 0, 0, 0, 0, 0 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 0, 0, 0, 0 }, &values);
}

test "beadSort - all same value" {
    var values = [_]u5{ 7, 7, 7, 7, 7 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 7, 7, 7, 7, 7 }, &values);
}

test "beadSort - already sorted" {
    var values = [_]u5{ 1, 2, 3, 4, 5 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 2, 3, 4, 5 }, &values);
}

test "beadSort - reverse sorted" {
    var values = [_]u5{ 5, 4, 3, 2, 1 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 2, 3, 4, 5 }, &values);
}

test "beadSort - random order small" {
    var values = [_]u5{ 3, 1, 4, 1, 5, 9, 2, 6 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 1, 2, 3, 4, 5, 6, 9 }, &values);
}

test "beadSort - with zeros" {
    var values = [_]u5{ 5, 0, 3, 0, 1, 0, 2 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 0, 0, 1, 2, 3, 5 }, &values);
}

test "beadSort - maximum values" {
    var values = [_]u5{ 31, 31, 31 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 31, 31, 31 }, &values);
}

test "beadSort - mix with max values" {
    var values = [_]u5{ 31, 0, 15, 1, 30, 2 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 1, 2, 15, 30, 31 }, &values);
}

test "beadSort - full range" {
    var values = [_]u5{ 10, 20, 5, 15, 25, 0, 30, 31, 1, 29 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 1, 5, 10, 15, 20, 25, 29, 30, 31 }, &values);
}

test "beadSort - 16 elements" {
    var values = [_]u5{ 15, 8, 23, 4, 16, 12, 31, 2, 19, 7, 25, 11, 3, 28, 1, 20 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 2, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 25, 28, 31 }, &values);
}

test "beadSort - 32 elements (maximum)" {
    var values = [_]u5{
        15, 23, 8,  19, 31, 4,  12, 27, 2,  16, 9,
        25, 6,  18, 29, 1,  14, 22, 7,  20, 30, 3,
        11, 26, 5,  17, 28, 0,  13, 24, 10, 21,
    };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    }, &values);
}

test "beadSort - duplicates throughout" {
    var values = [_]u5{ 5, 3, 5, 1, 3, 5, 1, 3, 1 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 1, 1, 3, 3, 3, 5, 5, 5 }, &values);
}

test "beadSort - mostly duplicates" {
    var values = [_]u5{ 7, 7, 7, 7, 7, 7, 3, 7, 7, 7, 15, 7 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 15 }, &values);
}

test "beadSort - alternating pattern" {
    var values = [_]u5{ 1, 31, 1, 31, 1, 31, 1, 31 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 1, 1, 1, 31, 31, 31, 31 }, &values);
}

test "beadSort - sequential with gaps" {
    var values = [_]u5{ 20, 10, 30, 0, 5, 15, 25 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 5, 10, 15, 20, 25, 30 }, &values);
}

test "beadSort - powers of two" {
    var values = [_]u5{ 16, 1, 8, 4, 2 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 2, 4, 8, 16 }, &values);
}

test "beadSort - stability check with metadata" {
    // While bead sort is not stable in general, we can verify it sorts correctly
    var values = [_]u5{ 5, 5, 5, 3, 3, 3 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 3, 3, 3, 5, 5, 5 }, &values);
}

test "beadSort - worst case for gravity (all descending)" {
    var values = [_]u5{ 20, 19, 18, 17, 16, 15, 14, 13, 12, 11 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 }, &values);
}

test "beadSort - prime numbers" {
    var values = [_]u5{ 29, 13, 17, 2, 23, 19, 11, 7, 5, 3, 31 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 }, &values);
}

test "beadSort - fibonacci-like sequence" {
    var values = [_]u5{ 21, 8, 13, 5, 3, 2, 1, 1 };
    bead_sort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 1, 2, 3, 5, 8, 13, 21 }, &values);
}

test "beadSort - 32 elements (maximum capacity)" {
    var values: [32]u5 = undefined;
    // Fill with reverse order
    for (0..32) |i| {
        values[i] = @intCast(31 - i);
    }

    bead_sort(&values);

    for (0..32) |i| {
        try testing.expectEqual(@as(u5, @intCast(i)), values[i]);
    }
}

test "beadSort - 32 elements random pattern" {
    var values = [_]u5{
        15, 23, 8,  19, 31, 4,  12, 27, 2,  16, 9,
        25, 6,  18, 29, 1,  14, 22, 7,  20, 30, 3,
        11, 26, 5,  17, 28, 0,  13, 24, 10, 21,
    };
    bead_sort(&values);

    try testing.expectEqualSlices(u5, &[_]u5{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    }, &values);
}

test "beadSort - performance with 32 random values" {
    var values: [32]u5 = undefined;
    for (0..32) |i| {
        values[i] = @intCast((i * 37) % 32);
    }

    const start = std.time.nanoTimestamp();
    bead_sort(&values);
    const end = std.time.nanoTimestamp();

    for (0..31) |i| {
        try testing.expect(values[i] <= values[i + 1]);
    }

    const duration_ns = end - start;
    std.debug.print("\nBead sort (32 elements) took: {} ns\n", .{duration_ns});
}

test "hybridMergeSort - correctness" {
    const thresholds = [_]usize{ 4, 8, 16, 32 };

    for (thresholds) |threshold| {
        var values = [_]u5{ 15, 8, 23, 4, 16, 12, 31, 2, 19, 7, 25, 11, 3, 28, 1, 20 };
        hybrid_merge_sort(&values, threshold);

        for (0..values.len - 1) |i| {
            try testing.expect(values[i] <= values[i + 1]);
        }
    }
}

test "hybridQuickSort - correctness" {
    const thresholds = [_]usize{ 4, 8, 16, 32 };

    for (thresholds) |threshold| {
        var values = [_]u5{ 15, 8, 23, 4, 16, 12, 31, 2, 19, 7, 25, 11, 3, 28, 1, 20 };
        hybrid_quick_sort(&values, threshold);

        for (0..values.len - 1) |i| {
            try testing.expect(values[i] <= values[i + 1]);
        }
    }
}

test "hybridIntroSort - correctness" {
    const thresholds = [_]usize{ 4, 8, 16, 32 };

    for (thresholds) |threshold| {
        var values = [_]u5{ 15, 8, 23, 4, 16, 12, 31, 2, 19, 7, 25, 11, 3, 28, 1, 20 };
        hybrid_intro_sort(&values, threshold);

        for (0..values.len - 1) |i| {
            try testing.expect(values[i] <= values[i + 1]);
        }
    }
}

test "benchmark - hybrid algorithms with various thresholds" {
    const iterations = 5000;
    const array_sizes = [_]usize{ 32, 64, 128, 256 };
    const thresholds = [_]usize{ 4, 8, 16, 32 };

    std.debug.print("\n\n=== HYBRID ALGORITHM THRESHOLD TUNING ===\n", .{});
    std.debug.print("Testing {d} iterations per configuration\n\n", .{iterations});

    for (array_sizes) |size| {
        std.debug.print("Array size: {d}\n", .{size});
        std.debug.print("{s: <50} {s: >15}\n", .{ "Algorithm", "Avg (ns)" });
        std.debug.print("{s:-<70}\n", .{""});

        // Generate test data
        var test_data = std.heap.page_allocator.alloc(u5, size) catch unreachable;
        defer std.heap.page_allocator.free(test_data);
        for (0..size) |i| test_data[i] = @intCast((i * 37) % 32);

        // Benchmark std.mem.sort (baseline)
        const values = std.heap.page_allocator.alloc(u5, size) catch unreachable;
        defer std.heap.page_allocator.free(values);

        const std_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            @memcpy(values, test_data);
            std.mem.sort(u5, values, {}, std.sort.asc(u5));
        }
        const std_end = std.time.nanoTimestamp();
        const std_avg = @divTrunc(std_end - std_start, iterations);

        std.debug.print("{s: <50} {d: >15}\n", .{ "std.mem.sort (pdqsort)", std_avg });

        // Benchmark hybrid merge sort with various thresholds
        for (thresholds) |threshold| {
            const start = std.time.nanoTimestamp();
            for (0..iterations) |_| {
                @memcpy(values, test_data);
                hybrid_merge_sort(values, threshold);
            }
            const end = std.time.nanoTimestamp();
            const avg = @divTrunc(end - start, iterations);

            var label_buf: [50]u8 = undefined;
            const label = std.fmt.bufPrint(&label_buf, "hybridMergeSort (threshold={d})", .{threshold}) catch unreachable;
            std.debug.print("{s: <50} {d: >15}", .{ label, avg });

            if (avg < std_avg) {
                const speedup = @divTrunc(std_avg * 100, avg);
                std.debug.print("  ✓ {d}.{d:0>2}x faster\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
            } else {
                std.debug.print("\n", .{});
            }
        }

        for (thresholds) |threshold| {
            const start = std.time.nanoTimestamp();
            for (0..iterations) |_| {
                @memcpy(values, test_data);
                hybrid_quick_sort(values, threshold);
            }
            const end = std.time.nanoTimestamp();
            const avg = @divTrunc(end - start, iterations);

            var label_buf: [50]u8 = undefined;
            const label = std.fmt.bufPrint(&label_buf, "hybridQuickSort (threshold={d})", .{threshold}) catch unreachable;
            std.debug.print("{s: <50} {d: >15}", .{ label, avg });

            if (avg < std_avg) {
                const speedup = @divTrunc(std_avg * 100, avg);
                std.debug.print("  ✓ {d}.{d:0>2}x faster\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
            } else {
                std.debug.print("\n", .{});
            }
        }

        for (thresholds) |threshold| {
            const start = std.time.nanoTimestamp();
            for (0..iterations) |_| {
                @memcpy(values, test_data);
                hybrid_intro_sort(values, threshold);
            }
            const end = std.time.nanoTimestamp();
            const avg = @divTrunc(end - start, iterations);

            var label_buf: [50]u8 = undefined;
            const label = std.fmt.bufPrint(&label_buf, "hybridIntroSort (threshold={d})", .{threshold}) catch unreachable;
            std.debug.print("{s: <50} {d: >15}", .{ label, avg });

            if (avg < std_avg) {
                const speedup = @divTrunc(std_avg * 100, avg);
                std.debug.print("  ✓ {d}.{d:0>2}x faster\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
            } else {
                std.debug.print("\n", .{});
            }
        }

        std.debug.print("\n", .{});
    }

    std.debug.print("=== END HYBRID TUNING ===\n\n", .{});
}

test "benchmark - beadSort vs std.sort.insertion vs std.mem.sort" {
    const iterations = 10000;

    // Test data: various patterns
    const patterns = [_]struct {
        name: []const u8,
        data: [32]u5,
    }{
        .{ .name = "Random", .data = blk: {
            var arr: [32]u5 = undefined;
            for (0..32) |i| arr[i] = @intCast((i * 37) % 32);
            break :blk arr;
        } },
        .{ .name = "Sorted", .data = blk: {
            var arr: [32]u5 = undefined;
            for (0..32) |i| arr[i] = @intCast(i);
            break :blk arr;
        } },
        .{ .name = "Reverse", .data = blk: {
            var arr: [32]u5 = undefined;
            for (0..32) |i| arr[i] = @intCast(31 - i);
            break :blk arr;
        } },
        .{ .name = "AllSame", .data = [_]u5{15} ** 32 },
        .{ .name = "FewUnique", .data = blk: {
            var arr: [32]u5 = undefined;
            for (0..32) |i| arr[i] = @intCast(i % 4);
            break :blk arr;
        } },
    };

    std.debug.print("\n\n=== BENCHMARK RESULTS ===\n", .{});
    std.debug.print("Running {d} iterations per pattern\n\n", .{iterations});

    for (patterns) |pattern| {
        var values1: [32]u5 = pattern.data;
        var values2: [32]u5 = pattern.data;
        var values3: [32]u5 = pattern.data;

        // Benchmark beadSort
        const bead_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            values1 = pattern.data;
            bead_sort(&values1);
        }
        const bead_end = std.time.nanoTimestamp();
        const bead_total = bead_end - bead_start;

        // Benchmark std.sort.insertion
        const insertion_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            values2 = pattern.data;
            std.sort.insertion(u5, &values2, {}, std.sort.asc(u5));
        }
        const insertion_end = std.time.nanoTimestamp();
        const insertion_total = insertion_end - insertion_start;

        // Benchmark std.mem.sort (pdqsort)
        const pdq_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            values3 = pattern.data;
            std.mem.sort(u5, &values3, {}, std.sort.asc(u5));
        }
        const pdq_end = std.time.nanoTimestamp();
        const pdq_total = pdq_end - pdq_start;

        // Verify all produce same result
        bead_sort(&values1);
        std.sort.insertion(u5, &values2, {}, std.sort.asc(u5));
        std.mem.sort(u5, &values3, {}, std.sort.asc(u5));
        try testing.expectEqualSlices(u5, &values1, &values2);
        try testing.expectEqualSlices(u5, &values1, &values3);

        // Calculate average per iteration
        const bead_avg = @divTrunc(bead_total, iterations);
        const insertion_avg = @divTrunc(insertion_total, iterations);
        const pdq_avg = @divTrunc(pdq_total, iterations);

        std.debug.print("Pattern: {s: <12}\n", .{pattern.name});
        std.debug.print("  beadSort:           {d: >6} ns/iter  (total: {d: >10} ns)\n", .{ bead_avg, bead_total });
        std.debug.print("  std.sort.insertion: {d: >6} ns/iter  (total: {d: >10} ns)  ", .{ insertion_avg, insertion_total });
        if (bead_avg < insertion_avg) {
            const speedup = @divTrunc(insertion_avg * 100, bead_avg);
            std.debug.print("{d}.{d:0>2}x slower\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
        } else {
            const speedup = @divTrunc(bead_avg * 100, insertion_avg);
            std.debug.print("{d}.{d:0>2}x faster\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
        }
        std.debug.print("  std.mem.sort (pdq): {d: >6} ns/iter  (total: {d: >10} ns)  ", .{ pdq_avg, pdq_total });
        if (bead_avg < pdq_avg) {
            const speedup = @divTrunc(pdq_avg * 100, bead_avg);
            std.debug.print("{d}.{d:0>2}x slower\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
        } else {
            const speedup = @divTrunc(bead_avg * 100, pdq_avg);
            std.debug.print("{d}.{d:0>2}x faster\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("=== END BENCHMARK ===\n\n", .{});
}
