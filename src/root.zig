const std = @import("std");

pub fn gravitySort32(rows: *[32]u32) void {
    inline for (0..32) |row| {
        rows[row] = @truncate((@as(u64, 1) << @as(u6, @intCast(@popCount(rows[row])))) - 1);
    }
}

/// Naive transpose for testing - transposes bit (row, col) to bit (col, row)
fn naiveTranspose32x32(A: *[32]u32) void {
    var temp: [32]u32 = [_]u32{0} ** 32;

    // For each row
    for (0..32) |row| {
        // For each column (bit position, MSB=0, LSB=31)
        for (0..32) |col| {
            // Get bit at (row, col)
            const bit = (A[row] >> @intCast(31 - col)) & 1;
            // Set bit at (col, row)
            temp[col] |= bit << @intCast(31 - row);
        }
    }

    A.* = temp;
}

/// Transpose a 32x32 bit matrix using the bitwise hack algorithm (loop version)
/// Based on Hacker's Delight by Henry S. Warren, Jr.
/// Bit ordering: bit 0 is the MSB (leftmost), bit 31 is the LSB (rightmost)
/// Input: array where each u32 represents a row of 32 bits
/// Output: array where each u32 represents what was previously a column
pub fn transpose32x32Loop(A: *[32]u32) void {
    var j: u32 = 16;
    var m: u32 = 0x0000FFFF;
    var k: u32 = 0;

    while (j != 0) {
        k = 0;
        while (k < 32) {
            const t = (A[k] ^ (A[k+j] >> @intCast(j))) & m;
            A[k] ^= t;
            A[k+j] ^= t << @intCast(j);
            k = (k + j + 1) & ~j;
        }
        j >>= 1;
        m ^= m << @intCast(j);
    }
}

/// Straight-line unrolled transpose for maximum performance
/// Based on Hacker's Delight straight-line version
inline fn swap(a: *u32, b: *u32, j: u5, m: u32) void {
    const t = (a.* ^ (b.* >> j)) & m;
    a.* ^= t;
    b.* ^= t << j;
}

pub inline fn transpose32x32Unrolled(A: *[32]u32) void {
    // Stage 1: j=16, m=0x0000FFFF
    // k sequence: 0, 16
    const m1: u32 = 0x0000FFFF;
    inline for (0..16) |i| swap(&A[i], &A[i+16], 16, m1);

    // Stage 2: j=8, m=0x00FF00FF
    // k sequence: 0, 8, 16, 24
    const m2: u32 = 0x00FF00FF;
    inline for (0..8) |i| {
        swap(&A[i], &A[i+8], 8, m2);
        swap(&A[i+16], &A[i+24], 8, m2);
    }

    // Stage 3: j=4, m=0x0F0F0F0F
    // k sequence: 0, 4, 8, 12, 16, 20, 24, 28
    const m3: u32 = 0x0F0F0F0F;
    inline for (0..4) |i| {
        swap(&A[i], &A[i+4], 4, m3);
        swap(&A[i+8], &A[i+12], 4, m3);
        swap(&A[i+16], &A[i+20], 4, m3);
        swap(&A[i+24], &A[i+28], 4, m3);
    }

    // Stage 4: j=2, m=0x33333333
    // k sequence: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    const m4: u32 = 0x33333333;
    inline for (0..2) |i| {
        swap(&A[i], &A[i+2], 2, m4);
        swap(&A[i+4], &A[i+6], 2, m4);
        swap(&A[i+8], &A[i+10], 2, m4);
        swap(&A[i+12], &A[i+14], 2, m4);
        swap(&A[i+16], &A[i+18], 2, m4);
        swap(&A[i+20], &A[i+22], 2, m4);
        swap(&A[i+24], &A[i+26], 2, m4);
        swap(&A[i+28], &A[i+30], 2, m4);
    }

    // Stage 5: j=1, m=0x55555555
    // k sequence: 0, 1, 2, 3, ..., 31
    const m5: u32 = 0x55555555;
    inline for (0..16) |i| {
        swap(&A[i*2], &A[i*2+1], 1, m5);
    }
}

/// Default transpose - uses unrolled version for best performance
pub inline fn transpose32x32(A: *[32]u32) void {
    transpose32x32Unrolled(A);
}

// ============================================================================
// 256x256 TRANSPOSE FOR U8 VALUES
// ============================================================================

/// Naive transpose for testing - transposes 256x256 bit matrix
fn naiveTranspose256x256(A: *[256]u256) void {
    var temp: [256]u256 = [_]u256{0} ** 256;

    for (0..256) |row| {
        for (0..256) |col| {
            const bit = (A[row] >> @intCast(255 - col)) & 1;
            temp[col] |= bit << @intCast(255 - row);
        }
    }

    A.* = temp;
}

/// Transpose a 256x256 bit matrix using the bitwise hack algorithm (loop version)
pub fn transpose256x256Loop(A: *[256]u256) void {
    var j: u32 = 128;
    var m: u256 = (1 << 128) - 1; // 128 ones on the right
    var k: u32 = 0;

    while (j != 0) {
        k = 0;
        while (k < 256) {
            const t = (A[k] ^ (A[k+j] >> @intCast(j))) & m;
            A[k] ^= t;
            A[k+j] ^= t << @intCast(j);
            k = (k + j + 1) & ~j;
        }
        j >>= 1;
        m ^= m << @intCast(j);
    }
}

/// Swap helper for 256x256 transpose
inline fn swap256(a: *u256, b: *u256, j: u8, m: u256) void {
    const t = (a.* ^ (b.* >> j)) & m;
    a.* ^= t;
    b.* ^= t << j;
}

/// Semi-unrolled transpose for 256x256 matrix (8 stages)
/// Note: Fully unrolling causes excessive compile times, so we use hybrid approach
pub fn transpose256x256Unrolled(A: *[256]u256) void {
    // Just use the loop version for now - full unrolling is too expensive for 256x256
    transpose256x256Loop(A);
}

/// Default 256x256 transpose - uses loop version due to size
pub inline fn transpose256x256(A: *[256]u256) void {
    transpose256x256Loop(A);
}

/// Bead sort for an array of u8 values (0-255)
/// Uses 256x256 matrix with u256 rows
/// Maximum 256 elements supported
pub inline fn beadSort256(values: []u8) void {
    if (values.len == 0) return;
    std.debug.assert(values.len <= 256);

    var matrix: [256]u256 = [_]u256{0} ** 256;

    // All ones mask for u256
    const all_ones: u256 = ~@as(u256, 0);

    // Step 1: Convert to unary (left-aligned)
    for (values, 0..) |val, i| {
        if (val > 0) {
            const shift = @as(u9, 256) - @as(u9, val);
            matrix[i] = all_ones << @intCast(shift);
        }
    }

    // Step 2: Transpose
    transpose256x256(&matrix);

    // Step 3: Apply gravity
    for (0..256) |i| {
        const count = @popCount(matrix[i]);
        if (count > 0) {
            const shift = @as(u9, 256) - @as(u9, count);
            matrix[i] = all_ones << @intCast(shift);
        } else {
            matrix[i] = 0;
        }
    }

    // Step 4: Transpose back
    transpose256x256(&matrix);

    // Step 5: Extract sorted values (reverse order for ascending)
    for (values, 0..) |*val, i| {
        val.* = @intCast(@popCount(matrix[values.len - 1 - i]));
    }
}

// ============================================================================
// HYBRID SORTING ALGORITHMS
// ============================================================================

/// Hybrid merge sort that uses bead sort for small subarrays
pub fn hybridMergeSort(values: []u5, threshold: usize) void {
    if (values.len <= 1) return;

    if (values.len <= threshold) {
        beadSort(values);
        return;
    }

    const mid = values.len / 2;
    hybridMergeSort(values[0..mid], threshold);
    hybridMergeSort(values[mid..], threshold);

    // Merge
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

/// Hybrid quicksort that uses bead sort for small subarrays
pub fn hybridQuickSort(values: []u5, threshold: usize) void {
    if (values.len <= 1) return;

    if (values.len <= threshold) {
        beadSort(values);
        return;
    }

    // Partition using median-of-three
    const pivot_idx = medianOfThree(values);
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

    // Recursively sort partitions
    if (i > 0) hybridQuickSort(values[0..i], threshold);
    if (i + 1 < values.len) hybridQuickSort(values[i + 1..], threshold);
}

fn medianOfThree(values: []u5) usize {
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

/// Hybrid introsort-style algorithm: quicksort with bead sort for small arrays
/// and fallback to heap sort for deep recursion
pub fn hybridIntroSort(values: []u5, threshold: usize) void {
    const max_depth = 2 * std.math.log2_int(usize, values.len + 1);
    introSortImpl(values, threshold, max_depth);
}

fn introSortImpl(values: []u5, threshold: usize, depth_limit: usize) void {
    if (values.len <= 1) return;

    if (values.len <= threshold) {
        beadSort(values);
        return;
    }

    if (depth_limit == 0) {
        // Fallback to heap sort (use std.sort.heap)
        std.sort.heap(u5, values, {}, std.sort.asc(u5));
        return;
    }

    // Quicksort partition
    const pivot_idx = medianOfThree(values);
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

    if (i > 0) introSortImpl(values[0..i], threshold, depth_limit - 1);
    if (i + 1 < values.len) introSortImpl(values[i + 1..], threshold, depth_limit - 1);
}

/// Bead sort for an array of u5 values (0-31)
/// Uses bitwise matrix transpose to simulate gravity
/// Maximum 32 elements supported (32x32 matrix)
pub inline fn beadSort(values: []u5) void {
    if (values.len == 0) return;
    std.debug.assert(values.len <= 32); // Assert it's a short list

    // Initialize 32x32 matrix with zeros
    // Each row can hold up to 32 beads (bits)
    // We have 32 rows to support up to 32 values
    var matrix: [32]u32 = [_]u32{0} ** 32;

    // Step 1: Convert each u5 value to unary (base-1) representation
    // Value N becomes N ones on the LEFT (MSB side): e.g., 5 → 0b11111000...
    // This is because the transpose treats bit 0 as the leftmost bit
    for (values, 0..) |val, i| {
        if (val > 0) {
            // Create N ones on the left: shift left from all 1s
            // val is u5 (0-31), so 32-val is 1-32
            const shift = @as(u6, 32) - @as(u6, val);
            matrix[i] = std.math.shl(u32, 0xFFFFFFFF, shift);
        }
    }

    // Step 2: Transpose the matrix (rows become columns)
    transpose32x32(&matrix);

    // Step 3: Apply gravity by counting beads in each column (now row)
    // and converting back to unary (left-aligned)
    for (0..32) |i| {
        const count = @popCount(matrix[i]);
        if (count > 0) {
            // Create count ones on the left
            const shift = @as(u6, 32) - @as(u6, count);
            matrix[i] = std.math.shl(u32, 0xFFFFFFFF, shift);
        } else {
            matrix[i] = 0;
        }
    }

    // Step 4: Transpose back (columns become rows again)
    transpose32x32(&matrix);

    // Step 5: Extract sorted values using popcount
    // Note: Due to gravity pulling down, values are in descending order
    // We need to reverse to get ascending order
    for (values, 0..) |*val, i| {
        val.* = @intCast(@popCount(matrix[values.len - 1 - i]));
    }
}

// ============================================================================
// TESTS
// ============================================================================

const testing = std.testing;

test "transpose32x32 vs naive - identity matrix" {
    var matrix1: [32]u32 = [_]u32{0} ** 32;
    var matrix2: [32]u32 = [_]u32{0} ** 32;

    // Create identity matrix (diagonal of 1s, MSB-side)
    inline for (0..32) |i| {
        const val = @as(u32, 1) << @intCast(31 - i);
        matrix1[i] = val;
        matrix2[i] = val;
    }

    transpose32x32(&matrix1);
    naiveTranspose32x32(&matrix2);

    try testing.expectEqualSlices(u32, &matrix2, &matrix1);
}

test "transpose32x32 vs naive - all ones in row 0" {
    var matrix1: [32]u32 = [_]u32{0} ** 32;
    var matrix2: [32]u32 = [_]u32{0} ** 32;

    matrix1[0] = 0xFFFFFFFF;
    matrix2[0] = 0xFFFFFFFF;

    transpose32x32(&matrix1);
    naiveTranspose32x32(&matrix2);

    try testing.expectEqualSlices(u32, &matrix2, &matrix1);
}

test "transpose32x32 vs naive - random pattern" {
    var matrix1: [32]u32 = undefined;
    var matrix2: [32]u32 = undefined;

    for (0..32) |i| {
        const val = @as(u32, @intCast(i * 0x01010101));
        matrix1[i] = val;
        matrix2[i] = val;
    }

    transpose32x32(&matrix1);
    naiveTranspose32x32(&matrix2);

    try testing.expectEqualSlices(u32, &matrix2, &matrix1);
}

test "benchmark - transpose implementations" {
    const iterations = 50000;

    std.debug.print("\n\n=== TRANSPOSE BENCHMARK ===\n", .{});
    std.debug.print("Testing {d} iterations\n\n", .{iterations});

    // Test data
    var test_matrix: [32]u32 = undefined;
    for (0..32) |i| test_matrix[i] = @as(u32, @intCast(i * 0x01010101));

    // Benchmark loop version
    var matrix1 = test_matrix;
    const loop_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        matrix1 = test_matrix;
        transpose32x32Loop(&matrix1);
    }
    const loop_end = std.time.nanoTimestamp();
    const loop_total = loop_end - loop_start;
    const loop_avg = @divTrunc(loop_total, iterations);

    // Benchmark unrolled version
    var matrix2 = test_matrix;
    const unrolled_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        matrix2 = test_matrix;
        transpose32x32Unrolled(&matrix2);
    }
    const unrolled_end = std.time.nanoTimestamp();
    const unrolled_total = unrolled_end - unrolled_start;
    const unrolled_avg = @divTrunc(unrolled_total, iterations);

    // Benchmark naive version (for reference)
    var matrix3 = test_matrix;
    const naive_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        matrix3 = test_matrix;
        naiveTranspose32x32(&matrix3);
    }
    const naive_end = std.time.nanoTimestamp();
    const naive_total = naive_end - naive_start;
    const naive_avg = @divTrunc(naive_total, iterations);

    // Verify all produce same result
    matrix1 = test_matrix;
    matrix2 = test_matrix;
    matrix3 = test_matrix;
    transpose32x32Loop(&matrix1);
    transpose32x32Unrolled(&matrix2);
    naiveTranspose32x32(&matrix3);
    try testing.expectEqualSlices(u32, &matrix1, &matrix2);
    try testing.expectEqualSlices(u32, &matrix1, &matrix3);

    std.debug.print("{s: <30} {d: >10} ns/iter  (total: {d: >12} ns)\n", .{"naiveTranspose (O(n²))", naive_avg, naive_total});
    std.debug.print("{s: <30} {d: >10} ns/iter  (total: {d: >12} ns)", .{"transpose32x32Loop", loop_avg, loop_total});

    if (loop_avg < naive_avg) {
        const speedup = @divTrunc(naive_avg * 100, loop_avg);
        std.debug.print("  ✓ {d}.{d:0>2}x faster than naive\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
    } else {
        std.debug.print("\n", .{});
    }

    std.debug.print("{s: <30} {d: >10} ns/iter  (total: {d: >12} ns)", .{"transpose32x32Unrolled", unrolled_avg, unrolled_total});

    if (unrolled_avg < loop_avg) {
        const speedup = @divTrunc(loop_avg * 100, unrolled_avg);
        std.debug.print("  ✓ {d}.{d:0>2}x faster than loop\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
    } else {
        const slowdown = @divTrunc(unrolled_avg * 100, loop_avg);
        std.debug.print("  ({d}.{d:0>2}x slower than loop)\n", .{@divTrunc(slowdown, 100), @rem(slowdown, 100)});
    }

    if (unrolled_avg < naive_avg) {
        const speedup = @divTrunc(naive_avg * 100, unrolled_avg);
        std.debug.print("{s: <30} ✓ Unrolled is {d}.{d:0>2}x faster than naive\n", .{"", @divTrunc(speedup, 100), @rem(speedup, 100)});
    }

    std.debug.print("\n=== END TRANSPOSE BENCHMARK ===\n\n", .{});
}

test "transpose32x32 - double transpose returns original" {
    var matrix: [32]u32 = undefined;
    // Create a test pattern
    for (0..32) |i| {
        matrix[i] = @as(u32, @intCast(i * 0x01010101));
    }

    const original = matrix;
    transpose32x32(&matrix);
    transpose32x32(&matrix);

    try testing.expectEqualSlices(u32, &original, &matrix);
}

test "beadSort - empty array" {
    var values: [0]u5 = undefined;
    beadSort(&values);
    // Should not crash
}

test "beadSort - single element" {
    var values = [_]u5{15};
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{15}, &values);
}

test "beadSort - two elements sorted" {
    var values = [_]u5{5, 10};
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{5, 10}, &values);
}

test "beadSort - two elements unsorted" {
    var values = [_]u5{10, 5};
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{5, 10}, &values);
}

test "beadSort - all zeros" {
    var values = [_]u5{ 0, 0, 0, 0, 0 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 0, 0, 0, 0 }, &values);
}

test "beadSort - all same value" {
    var values = [_]u5{ 7, 7, 7, 7, 7 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 7, 7, 7, 7, 7 }, &values);
}

test "beadSort - already sorted" {
    var values = [_]u5{ 1, 2, 3, 4, 5 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 2, 3, 4, 5 }, &values);
}

test "beadSort - reverse sorted" {
    var values = [_]u5{ 5, 4, 3, 2, 1 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 2, 3, 4, 5 }, &values);
}

test "beadSort - random order small" {
    var values = [_]u5{ 3, 1, 4, 1, 5, 9, 2, 6 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 1, 2, 3, 4, 5, 6, 9 }, &values);
}

test "beadSort - with zeros" {
    var values = [_]u5{ 5, 0, 3, 0, 1, 0, 2 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 0, 0, 1, 2, 3, 5 }, &values);
}

test "beadSort - maximum values" {
    var values = [_]u5{ 31, 31, 31 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 31, 31, 31 }, &values);
}

test "beadSort - mix with max values" {
    var values = [_]u5{ 31, 0, 15, 1, 30, 2 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 1, 2, 15, 30, 31 }, &values);
}

test "beadSort - full range" {
    var values = [_]u5{ 10, 20, 5, 15, 25, 0, 30, 31, 1, 29 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 1, 5, 10, 15, 20, 25, 29, 30, 31 }, &values);
}

test "beadSort - 16 elements" {
    var values = [_]u5{ 15, 8, 23, 4, 16, 12, 31, 2, 19, 7, 25, 11, 3, 28, 1, 20 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 2, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 25, 28, 31 }, &values);
}

test "beadSort - 32 elements (maximum)" {
    var values = [_]u5{
        15, 23, 8,  19, 31, 4,  12, 27, 2,  16, 9,
        25, 6,  18, 29, 1,  14, 22, 7,  20, 30, 3,
        11, 26, 5,  17, 28, 0,  13, 24, 10, 21,
    };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    }, &values);
}

test "beadSort - duplicates throughout" {
    var values = [_]u5{ 5, 3, 5, 1, 3, 5, 1, 3, 1 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 1, 1, 3, 3, 3, 5, 5, 5 }, &values);
}

test "beadSort - mostly duplicates" {
    var values = [_]u5{ 7, 7, 7, 7, 7, 7, 3, 7, 7, 7, 15, 7 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 15 }, &values);
}

test "beadSort - alternating pattern" {
    var values = [_]u5{ 1, 31, 1, 31, 1, 31, 1, 31 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 1, 1, 1, 31, 31, 31, 31 }, &values);
}

test "beadSort - sequential with gaps" {
    var values = [_]u5{ 20, 10, 30, 0, 5, 15, 25 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 0, 5, 10, 15, 20, 25, 30 }, &values);
}

test "beadSort - powers of two" {
    var values = [_]u5{ 16, 1, 8, 4, 2 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 2, 4, 8, 16 }, &values);
}

test "beadSort - stability check with metadata" {
    // While bead sort is not stable in general, we can verify it sorts correctly
    var values = [_]u5{ 5, 5, 5, 3, 3, 3 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 3, 3, 3, 5, 5, 5 }, &values);
}

test "beadSort - worst case for gravity (all descending)" {
    var values = [_]u5{ 20, 19, 18, 17, 16, 15, 14, 13, 12, 11 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 }, &values);
}

test "beadSort - prime numbers" {
    var values = [_]u5{ 29, 13, 17, 2, 23, 19, 11, 7, 5, 3, 31 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 }, &values);
}

test "beadSort - fibonacci-like sequence" {
    var values = [_]u5{ 21, 8, 13, 5, 3, 2, 1, 1 };
    beadSort(&values);
    try testing.expectEqualSlices(u5, &[_]u5{ 1, 1, 2, 3, 5, 8, 13, 21 }, &values);
}

test "beadSort - 32 elements (maximum capacity)" {
    var values: [32]u5 = undefined;
    // Fill with reverse order
    for (0..32) |i| {
        values[i] = @intCast(31 - i);
    }

    beadSort(&values);

    // Verify sorted 0..31
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
    beadSort(&values);

    // Verify sorted
    try testing.expectEqualSlices(u5, &[_]u5{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    }, &values);
}

test "beadSort - performance with 32 random values" {
    var values: [32]u5 = undefined;
    // Create a known pattern for consistent testing
    for (0..32) |i| {
        values[i] = @intCast((i * 37) % 32); // Pseudo-random pattern
    }

    const start = std.time.nanoTimestamp();
    beadSort(&values);
    const end = std.time.nanoTimestamp();

    // Verify it's sorted
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
        hybridMergeSort(&values, threshold);

        for (0..values.len - 1) |i| {
            try testing.expect(values[i] <= values[i + 1]);
        }
    }
}

test "hybridQuickSort - correctness" {
    const thresholds = [_]usize{ 4, 8, 16, 32 };

    for (thresholds) |threshold| {
        var values = [_]u5{ 15, 8, 23, 4, 16, 12, 31, 2, 19, 7, 25, 11, 3, 28, 1, 20 };
        hybridQuickSort(&values, threshold);

        for (0..values.len - 1) |i| {
            try testing.expect(values[i] <= values[i + 1]);
        }
    }
}

test "hybridIntroSort - correctness" {
    const thresholds = [_]usize{ 4, 8, 16, 32 };

    for (thresholds) |threshold| {
        var values = [_]u5{ 15, 8, 23, 4, 16, 12, 31, 2, 19, 7, 25, 11, 3, 28, 1, 20 };
        hybridIntroSort(&values, threshold);

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
        std.debug.print("{s: <50} {s: >15}\n", .{"Algorithm", "Avg (ns)"});
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

        std.debug.print("{s: <50} {d: >15}\n", .{"std.mem.sort (pdqsort)", std_avg});

        // Benchmark hybrid merge sort with various thresholds
        for (thresholds) |threshold| {
            const start = std.time.nanoTimestamp();
            for (0..iterations) |_| {
                @memcpy(values, test_data);
                hybridMergeSort(values, threshold);
            }
            const end = std.time.nanoTimestamp();
            const avg = @divTrunc(end - start, iterations);

            var label_buf: [50]u8 = undefined;
            const label = std.fmt.bufPrint(&label_buf, "hybridMergeSort (threshold={d})", .{threshold}) catch unreachable;
            std.debug.print("{s: <50} {d: >15}", .{label, avg});

            if (avg < std_avg) {
                const speedup = @divTrunc(std_avg * 100, avg);
                std.debug.print("  ✓ {d}.{d:0>2}x faster\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
            } else {
                std.debug.print("\n", .{});
            }
        }

        // Benchmark hybrid quicksort with various thresholds
        for (thresholds) |threshold| {
            const start = std.time.nanoTimestamp();
            for (0..iterations) |_| {
                @memcpy(values, test_data);
                hybridQuickSort(values, threshold);
            }
            const end = std.time.nanoTimestamp();
            const avg = @divTrunc(end - start, iterations);

            var label_buf: [50]u8 = undefined;
            const label = std.fmt.bufPrint(&label_buf, "hybridQuickSort (threshold={d})", .{threshold}) catch unreachable;
            std.debug.print("{s: <50} {d: >15}", .{label, avg});

            if (avg < std_avg) {
                const speedup = @divTrunc(std_avg * 100, avg);
                std.debug.print("  ✓ {d}.{d:0>2}x faster\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
            } else {
                std.debug.print("\n", .{});
            }
        }

        // Benchmark hybrid introsort with various thresholds
        for (thresholds) |threshold| {
            const start = std.time.nanoTimestamp();
            for (0..iterations) |_| {
                @memcpy(values, test_data);
                hybridIntroSort(values, threshold);
            }
            const end = std.time.nanoTimestamp();
            const avg = @divTrunc(end - start, iterations);

            var label_buf: [50]u8 = undefined;
            const label = std.fmt.bufPrint(&label_buf, "hybridIntroSort (threshold={d})", .{threshold}) catch unreachable;
            std.debug.print("{s: <50} {d: >15}", .{label, avg});

            if (avg < std_avg) {
                const speedup = @divTrunc(std_avg * 100, avg);
                std.debug.print("  ✓ {d}.{d:0>2}x faster\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
            } else {
                std.debug.print("\n", .{});
            }
        }

        std.debug.print("\n", .{});
    }

    std.debug.print("=== END HYBRID TUNING ===\n\n", .{});
}

/// Non-inlined version for benchmark comparison
fn beadSortNoInline(values: []u5) void {
    if (values.len == 0) return;
    std.debug.assert(values.len <= 32);

    var matrix: [32]u32 = [_]u32{0} ** 32;

    for (values, 0..) |val, i| {
        if (val > 0) {
            const shift = @as(u6, 32) - @as(u6, val);
            matrix[i] = std.math.shl(u32, 0xFFFFFFFF, shift);
        }
    }

    transpose32x32Loop(&matrix);

    for (0..32) |i| {
        const count = @popCount(matrix[i]);
        if (count > 0) {
            const shift = @as(u6, 32) - @as(u6, count);
            matrix[i] = std.math.shl(u32, 0xFFFFFFFF, shift);
        } else {
            matrix[i] = 0;
        }
    }

    transpose32x32Loop(&matrix);

    for (values, 0..) |*val, i| {
        val.* = @intCast(@popCount(matrix[values.len - 1 - i]));
    }
}

test "benchmark - beadSort optimization impact" {
    const iterations = 20000;

    std.debug.print("\n\n=== BEADSORT OPTIMIZATION BENCHMARK ===\n", .{});
    std.debug.print("Testing {d} iterations on 32-element arrays\n\n", .{iterations});

    var test_data: [32]u5 = undefined;
    for (0..32) |i| test_data[i] = @intCast((i * 37) % 32);

    // Benchmark non-inlined version with loop transpose
    var values1 = test_data;
    const noinline_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        values1 = test_data;
        beadSortNoInline(&values1);
    }
    const noinline_end = std.time.nanoTimestamp();
    const noinline_total = noinline_end - noinline_start;
    const noinline_avg = @divTrunc(noinline_total, iterations);

    // Benchmark inlined version with unrolled transpose
    var values2 = test_data;
    const inline_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        values2 = test_data;
        beadSort(&values2);
    }
    const inline_end = std.time.nanoTimestamp();
    const inline_total = inline_end - inline_start;
    const inline_avg = @divTrunc(inline_total, iterations);

    // Verify both produce same result
    values1 = test_data;
    values2 = test_data;
    beadSortNoInline(&values1);
    beadSort(&values2);
    try testing.expectEqualSlices(u5, &values1, &values2);

    std.debug.print("{s: <40} {d: >8} ns/iter  (total: {d: >12} ns)\n", .{"beadSort (no inline + loop)", noinline_avg, noinline_total});
    std.debug.print("{s: <40} {d: >8} ns/iter  (total: {d: >12} ns)", .{"beadSort (inline + unrolled)", inline_avg, inline_total});

    if (inline_avg < noinline_avg) {
        const speedup = @divTrunc(noinline_avg * 100, inline_avg);
        std.debug.print("  ✓ {d}.{d:0>2}x faster\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
        const saved = noinline_avg - inline_avg;
        std.debug.print("{s: <40} Optimization saves {d} ns per sort\n", .{"", saved});
    } else {
        std.debug.print("\n", .{});
    }

    std.debug.print("\n=== END OPTIMIZATION BENCHMARK ===\n\n", .{});
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
        }},
        .{ .name = "Sorted", .data = blk: {
            var arr: [32]u5 = undefined;
            for (0..32) |i| arr[i] = @intCast(i);
            break :blk arr;
        }},
        .{ .name = "Reverse", .data = blk: {
            var arr: [32]u5 = undefined;
            for (0..32) |i| arr[i] = @intCast(31 - i);
            break :blk arr;
        }},
        .{ .name = "AllSame", .data = [_]u5{15} ** 32 },
        .{ .name = "FewUnique", .data = blk: {
            var arr: [32]u5 = undefined;
            for (0..32) |i| arr[i] = @intCast(i % 4);
            break :blk arr;
        }},
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
            beadSort(&values1);
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
        beadSort(&values1);
        std.sort.insertion(u5, &values2, {}, std.sort.asc(u5));
        std.mem.sort(u5, &values3, {}, std.sort.asc(u5));
        try testing.expectEqualSlices(u5, &values1, &values2);
        try testing.expectEqualSlices(u5, &values1, &values3);

        // Calculate average per iteration
        const bead_avg = @divTrunc(bead_total, iterations);
        const insertion_avg = @divTrunc(insertion_total, iterations);
        const pdq_avg = @divTrunc(pdq_total, iterations);

        std.debug.print("Pattern: {s: <12}\n", .{pattern.name});
        std.debug.print("  beadSort:           {d: >6} ns/iter  (total: {d: >10} ns)\n", .{bead_avg, bead_total});
        std.debug.print("  std.sort.insertion: {d: >6} ns/iter  (total: {d: >10} ns)  ", .{insertion_avg, insertion_total});
        if (bead_avg < insertion_avg) {
            const speedup = @divTrunc(insertion_avg * 100, bead_avg);
            std.debug.print("{d}.{d:0>2}x slower\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
        } else {
            const speedup = @divTrunc(bead_avg * 100, insertion_avg);
            std.debug.print("{d}.{d:0>2}x faster\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
        }
        std.debug.print("  std.mem.sort (pdq): {d: >6} ns/iter  (total: {d: >10} ns)  ", .{pdq_avg, pdq_total});
        if (bead_avg < pdq_avg) {
            const speedup = @divTrunc(pdq_avg * 100, bead_avg);
            std.debug.print("{d}.{d:0>2}x slower\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
        } else {
            const speedup = @divTrunc(bead_avg * 100, pdq_avg);
            std.debug.print("{d}.{d:0>2}x faster\n", .{@divTrunc(speedup, 100), @rem(speedup, 100)});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("=== END BENCHMARK ===\n\n", .{});
}

// ============================================================================
// U8 TESTS AND BENCHMARKS
// ============================================================================

test "transpose256x256 vs naive - identity matrix" {
    var matrix1: [256]u256 = [_]u256{0} ** 256;
    var matrix2: [256]u256 = [_]u256{0} ** 256;

    // Create identity matrix (diagonal of 1s, MSB-side)
    inline for (0..256) |i| {
        const val = @as(u256, 1) << @intCast(255 - i);
        matrix1[i] = val;
        matrix2[i] = val;
    }

    transpose256x256(&matrix1);
    naiveTranspose256x256(&matrix2);

    try testing.expectEqualSlices(u256, &matrix2, &matrix1);
}

test "transpose256x256 vs naive - first row all ones" {
    var matrix1: [256]u256 = [_]u256{0} ** 256;
    var matrix2: [256]u256 = [_]u256{0} ** 256;

    const all_ones: u256 = ~@as(u256, 0);
    matrix1[0] = all_ones;
    matrix2[0] = all_ones;

    transpose256x256(&matrix1);
    naiveTranspose256x256(&matrix2);

    try testing.expectEqualSlices(u256, &matrix2, &matrix1);
}

test "transpose256x256 - double transpose returns original" {
    var matrix: [256]u256 = undefined;
    for (0..256) |i| {
        matrix[i] = @as(u256, @intCast(i)) * 0x0101010101010101;
    }

    const original = matrix;
    transpose256x256(&matrix);
    transpose256x256(&matrix);

    try testing.expectEqualSlices(u256, &original, &matrix);
}

test "beadSort256 - empty array" {
    var values: [0]u8 = undefined;
    beadSort256(&values);
}

test "beadSort256 - single element" {
    var values = [_]u8{127};
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{127}, &values);
}

test "beadSort256 - two elements" {
    var values = [_]u8{ 200, 50 };
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{ 50, 200 }, &values);
}

test "beadSort256 - all zeros" {
    var values = [_]u8{ 0, 0, 0, 0, 0 };
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0, 0 }, &values);
}

test "beadSort256 - all same value" {
    var values = [_]u8{ 128, 128, 128, 128 };
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{ 128, 128, 128, 128 }, &values);
}

test "beadSort256 - already sorted" {
    var values = [_]u8{ 10, 20, 30, 40, 50 };
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{ 10, 20, 30, 40, 50 }, &values);
}

test "beadSort256 - reverse sorted" {
    var values = [_]u8{ 250, 200, 150, 100, 50 };
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{ 50, 100, 150, 200, 250 }, &values);
}

test "beadSort256 - random order" {
    var values = [_]u8{ 100, 50, 200, 25, 150, 75, 225, 125 };
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{ 25, 50, 75, 100, 125, 150, 200, 225 }, &values);
}

test "beadSort256 - with zeros" {
    var values = [_]u8{ 100, 0, 50, 0, 200, 0, 150 };
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 50, 100, 150, 200 }, &values);
}

test "beadSort256 - maximum values" {
    var values = [_]u8{ 255, 255, 255 };
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{ 255, 255, 255 }, &values);
}

test "beadSort256 - full range" {
    var values = [_]u8{ 255, 0, 128, 64, 192, 32, 224, 96, 160 };
    beadSort256(&values);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 32, 64, 96, 128, 160, 192, 224, 255 }, &values);
}

test "beadSort256 - 64 elements" {
    var values: [64]u8 = undefined;
    for (0..64) |i| {
        values[i] = @intCast((i * 73) % 256);
    }

    beadSort256(&values);

    // Verify sorted
    for (0..63) |i| {
        try testing.expect(values[i] <= values[i + 1]);
    }
}

test "beadSort256 - 256 elements (maximum)" {
    var values: [256]u8 = undefined;
    for (0..256) |i| {
        values[i] = @intCast(255 - i);
    }

    beadSort256(&values);

    // Verify sorted 0..255
    for (0..256) |i| {
        try testing.expectEqual(@as(u8, @intCast(i)), values[i]);
    }
}

test "benchmark - transpose256x256 implementations" {
    const iterations = 1000;

    std.debug.print("\n\n=== TRANSPOSE 256x256 BENCHMARK ===\n", .{});
    std.debug.print("Testing {d} iterations\n\n", .{iterations});

    // Test data
    var test_matrix: [256]u256 = undefined;
    for (0..256) |i| test_matrix[i] = @as(u256, @intCast(i)) * 0x0101010101010101;

    // Benchmark loop version
    var matrix1 = test_matrix;
    const loop_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        matrix1 = test_matrix;
        transpose256x256Loop(&matrix1);
    }
    const loop_end = std.time.nanoTimestamp();
    const loop_total = loop_end - loop_start;
    const loop_avg = @divTrunc(loop_total, iterations);

    // Benchmark unrolled version
    var matrix2 = test_matrix;
    const unrolled_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        matrix2 = test_matrix;
        transpose256x256Unrolled(&matrix2);
    }
    const unrolled_end = std.time.nanoTimestamp();
    const unrolled_total = unrolled_end - unrolled_start;
    const unrolled_avg = @divTrunc(unrolled_total, iterations);

    // Benchmark naive version (fewer iterations - it's slow!)
    var matrix3 = test_matrix;
    const naive_iterations = 100;
    const naive_start = std.time.nanoTimestamp();
    for (0..naive_iterations) |_| {
        matrix3 = test_matrix;
        naiveTranspose256x256(&matrix3);
    }
    const naive_end = std.time.nanoTimestamp();
    const naive_total = naive_end - naive_start;
    const naive_avg = @divTrunc(naive_total, naive_iterations);

    // Verify all produce same result
    matrix1 = test_matrix;
    matrix2 = test_matrix;
    matrix3 = test_matrix;
    transpose256x256Loop(&matrix1);
    transpose256x256Unrolled(&matrix2);
    naiveTranspose256x256(&matrix3);
    try testing.expectEqualSlices(u256, &matrix1, &matrix2);
    try testing.expectEqualSlices(u256, &matrix1, &matrix3);

    std.debug.print("{s: <30} {d: >10} ns/iter  (total: {d: >12} ns)\n", .{ "naiveTranspose (O(n²))", naive_avg, naive_total });
    std.debug.print("{s: <30} {d: >10} ns/iter  (total: {d: >12} ns)", .{ "transpose256x256Loop", loop_avg, loop_total });

    if (loop_avg < naive_avg) {
        const speedup = @divTrunc(naive_avg * 100, loop_avg);
        std.debug.print("  ✓ {d}.{d:0>2}x faster than naive\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
    } else {
        std.debug.print("\n", .{});
    }

    std.debug.print("{s: <30} {d: >10} ns/iter  (total: {d: >12} ns)", .{ "transpose256x256Unrolled", unrolled_avg, unrolled_total });

    if (unrolled_avg < loop_avg) {
        const speedup = @divTrunc(loop_avg * 100, unrolled_avg);
        std.debug.print("  ✓ {d}.{d:0>2}x faster than loop\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
    } else {
        const slowdown = @divTrunc(unrolled_avg * 100, loop_avg);
        std.debug.print("  ({d}.{d:0>2}x slower than loop)\n", .{ @divTrunc(slowdown, 100), @rem(slowdown, 100) });
    }

    if (unrolled_avg < naive_avg) {
        const speedup = @divTrunc(naive_avg * 100, unrolled_avg);
        std.debug.print("{s: <30} ✓ Unrolled is {d}.{d:0>2}x faster than naive\n", .{ "", @divTrunc(speedup, 100), @rem(speedup, 100) });
    }

    std.debug.print("\n=== END TRANSPOSE 256x256 BENCHMARK ===\n\n", .{});
}

test "benchmark - beadSort256 vs std.mem.sort for u8 arrays" {
    const iterations = 1000;

    const patterns = [_]struct {
        name: []const u8,
        data: [64]u8,
    }{
        .{ .name = "Random", .data = blk: {
            var arr: [64]u8 = undefined;
            for (0..64) |i| arr[i] = @intCast((i * 73) % 256);
            break :blk arr;
        } },
        .{ .name = "Sorted", .data = blk: {
            var arr: [64]u8 = undefined;
            for (0..64) |i| arr[i] = @intCast(i * 4);
            break :blk arr;
        } },
        .{ .name = "Reverse", .data = blk: {
            var arr: [64]u8 = undefined;
            for (0..64) |i| arr[i] = @intCast(255 - i * 4);
            break :blk arr;
        } },
        .{ .name = "AllSame", .data = [_]u8{128} ** 64 },
        .{ .name = "FewUnique", .data = blk: {
            var arr: [64]u8 = undefined;
            for (0..64) |i| arr[i] = @intCast((i % 8) * 32);
            break :blk arr;
        } },
    };

    std.debug.print("\n\n=== BEADSORT256 BENCHMARK (64 elements) ===\n", .{});
    std.debug.print("Running {d} iterations per pattern\n\n", .{iterations});

    for (patterns) |pattern| {
        var values1: [64]u8 = pattern.data;
        var values2: [64]u8 = pattern.data;

        // Benchmark beadSort256
        const bead_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            values1 = pattern.data;
            beadSort256(&values1);
        }
        const bead_end = std.time.nanoTimestamp();
        const bead_total = bead_end - bead_start;

        // Benchmark std.mem.sort
        const std_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            values2 = pattern.data;
            std.mem.sort(u8, &values2, {}, std.sort.asc(u8));
        }
        const std_end = std.time.nanoTimestamp();
        const std_total = std_end - std_start;

        // Verify both produce same result
        beadSort256(&values1);
        std.mem.sort(u8, &values2, {}, std.sort.asc(u8));
        try testing.expectEqualSlices(u8, &values1, &values2);

        const bead_avg = @divTrunc(bead_total, iterations);
        const std_avg = @divTrunc(std_total, iterations);

        std.debug.print("Pattern: {s: <12}\n", .{pattern.name});
        std.debug.print("  beadSort256:        {d: >8} ns/iter  (total: {d: >12} ns)\n", .{ bead_avg, bead_total });
        std.debug.print("  std.mem.sort (pdq): {d: >8} ns/iter  (total: {d: >12} ns)  ", .{ std_avg, std_total });

        if (bead_avg < std_avg) {
            const speedup = @divTrunc(std_avg * 100, bead_avg);
            std.debug.print("{d}.{d:0>2}x slower\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
        } else {
            const speedup = @divTrunc(bead_avg * 100, std_avg);
            std.debug.print("{d}.{d:0>2}x faster\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("=== END BEADSORT256 BENCHMARK ===\n\n", .{});
}

test "benchmark - beadSort256 array size scaling" {
    const iterations = 500;
    const sizes = [_]usize{ 32, 64, 128, 256 };

    std.debug.print("\n\n=== BEADSORT256 SIZE SCALING ===\n", .{});
    std.debug.print("Testing {d} iterations per size\n\n", .{iterations});

    for (sizes) |size| {
        const test_data = std.heap.page_allocator.alloc(u8, size) catch unreachable;
        defer std.heap.page_allocator.free(test_data);
        for (0..size) |i| test_data[i] = @intCast((i * 73) % 256);

        const values1 = std.heap.page_allocator.alloc(u8, size) catch unreachable;
        defer std.heap.page_allocator.free(values1);

        const values2 = std.heap.page_allocator.alloc(u8, size) catch unreachable;
        defer std.heap.page_allocator.free(values2);

        // Benchmark beadSort256
        const bead_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            @memcpy(values1, test_data);
            beadSort256(values1);
        }
        const bead_end = std.time.nanoTimestamp();
        const bead_avg = @divTrunc(bead_end - bead_start, iterations);

        // Benchmark std.mem.sort
        const std_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            @memcpy(values2, test_data);
            std.mem.sort(u8, values2, {}, std.sort.asc(u8));
        }
        const std_end = std.time.nanoTimestamp();
        const std_avg = @divTrunc(std_end - std_start, iterations);

        std.debug.print("Size: {d: <4}  beadSort256: {d: >8} ns  |  std.mem.sort: {d: >8} ns  |  ", .{ size, bead_avg, std_avg });

        if (bead_avg < std_avg) {
            const speedup = @divTrunc(std_avg * 100, bead_avg);
            std.debug.print("✓ {d}.{d:0>2}x faster\n", .{ @divTrunc(speedup, 100), @rem(speedup, 100) });
        } else {
            const slowdown = @divTrunc(bead_avg * 100, std_avg);
            std.debug.print("{d}.{d:0>2}x slower\n", .{ @divTrunc(slowdown, 100), @rem(slowdown, 100) });
        }
    }

    std.debug.print("\n=== END SIZE SCALING ===\n\n", .{});
}
