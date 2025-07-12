// RUN: sdfg-opt %s | FileCheck %s

// Basic single-dimension array.
// CHECK: %[[A:.*]] = sdfg.alloca : !sdfg.array<10 x f32>

// Nested two-dimensional array (outer length 5, inner length 10).
// CHECK: %[[B:.*]] = sdfg.alloca : !sdfg.array<5 x !sdfg.array<10 x f32>>

// Three-dimensional nested array.
// CHECK: %[[C:.*]] = sdfg.alloca : !sdfg.array<2 x !sdfg.array<3 x !sdfg.array<4 x i8>>>

module {
  %a = sdfg.alloca : !sdfg.array<10 x f32>
  %b = sdfg.alloca : !sdfg.array<5 x !sdfg.array<10 x f32>>
  %c = sdfg.alloca : !sdfg.array<2 x !sdfg.array<3 x !sdfg.array<4 x i8>>>
} 