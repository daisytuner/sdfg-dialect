// RUN: sdfg-opt %s | FileCheck %s

// CHECK: sdfg.sdfg @empty
sdfg.sdfg @empty attributes {num_args = 0 : i32} {
  ^bb0:
}
