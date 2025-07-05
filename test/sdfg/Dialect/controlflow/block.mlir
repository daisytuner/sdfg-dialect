// RUN: sdfg-opt %s | FileCheck %s

// CHECK: sdfg.block {
sdfg.sdfg @empty attributes {num_args = 0 : i32} {
  sdfg.block {
    ^bb0:
  }
}
