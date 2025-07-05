// RUN: sdfg-opt %s | FileCheck %s

// CHECK: %0 = sdfg.library_node "add" ins %cst, %cst_0 : f64, f64 outs %cst_1 : f64 -> f64
sdfg.sdfg @empty attributes {num_args = 0 : i32} {
  sdfg.block {
    %a = arith.constant 2.0 : f64
    %b = arith.constant 3.0 : f64
    %result = arith.constant 0.0 : f64
    %0 = sdfg.library_node "add" ins %a, %b : f64, f64 outs %result : f64 -> f64
  }
}
