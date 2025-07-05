// RUN: sdfg-opt %s | FileCheck %s

// CHECK: %0:2 = sdfg.library_node "add2" ins %cst, %cst_0 : f64, f64 outs %cst_1, %cst_2 : f64, f64 -> f64, f64
sdfg.sdfg @empty attributes {num_args = 0 : i32} {
  sdfg.block {
    %a = arith.constant 2.0 : f64
    %b = arith.constant 3.0 : f64
    %result1 = arith.constant 0.0 : f64
    %result2 = arith.constant 0.0 : f64
    %0:2 = sdfg.library_node "add2" ins %a, %b : f64, f64 outs %result1, %result2 : f64, f64 -> f64, f64
  }
}
